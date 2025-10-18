import logging
import os
from collections import defaultdict

import torch
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
try:
    import wandb
except Exception:
    wandb = None

from loss.negpearsonloss import Neg_Pearson
from loss.psdmse import PSD_MSE
from processors.post_processor import PostProcessor


class BaseTrainer:
    """Base trainer subclass for all models with wandb support."""

    def __init__(self, model, config, train_loader=None, val_loader=None, test_loader=None):
        """
        Initialize the trainer with model, config, and data loaders.

        Args:
            model: Models can be EfficientPhys, DeepPhys, TS-CAN, or VIRENet
            config: Configuration dictionary
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
        """

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self._load_params()

        self.min_valid_loss = None
        self.best_epoch = 0

        # Move model to device and wrap with DataParallel if multiple GPUs are available
        self.model = model.to(self.device)
        if self.num_of_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.num_of_gpu)))

        # Configure loss function
        self.criterion = defaultdict()
        self._setup_loss()

        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0,
            foreach=False,
            fused=False
        )

        # OneCycleLR scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.learning_rate,
            epochs=self.max_epoch_num,
            steps_per_epoch=self.num_train_batches)

        # Initialize post processor
        self.post_processor = PostProcessor()

        # Wandb starts watching model (optional)
        self.use_wandb = bool(self.config.get('LOGGING', {}).get('USE_WANDB', False)) and (wandb is not None)
        if self.use_wandb:
            wandb.watch(self.model)

    def _load_params(self):
        # Model related parameters
        self.frame_depth = self.config['MODEL']['FRAME_DEPTH']
        self.model_name = self.config['MODEL']['NAME']
        self.model_dir = os.path.join(self.config['DATA_PATH']['OUTPUT_DIR'], 'model', self.model_name, self.config['NAME'])

        # Train related parameters
        self.max_epoch_num = self.config['TRAIN']['EPOCHES']
        self.batch_size = self.config['TRAIN']['BATCH_SIZE']
        self.learning_rate = float(self.config['TRAIN']['LR'])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_of_gpu = torch.cuda.device_count()
        self.base_len = self.num_of_gpu * self.frame_depth

        self.num_train_batches = len(self.train_loader) if self.train_loader else 0

        # Pre-process parameters
        self.do_chunk = self.config['TRAIN']['DATA'][0]['PREPROCESS'].get('DO_CHUNK', True)
        self.chunk_len = self.config['TRAIN']['DATA'][0]['PREPROCESS'].get('CHUNK_LENGTH', 180) \
            if self.do_chunk else 0

        self.label_type = self.config['TRAIN']['DATA'][0]['PREPROCESS']['LABEL_NORMALIZE_TYPE']
        if self.label_type in ["Standardized", "Raw"]:
            self.diff_flag_test = False
        elif self.label_type == "DiffNormalized":
            self.diff_flag_test = True
        else:
            self.logger.error(f"Unsupported label type: {self.label_type}")
            raise ValueError(f"Unsupported label type: {self.label_type}")

        self.do_optical_flow = self.config['TRAIN']['DATA'][0]['PREPROCESS'].get('DO_OPTICAL_FLOW', False)
        self.in_channels = self.config['MODEL'].get('IN_CHANNELS', 3)

        # Post-process parameters
        self.use_post_process = self.config['INFERENCE'].get('USE_POST_PROCESS', False)
        self.eval_method = self.config['INFERENCE']['EVALUATION_METHOD']

    def _setup_loss(self):
        """Configure the loss function based on config."""
        loss_type = self.config.get('MODEL', {}).get('LOSS', 'psd_mse')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for d in self.config[split]['DATA']:
                if loss_type == 'psd_mse':
                    # Determine frequency range based on dataset
                    if d['DATASET'] in ['AIR_125', 'AIR_400', 'AIRFLOW']:
                        low, high = 0.3, 0.8
                    else:
                        low, high = 0.08, 0.5
                    self.criterion[d['DATASET']] = PSD_MSE(high_pass=high, low_pass=low)
                elif loss_type == 'neg_pearson':
                    self.criterion[d['DATASET']] = Neg_Pearson()
                else:
                    self.criterion[d['DATASET']] = torch.nn.MSELoss()

    def train(self):
        """Training routine for the model with NaN handling and gradient clipping."""
        if self.train_loader is None:
            self.logger.error("No data for training")
            raise ValueError("No data for training")

        # Set a gradient clipping value to prevent exploding gradients
        grad_clip_value = 1.0

        # Training loop in epoches
        for epoch in range(self.max_epoch_num):
            self.logger.info(f"\n====Training Epoch: {epoch}====")

            batch_losses = []
            batch_subj_counts = []
            nan_batches = 0

            self.model.train()

            # Training loop in batches
            tbar = tqdm(self.train_loader, ncols=80)
            for batch_idx, batch in enumerate(tbar):
                tbar.set_description(f"Train epoch {epoch}")

                subj_losses = []
                running_loss = 0.0

                data, labels, subj_indices, chunk_indices, dataset_names, fs_list = batch
                data = data.to(self.device)  # (N,D,6,H,W)
                labels = labels.to(self.device)

                # Check for NaN or Inf in input data
                if torch.isnan(data).any() or torch.isinf(data).any():
                    self.logger.debug(f"NaN/Inf detected in input data batch {batch_idx}, replacing with zeros")
                    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                if torch.isnan(labels).any() or torch.isinf(labels).any():
                    self.logger.debug(f"NaN/Inf detected in labels batch {batch_idx}, replacing with zeros")
                    labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)

                # DeepPhys and TSCAN require 6-channel input (3 diff, 3 raw), others require 2 or 3 channel frame or flow
                if self.model_name.lower() not in ["deepphys", "tscan"]:
                    if self.do_optical_flow:
                        data = data[:, :, :self.in_channels, ...]
                    else:
                        data = data[:, :, self.in_channels:, ...]

                    # Flatten for conv input shape
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)

                # Ensure data_i and labels_i length is divisible by base_len
                valid_len = (data.shape[0] // self.base_len) * self.base_len
                data = data[:valid_len]
                labels = labels[:valid_len]

                # Add one more frame for EfficientPhys since it does torch.diff for the input
                if self.model_name.lower().startswith("efficientphys"):
                    last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                    data = torch.cat((data, last_frame), 0)

                # Reset gradients before Forward pass
                self.optimizer.zero_grad()

                try:
                    # Forward pass
                    pred = self.model(data)

                    # Check for NaN in model output
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                        self.logger.debug(f"NaN/Inf in model output batch {batch_idx}, replacing with zeros")
                        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

                    # Aggregate batch data by subject
                    subj_predictions, subj_labels, dataset_subject_map, fs_subject_map = self._aggregate_batch_by_subject(
                        pred, labels, subj_indices, dataset_names, fs_list
                    )

                    for subj in subj_predictions:
                        pred_subj = torch.cat(subj_predictions[subj], dim=0)
                        labels_subj = torch.cat(subj_labels[subj], dim=0)
                        ds_name = dataset_subject_map[subj]
                        fs = fs_subject_map[subj]

                        # Calculate subject level loss
                        loss_subj = self._calculate_subject_loss(pred_subj, labels_subj, ds_name, fs)

                        # Check if loss is valid
                        if torch.isnan(loss_subj) or torch.isinf(loss_subj):
                            self.logger.debug(f"NaN/Inf loss in dataset {ds_name} subject {pred_subj}, skipping")
                            continue

                        subj_losses.append(loss_subj)

                    if len(subj_losses) == 0:
                        self.logger.debug(f"All dataset losses are invalid in batch {batch_idx}, skipping batch")
                        nan_batches += 1
                        batch_losses.append(0.1)
                        tbar.set_postfix(loss=0.1, nan_batches=nan_batches)
                        continue

                    # Backward pass
                    batch_total_loss = sum(subj_losses)
                    batch_total_loss.backward()

                    # Update model for each batch
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_value)  # Clip gradient
                    self.optimizer.step()  # Update model weights
                    self.scheduler.step()  # Update learning rate

                    # Update batch metrics
                    batch_losses.append(batch_total_loss.item())
                    batch_subj_counts.append(len(subj_losses))
                    running_loss += batch_total_loss.item()

                    # Print progress every 100 batches
                    if batch_idx % 100 == 99:
                        self.logger.info(f'[{epoch}, {batch_idx + 1:5d}] avg loss: {(sum(batch_losses[-100:]) / sum(batch_subj_counts[-100:])):.3f}, NaN batches: {nan_batches}')
                        running_loss = 0
                    tbar.set_postfix(loss=running_loss, nan_batches=nan_batches)

                except Exception as e:
                    if "nan" in str(e).lower() or "inf" in str(e).lower():
                        self.logger.exception(f"Error in batch {batch_idx}")
                        nan_batches += 1
                        tbar.set_postfix(nan_batches=nan_batches)
                    else:
                        raise

            # Calculate average training loss for the epoch (excluding NaN batches)
            batch_losses, batch_subj_counts = zip(*[
                (loss, count) for loss, count in zip(batch_losses, batch_subj_counts)
                if not (np.isnan(loss) or np.isinf(loss))
            ])
            epoch_loss = sum(batch_losses) / sum(batch_subj_counts) if batch_losses else 0.1

            # Log metrics to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "nan_batches": nan_batches,
                    "learning_rate": self.scheduler.get_last_lr()[0]
                })

            # Save model checkpoint
            self._save_model(epoch)

            # Validation
            if self.val_loader is not None:
                valid_loss = self.validate()
                self.logger.info(f'Validation loss: {valid_loss}')

                # Log validation loss to wandb
                if self.use_wandb:
                    wandb.log({"val_loss": valid_loss})

                # Track best model
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    self.logger.info(f"Update best model! Best epoch: {self.best_epoch}")

                    # Save best model
                    best_model_path = os.path.join(self.model_dir, f"{self.model_name}_best.pth")
                    torch.save(self.model.state_dict(), best_model_path)
                    if self.use_wandb:
                        wandb.save(best_model_path)

        self.logger.info(f"Best trained epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss}")

    def validate(self):
        """Validation routine for the model."""
        if self.val_loader is None:
            self.logger.error("No data for validation")
            raise ValueError("No data for validation")

        self.logger.info("\n===Validating===")

        all_predictions = defaultdict(list)
        all_labels = defaultdict(list)
        all_dataset_subject_map = defaultdict(str)
        all_fs_subject_map = defaultdict(int)

        all_losses = []
        self.model.eval()

        with torch.no_grad():
            vbar = tqdm(self.val_loader, ncols=80)
            for batch_idx, batch in enumerate(vbar):
                vbar.set_description("Validation")

                # Get data and labels from batch
                data, labels, subj_indices, chunk_indices, dataset_names, fs_list = batch
                data = data.to(self.device)
                labels = labels.to(self.device)

                # DeepPhys and TSCAN require 6-channel input (3 diff, 3 raw), others require 2 or 3 channel frame or flow
                if self.model_name.lower() not in ["deepphys", "tscan"]:
                    if self.do_optical_flow:
                        data = data[:, :, :self.in_channels, ...]
                    else:
                        data = data[:, :, self.in_channels:, ...]

                # Flatten for conv input shape
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)

                # Ensure data and labels length is divisible by base_len
                valid_len = (data.shape[0] // self.base_len) * self.base_len
                data = data[:valid_len]
                labels = labels[:valid_len]

                # Add one more frame for EfficientPhys since it does torch.diff for the input
                if self.model_name.lower().startswith("efficientphys"):
                    last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                    data = torch.cat((data, last_frame), 0)

                # Forward pass
                pred = self.model(data)

                # Aggregate batch data by subject
                subj_predictions, subj_labels, dataset_subject_map, fs_subject_map = self._aggregate_batch_by_subject(
                    pred, labels, subj_indices, dataset_names, fs_list
                )

                for subj in subj_predictions:
                    all_predictions[subj].extend(subj_predictions[subj])
                    all_labels[subj].extend(subj_labels[subj])
                    all_dataset_subject_map[subj] = dataset_subject_map[subj]
                    all_fs_subject_map[subj] = fs_subject_map[subj]

        for subj in all_predictions:
            pred_subj = torch.cat(all_predictions[subj], dim=0)
            labels_subj = torch.cat(all_labels[subj], dim=0)
            ds_name = all_dataset_subject_map[subj]
            fs = all_fs_subject_map[subj]

            # Calculate subject level loss
            loss_subj = self._calculate_subject_loss(pred_subj, labels_subj, ds_name, fs)

            if torch.isnan(loss_subj) or torch.isinf(loss_subj):
                self.logger.debug(f"NaN/Inf loss for dataset {ds_name} subject {subj}, skipping")
                continue
            all_losses.append(loss_subj.item())

        # Calculate average validation loss
        return np.mean(all_losses)

    def test(self):
        """Testing routine for the model."""
        if self.test_loader is None:
            self.logger.error("No data for testing")
            raise ValueError("No data for testing")

        self.logger.info("\n===Testing===")
        all_predictions = defaultdict(list)
        all_labels = defaultdict(list)
        all_dataset_subject_map = defaultdict(str)
        all_fs_subject_map = defaultdict(int)

        # Load the best model for testing
        best_model_path = os.path.join(self.model_dir, f"{self.model_name}_best.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            self.logger.info(f"Testing using best model from: {best_model_path}")
        else:
            self.logger.info("Best model not found, using current model weights")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
                # Get data and labels from batch
                data, labels, subj_indices, chunk_indices, dataset_names, fs_list = batch
                data = data.to(self.device)
                labels = labels.to(self.device)

                # DeepPhys and TSCAN require 6-channel input (3 diff, 3 raw), others require 2 or 3 channel frame or flow
                if self.model_name.lower() not in ["deepphys", "tscan"]:
                    if self.do_optical_flow:
                        data = data[:, :, :self.in_channels, ...]
                    else:
                        data = data[:, :, self.in_channels:, ...]

                # Flatten for conv input shape
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)

                # Ensure data and labels length is divisible by base_len
                valid_len = (data.shape[0] // self.base_len) * self.base_len
                data = data[:valid_len]
                labels = labels[:valid_len]

                # Add one more frame for EfficientPhys since it does torch.diff for the input
                if self.model_name.lower().startswith("efficientphys"):
                    last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                    data = torch.cat((data, last_frame), 0)

                # Forward pass
                pred = self.model(data)

                # Aggregate batch data by subject
                subj_predictions, subj_labels, dataset_subject_map, fs_subject_map = self._aggregate_batch_by_subject(
                    pred, labels, subj_indices, dataset_names, fs_list
                )

                for subj in subj_predictions:
                    all_predictions[subj].extend(subj_predictions[subj])
                    all_labels[subj].extend(subj_labels[subj])
                    all_dataset_subject_map[subj] = dataset_subject_map[subj]
                    all_fs_subject_map[subj] = fs_subject_map[subj]

        # Apply post-processing and collect final metrics
        final_predictions = {}
        final_labels = {}

        for subj in all_predictions:
            pred_subj = torch.cat(all_predictions[subj], dim=0)
            labels_subj = torch.cat(all_labels[subj], dim=0)
            ds_name = all_dataset_subject_map[subj]
            fs = all_fs_subject_map[subj]

            assert fs > 0, f"Invalid frame rate {fs} for subject {subj}"

            if self.use_post_process:
                pred_subj, labels_subj = self.post_processor.post_process(
                    pred_subj, labels_subj,
                    fs=fs,
                    diff_flag=self.diff_flag_test,
                    infant_flag=ds_name in ['AIR_125', 'AIR_400'],
                    use_bandpass=True,
                    eval_method=self.eval_method
                )
                pred_subj = torch.tensor(pred_subj, dtype=torch.float32, device=self.device).unsqueeze(0)
                labels_subj = torch.tensor(labels_subj, dtype=torch.float32, device=self.device).unsqueeze(0)

            final_predictions[subj] = {0: pred_subj}
            final_labels[subj] = {0: labels_subj}

        # Calculate metrics
        test_metrics = self._calculate_metrics(final_predictions, final_labels)

        # Log test metrics to wandb
        if self.use_wandb:
            wandb.log(test_metrics)

        self.logger.info("\nTest Results:")
        for key, value in test_metrics.items():
            self.logger.info(f"{key}: {value:.6f}")

        return test_metrics

    def _calculate_subject_loss(self, pred_subj, labels_subj, ds_name, fs):
        """Handle different loss function requirements and calculate subject level loss."""
        criterion = self.criterion[ds_name]
        if isinstance(criterion, PSD_MSE):
            pred_subj = pred_subj.view(1, -1)  # N=1 for each subject
            labels_subj = labels_subj.view(1, -1)
        return criterion(pred_subj, labels_subj, fs)

    def _aggregate_batch_by_subject(self, pred, labels, subj_indices, dataset_names, fs_list):
        subj_predictions = defaultdict(list)
        subj_labels = defaultdict(list)
        dataset_subject_map = defaultdict(str)
        fs_subject_map = defaultdict(int)

        for i in range(len(subj_indices)):
            subj_index = subj_indices[i]  # filename
            ds_name = dataset_names[i]
            pred_i = pred[i * self.chunk_len:(i + 1) * self.chunk_len]
            labels_i = labels[i * self.chunk_len:(i + 1) * self.chunk_len]
            fs = fs_list[i]

            subj_predictions[subj_index].append(pred_i)
            subj_labels[subj_index].append(labels_i)
            dataset_subject_map[subj_index] = ds_name
            fs_subject_map[subj_index] = fs

        return subj_predictions, subj_labels, dataset_subject_map, fs_subject_map

    def _calculate_metrics(self, predictions, labels):
        """
        Calculate evaluation metrics for rPPG predictions, including RMSE.

        Args:
            predictions: Dictionary of predictions by subject and chunk
            labels: Dictionary of ground truth labels by subject and chunk

        Returns:
            Dictionary of metrics including MSE, RMSE, and MAE
        """
        # Collect all predictions and labels
        all_preds = []
        all_labels = []

        for subj in predictions:
            for chunk in predictions[subj]:
                all_preds.append(predictions[subj][chunk].cpu().numpy())
                all_labels.append(labels[subj][chunk].cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculate metrics
        mse = np.mean((all_preds - all_labels) ** 2)
        rmse = np.sqrt(mse)  # Root Mean Square Error
        mae = np.mean(np.abs(all_preds - all_labels))

        if len(all_preds) <= 1 or np.std(all_preds) < 1e-6 or np.std(all_labels) < 1e-6:
            pearson_r = 0.0
        else:
            pearson_r, _ = pearsonr(all_preds.flatten(), all_labels.flatten())

        return {
            "test_mse": mse,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_pearson_r": pearson_r
        }

    def _save_model(self, epoch):
        """Save model checkpoint."""
        os.makedirs(self.model_dir, exist_ok=True)

        model_path = os.path.join(self.model_dir, f"{self.model_name}_Epoch{epoch}.pth")
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f'Saved Model Path: {model_path}')

        # Save to wandb
        if self.use_wandb:
            wandb.save(model_path)
