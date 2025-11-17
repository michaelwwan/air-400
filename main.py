"""Training entry point for the respiration models."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from dataloaders.base_dataset import BaseDataset

try:
    import wandb
except Exception:
    wandb = None

from dataloaders.air125_dataset import AIR125Dataset
from dataloaders.air400_dataset import AIR400Dataset
from dataloaders.cohface_dataset import COHFACEDataset
from models.deep_phys import DeepPhys
from models.ts_can import TSCAN
from models.efficient_phys import EfficientPhys
from models.vire_net import VIRENet
from trainers.base_trainer import BaseTrainer


def set_random_seeds(seed: int = 42) -> None:
    """Seed numpy/torch/random for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed dataloader workers independently."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_logger(log_file_name: str, logs_dir: str) -> logging.Logger:
    """Configure console/file logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Avoid adding multiple handlers during multiple runs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Stream handler for console (INFO and above)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(stream_handler)

    # File handler (DEBUG and above)
    file_handler = logging.FileHandler(os.path.join(logs_dir, log_file_name), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(file_handler)

    return logger


def listify_datasets(config: Dict[str, Any]) -> None:
    """Ensure each split DATA entry is a list for easier iteration."""
    for split in ['TRAIN', 'VALID', 'TEST']:
        if not isinstance(config[split]['DATA'], list):
            config[split]['DATA'] = [config[split]['DATA']]


def handle_cross_validation(
    config: Dict[str, Any],
    logger: logging.Logger,
    train_size_pct: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Dict[str, Sequence[str]]]]]:
    """Handle cross validation for AIR125 or AIR400 dataset."""
    enable_cv = config.get('CV', {}).get('ENABLE_CV', False)
    if not enable_cv:
        return [config], []

    # Get all subjects in listed datasets
    cv_datasets = config.get('CV', {}).get('CV_DATASETS', [])
    all_air_subjects = get_all_air_subjects(config, cv_datasets, logger)

    cv_configs = []
    cv_split_dataset_dicts = []

    # Shuffle all AIR subjects
    np.random.shuffle(all_air_subjects)

    cv_mode = config.get('CV', {}).get('CV_MODE', 'train-valid')
    cv_fold_num = config.get('CV', {}).get('CV_FOLD_NUM', 0)
    cv_test_size = config.get('CV', {}).get('CV_TEST_SIZE', 0)

    test_subjects = None

    if cv_mode == 'train-valid':
        # K-fold cross-validation with a fixed test set
        test_subjects = all_air_subjects[:cv_test_size]  # Fixed subjects for testing
        cv_subjects = all_air_subjects[cv_test_size:]  # Remaining subjects for CV
    elif cv_mode == 'train-test':
        # K-fold cross-validation with 1 fold in train as validation set
        cv_subjects = all_air_subjects
    else:
        logger.error(f"Unsupported CV_MODE: {cv_mode}")
        raise ValueError(f"Unsupported CV_MODE: {cv_mode}")

    # CV Split into folds
    fold_size = len(cv_subjects) // cv_fold_num
    folds = [cv_subjects[i * fold_size:(i + 1) * fold_size] for i in range(cv_fold_num - 1)]
    folds.append(cv_subjects[(cv_fold_num - 1) * fold_size:])  # Handle remainder

    for idx in range(cv_fold_num):
        if cv_mode == 'train-valid':
            val_subjects = folds[idx]
            train_subjects = [s for i, fold in enumerate(folds) if i != idx for s in fold]

            cv_split_dataset_dict = {
                'TRAIN': get_cv_split_subjects(train_subjects),
                'VALID': get_cv_split_subjects(val_subjects),
                'TEST': get_cv_split_subjects(test_subjects)
            }

        elif cv_mode == 'train-test':
            test_subjects = folds[idx]
            train_subjects = [s for i, fold in enumerate(folds) if i != idx for s in fold]

            # Randomly pick 1 fold from train as validation
            val_size = max(1, len(train_subjects) // (cv_fold_num-1))
            val_indices = random.sample(range(len(train_subjects)), val_size)
            val_subjects = [train_subjects[i] for i in val_indices]
            train_subjects = [s for i, s in enumerate(train_subjects) if i not in val_indices]

            if train_size_pct and train_size_pct in [25, 50, 75]:
                logger.info(f"Subsetting {train_size_pct} percent of train set size. Original: {len(train_subjects)}")
                keep_n = max(1,int(round(len(train_subjects) * train_size_pct / 100.0)))
                train_subjects = random.sample(train_subjects, keep_n)
                logger.info(f"After: {len(train_subjects)}")

            cv_split_dataset_dict = {
                'TRAIN': get_cv_split_subjects(tuple(sorted(train_subjects))),
                'VALID': get_cv_split_subjects(tuple(sorted(val_subjects))),
                'TEST': get_cv_split_subjects(tuple(sorted(test_subjects)))
            }

        else:
            logger.error(f"Unsupported CV_MODE: {cv_mode}")
            raise ValueError(f"Unsupported CV_MODE: {cv_mode}")

        cv_split_dataset_dicts.append(cv_split_dataset_dict)

        # Config for current CV fold
        temp_config = copy.deepcopy(config)

        # Change datasets in temp_config to calculated CV datasets
        for split in cv_split_dataset_dict.keys():
            # Present datasets already in config file
            present_datasets = set(d['DATASET'] for d in temp_config[split]['DATA'])

            # Update CV split datasets
            updated_data_config = []
            for d_config in temp_config[split]['DATA']:
                if d_config['DATASET'] in cv_split_dataset_dict[split]:
                    # Update to CV subjects
                    d_config['SPLIT_TYPE'] = 'custom'
                    d_config['SPLIT_SUBJECTS'] = cv_split_dataset_dict[split][d_config['DATASET']]
                    updated_data_config.append(d_config)
                elif d_config['DATASET'] == 'COHFACE':
                    # Keep COHFACE untouched
                    updated_data_config.append(d_config)
                else:
                    continue  # Drop non-CV or non-COHFACE datasets

            # Add missing dataset from cv_split_dataset_dict
            for dataset_name in cv_split_dataset_dict[split]:
                if dataset_name not in present_datasets:
                    updated_data_config.append({
                        'DATASET': dataset_name,
                        'SPLIT_TYPE': 'custom',
                        'SPLIT_SUBJECTS': cv_split_dataset_dict[split][dataset_name],
                        'SPLIT_RATIO': 0.0,
                        'DATA_FORMAT': config[split]['DATA']['DATA_FORMAT'] if isinstance(config[split]['DATA'], dict) else config[split]['DATA'][0]['DATA_FORMAT'],
                        'PREPROCESS': copy.deepcopy(config[split]['DATA']['PREPROCESS']) if isinstance(config[split]['DATA'], dict) else copy.deepcopy(config[split]['DATA'][0]['PREPROCESS']),
                    })

            temp_config[split]['DATA'] = updated_data_config

        cv_configs.append(temp_config)

    return cv_configs, cv_split_dataset_dicts


def get_all_air_subjects(config: Dict[str, Any], cv_datasets: Sequence[str], logger: logging.Logger) -> List[str]:
    """Get all unique subject IDs in all listed datasets."""
    air125_path = config['DATA_PATH']['AIR_125']
    air400_path = config['DATA_PATH']['AIR_400']
    air125_subjects = AIR125Dataset.get_unique_subjects(AIR125Dataset.get_raw_data(air125_path))
    air400_subjects = AIR400Dataset.get_unique_subjects(AIR400Dataset.get_raw_data(air400_path))
    if 'AIR_125' in cv_datasets and 'AIR_400' in cv_datasets:
        return [f"AIR_125_{s}" for s in air125_subjects] + [f"AIR_400_{s}" for s in air400_subjects]
    elif 'AIR_125' in cv_datasets:
        return [f"AIR_125_{s}" for s in air125_subjects]
    elif 'AIR_400' in cv_datasets:
        return [f"AIR_400_{s}" for s in air400_subjects]
    else:
        logger.error(f"Unsupported CV datasets: {cv_datasets}")
        raise ValueError(f"Unsupported CV datasets: {cv_datasets}")


def get_cv_split_subjects(subjects: Sequence[str]) -> Dict[str, List[str]]:
    """Convert [Dataset_name]_[Subject ID] subjects to a map."""
    cv_split_subjects = defaultdict(list)
    for s in subjects:
        dataset_name = '_'.join(s.split('_')[:-1])
        subject_id = s.split('_')[-1]
        cv_split_subjects[dataset_name].append(subject_id)
    return cv_split_subjects


def isolate_config_for_dataset(config: Dict[str, Any], curr_dataset: str) -> Dict[str, Any]:
    """Extract config containing only the specified dataset per split."""
    config = copy.deepcopy(config)
    for split in ['TRAIN', 'VALID', 'TEST']:
        data_config = config[split]['DATA']
        filtered = [d for d in data_config if d['DATASET'] == curr_dataset]
        if filtered:
            config[split]['DATA'] = filtered[0]
        else:
            # Leave the DATA as it is
            config[split]['DATA'] = None
    return config


def get_split_datasets(
    config: Dict[str, Any],
    dataset_map: Dict[str, type[BaseDataset]],
) -> Dict[str, Optional[ConcatDataset]]:
    """Create datasets for train, validation, and test splits."""
    datasets = {}
    for split in ['TRAIN', 'VALID', 'TEST']:
        # Concat the list of datasets
        combined_dataset_list = []
        for d_config in config[split]['DATA']:
            dataset_name = d_config['DATASET']

            # Leave only one dataset in all splits before passing to dataloader
            temp_config = isolate_config_for_dataset(config, dataset_name)

            combined_dataset_list.append(
                # Init dataset with dataset_map
                dataset_map[dataset_name](temp_config, split)
            )
        # Concat the combined dataset list
        if combined_dataset_list:
            datasets[split] = ConcatDataset(combined_dataset_list)
        else:
            datasets[split] = None

    return datasets


def main() -> None:
    """CLI entry point for training/evaluating models."""
    SEED = 100
    set_random_seeds(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--preprocess', action='store_true', required=False, help='Enable preprocessing only mode')
    args = parser.parse_args()
    
    preprocess_only = args.preprocess
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logs_dir = os.path.join(config["DATA_PATH"]["OUTPUT_DIR"], "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger = setup_logger(f"{config['NAME']}_{datetime.now()}.log", logs_dir)

    dataset_map = {
        'AIR_125': AIR125Dataset,
        'AIR_400': AIR400Dataset,
        'COHFACE': COHFACEDataset
    }

    model_map = {
        'EfficientPhys': EfficientPhys,
        'VIRENet': VIRENet,
        'DeepPhys': DeepPhys,
        'TSCAN': TSCAN
    }

    listify_datasets(config)
    train_size_pct = config.get('CV', {}).get('TRAIN_SIZE_PCT', None)
    configs, cv_split_dataset_dicts = handle_cross_validation(config, logger, train_size_pct)
    all_test_metrics = defaultdict(list)

    enable_cv = False
    if len(configs) > 1:
        enable_cv = True

    for i, config in enumerate(configs):
        if preprocess_only:
            config['NAME'] = config['NAME'] + '_preprocess'
        elif enable_cv:
            config['NAME'] = config['NAME'] + f'_cv_{i}'
        
        # Init wandb (optional)
        use_wandb = bool(config.get('LOGGING', {}).get('USE_WANDB', False)) and (wandb is not None)
        if use_wandb:
            wandb.init(project="infant-respiration",
                       name=config['NAME'],
                       config=config)
        else:
            wandb.run = None

        if enable_cv:
            logger.info(f"\n======Current CV fold: {i}======")
            logger.info(f"Train subjects: {cv_split_dataset_dicts[i]['TRAIN']}")
            logger.info(f"Valid subjects: {cv_split_dataset_dicts[i]['VALID']}")
            logger.info(f"Test subjects: {cv_split_dataset_dicts[i]['TEST']}")

        # Re-sync data loader and preprocessing RNG
        set_random_seeds(SEED)

        datasets = get_split_datasets(config, dataset_map)

        # Create data loaders
        train_loader = DataLoader(
            datasets['TRAIN'],
            batch_size=config['TRAIN']['BATCH_SIZE'],
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(SEED)
        )

        val_loader = None
        if datasets['VALID'] is not None:
            val_loader = DataLoader(
                datasets['VALID'],
                batch_size=config['INFERENCE']['BATCH_SIZE'],
                shuffle=False,
                num_workers=10,
                worker_init_fn=seed_worker,
            )

        test_loader = None
        if datasets['TEST'] is not None:
            test_loader = DataLoader(
                datasets['TEST'],
                batch_size=config['INFERENCE']['BATCH_SIZE'],
                shuffle=False,
                num_workers=10,
                worker_init_fn=seed_worker,
            )

        if preprocess_only:
            logger.info('Preprocessing only mode. Completed successfully.')
            return

        # Re-sync training RNG
        set_random_seeds(SEED)

        # Initialize model
        model_name = config['MODEL']['NAME']
        frame_h = config['TRAIN']['DATA'][0]['PREPROCESS']['DOWNSAMPLE_SIZE_BEFORE_TRAINING'][1]
        in_channels = config['MODEL']['IN_CHANNELS']
        if model_name == 'DeepPhys':
            model = model_map[model_name](
                in_channels=in_channels,
                img_size=frame_h
            )
        else:
            model = model_map[model_name](
                in_channels=in_channels,
                frame_depth=config['MODEL']['FRAME_DEPTH'],
                img_size=frame_h
            )

        # Initialize trainer
        trainer = BaseTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )

        # Train the model
        trainer.train()

        # Test the model
        test_metrics = trainer.test()

        logger.info(f"Training and testing completed successfully.")
        if i < len(configs) - 1 and getattr(wandb, 'run', None):
            wandb.run.finish(exit_code=0)

        for key, value in test_metrics.items():
            all_test_metrics[key].append(value)

    all_test_metrics_stat = defaultdict(float)
    for key, metrics in all_test_metrics.items():
        all_test_metrics_stat[f'{key}_mean'] = np.mean(metrics)
        all_test_metrics_stat[f'{key}_std'] = np.std(metrics)

    logger.info('Statistics among all test results:')
    logger.info(all_test_metrics_stat)

    if getattr(wandb, 'run', None):
        wandb.log(all_test_metrics_stat)


if __name__ == "__main__":
    main()
