import argparse
import json
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.DeepPhys import DeepPhys
from models.TS_CAN import TSCAN
from models.efficientphys import EfficientPhys
from models.virenetRGB import VIRENet
from dataloaders.inference_dataset import InferenceDataset
from processors.post_processor import PostProcessor
from main import set_random_seeds, setup_logger, seed_worker


def _get_test_split_cfg(config):
    split_cfg = config.get('TEST', {})
    data = split_cfg.get('DATA') if isinstance(split_cfg, dict) else None
    if data is None:
        raise KeyError(f"Config missing TEST.DATA")
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Config TEST.DATA list is empty")
        return data[0]
    if isinstance(data, dict):
        return data
    raise TypeError(f"Unexpected type for TEST.DATA: {type(data)}")


# def _process_signal(pp: PostProcessor, sig: np.ndarray, fs: float, diff_flag: bool, infant_flag: bool, use_bandpass: bool):
#     fs = float(fs)
#     if diff_flag:
#         sig = pp._detrend_signal(np.cumsum(sig), 100)
#     else:
#         sig = pp._detrend_signal(sig, 100)
#     if use_bandpass:
#         low, high = (0.3, 1.0) if infant_flag else (0.08, 0.5)
#         sig = pp._bandpass_filter(sig, fs, low, high)
#     return sig


# def _save_waveform(out_dir: str, base: str, fs: float, pred_seq: np.ndarray, label_seq: np.ndarray | None):
#     os.makedirs(out_dir, exist_ok=True)
#     t = np.arange(len(pred_seq)) / float(fs)
#     csv_path = os.path.join(out_dir, f"{base}_waveform.csv")
#     np.savetxt(csv_path, np.stack([t, pred_seq], axis=1), delimiter=",", header="time_sec,pred", comments="")
#     if label_seq is not None and len(label_seq) == len(pred_seq):
#         csv_lab_path = os.path.join(out_dir, f"{base}_waveform_with_label.csv")
#         np.savetxt(csv_lab_path, np.stack([t, pred_seq, label_seq], axis=1), delimiter=",", header="time_sec,pred,label", comments="")

#     fig, ax = plt.subplots(figsize=(8, 3))
#     ax.plot(t, pred_seq, label='pred', linewidth=1.0)
#     if label_seq is not None and len(label_seq) == len(pred_seq):
#         ax.plot(t, label_seq, label='label', linewidth=1.0, alpha=0.7)
#     ax.set_xlabel('Time (s)')
#     ax.set_ylabel('Signal (a.u.)')
#     ax.set_title('Respiration waveform')
#     ax.legend()
#     fig.tight_layout()
#     png_path = os.path.join(out_dir, f"{base}_waveform.png")
#     fig.savefig(png_path, dpi=150)
#     plt.close(fig)


def build_model(config, logger):
    model_name = config['MODEL']['NAME']
    split_cfg = _get_test_split_cfg(config)
    frame_h = split_cfg['PREPROCESS']['DOWNSAMPLE_SIZE_BEFORE_TRAINING'][1]
    in_channels = config['MODEL']['IN_CHANNELS']
    frame_depth = config['MODEL']['FRAME_DEPTH']

    if model_name == 'DeepPhys':
        return DeepPhys(in_channels=in_channels, img_size=frame_h)
    if model_name == 'TSCAN':
        return TSCAN(in_channels=in_channels, frame_depth=frame_depth, img_size=frame_h)
    if model_name == 'EfficientPhys':
        return EfficientPhys(in_channels=in_channels, frame_depth=frame_depth, img_size=frame_h)
    if model_name == 'VIRENet':
        return VIRENet(in_channels=in_channels, frame_depth=frame_depth, img_size=frame_h)
    logger.error(f"Unsupported model: {model_name}")
    raise ValueError(f"Unsupported model: {model_name}")


def run_inference(config, model, data_loader, logger):
    if data_loader is None:
        logger.error("No data for inference")
        raise ValueError("No data for inference")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_of_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
    in_channels = config['MODEL'].get('IN_CHANNELS', 3)
    model_name = config['MODEL']['NAME']
    frame_depth = config['MODEL']['FRAME_DEPTH']
    base_len = num_of_gpu * frame_depth
    
    preprocess_cfg = _get_test_split_cfg(config)['PREPROCESS']
    do_optical_flow = preprocess_cfg.get('DO_OPTICAL_FLOW', False)
    label_type = preprocess_cfg.get('LABEL_NORMALIZE_TYPE', 'DiffNormalized')
    
    diff_flag = True if label_type == 'DiffNormalized' else False
    eval_method = config['INFERENCE'].get('EVALUATION_METHOD', 'FFT')
    
    model = model.to(device)
    if num_of_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_of_gpu)))

    post_processor = PostProcessor()

    logger.info("\n===Inference===")
    model.eval()

    # Aggregate predictions across the whole video (per-video metric)
    all_preds = []
    all_labels = []
    fs = None
    infant_flag = None

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing"):
            data, labels, subj_indices, chunk_indices, infant_flag_list, fs_list = batch
            data = data.to(device)
            labels = labels.to(device)

            # Adapt channels like BaseTrainer
            if model_name.lower() not in ["deepphys", "tscan"]:
                if do_optical_flow:
                    data = data[:, :, :in_channels, ...]
                else:
                    data = data[:, :, in_channels:, ...]

            # Flatten for conv input shape
            N, D, C, H, W = data.shape
            data = data.view(N * D, C, H, W)
            labels = labels.view(-1, 1)

            # Ensure data and labels length is divisible by base_len
            valid_len = (data.shape[0] // base_len) * base_len
            data = data[:valid_len]
            labels = labels[:valid_len]

            # Add one more frame for EfficientPhys since it does torch.diff for the input
            if model_name.lower().startswith("efficientphys"):
                last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(num_of_gpu, 1, 1, 1)
                data = torch.cat((data, last_frame), 0)

            # Forward pass
            pred = model(data)

            all_preds.append(pred)
            all_labels.append(labels)
            if fs is None:
                fs = fs_list[0]
            if infant_flag is None:
                infant_flag = infant_flag_list[0]

    if not all_preds:
        return 0.0

    pred_seq = torch.cat(all_preds, dim=0).unsqueeze(0)
    label_seq = torch.cat(all_labels, dim=0).unsqueeze(0)

    pred_rr, _ = post_processor.post_process(
        pred_seq, label_seq,
        fs=fs,
        diff_flag=diff_flag,
        infant_flag=infant_flag,
        use_bandpass=True,
        eval_method=eval_method
    )

    return float(pred_rr)


def find_checkpoint(config, logger, explicit_path=None):
    if explicit_path:
        if not os.path.isfile(explicit_path):
            logger.error(f"Checkpoint not found: {explicit_path}")
            raise FileNotFoundError(f"Checkpoint not found: {explicit_path}")
        return explicit_path
    
    logger.info(f"No checkpoint provided, using best checkpoint from {config['DATA_PATH']['OUTPUT_DIR']}")
    model_name = config['MODEL']['NAME']
    model_dir = os.path.join(config['DATA_PATH']['OUTPUT_DIR'], 'model', model_name, config['NAME'])
    best_path = os.path.join(model_dir, f"{model_name}_best.pth")
    if os.path.isfile(best_path):
        return best_path
    
    # fallback: latest epoch file
    logger.info(f"No best checkpoint found, using latest epoch file from {model_dir}")
    candidates = [f for f in os.listdir(model_dir) if f.startswith(model_name) and f.endswith('.pth')]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    candidates.sort()
    return os.path.join(model_dir, candidates[-1])


def main():
    SEED = 100
    set_random_seeds(SEED)

    parser = argparse.ArgumentParser(description='Inference entry point for infant respiration models')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML used for training')
    parser.add_argument('--checkpoint', type=str, required=False, default=None, help='Path to trained model checkpoint (.pth)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logs_dir = os.path.join(config['DATA_PATH']['OUTPUT_DIR'], 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = setup_logger(f"inference_{datetime.now()}.log", logs_dir)

    # Determine input videos from config
    video_file = config['DATA_PATH']['VIDEO_FILE']
    video_dir = config['DATA_PATH']['VIDEO_DIR']
    assert (video_file is not None) ^ (video_dir is not None), "Provide exactly one of DATA_PATH.VIDEO_FILE or DATA_PATH.VIDEO_DIR"

    if video_file is not None:
        video_paths = [video_file]
    else:
        video_paths = [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]
        video_paths.sort()

    # Build and load model
    model = build_model(config, logger)
    ckpt_path = find_checkpoint(config, logger, args.checkpoint)
    model.load_state_dict(torch.load(ckpt_path))
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    out_dir = os.path.join(config['DATA_PATH']['OUTPUT_DIR'], 'inference')
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for vp in video_paths:
        # Re-sync data loader RNG
        set_random_seeds(SEED)

        test_loader = DataLoader(
            InferenceDataset(config, vp),
            batch_size=config['INFERENCE']['BATCH_SIZE'],
            shuffle=False,
            num_workers=10,
            worker_init_fn=seed_worker,
        )

        # Re-sync inference RNG
        set_random_seeds(SEED)

        rr_bpm = run_inference(config, model, test_loader, logger)
        base = os.path.splitext(os.path.basename(vp))[0]

        result = {
            'video': os.path.abspath(vp),
            'rr_bpm': float(rr_bpm),
        }
        logger.info(f"Inference result for video {vp}: {float(rr_bpm):1f} (BPM)")
        json_path = os.path.join(out_dir, f"{base}_rr.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved: {json_path}")
        summary.append(result)

    summary_path = os.path.join(out_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to: {summary_path}")


if __name__ == '__main__':
    main()

