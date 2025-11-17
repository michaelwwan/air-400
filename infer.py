"""Inference entry point for running pretrained respiration models."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.inference_dataset import InferenceDataset
from main import seed_worker, set_random_seeds, setup_logger
from models.deep_phys import DeepPhys
from models.ts_can import TSCAN
from models.efficient_phys import EfficientPhys
from models.vire_net import VIRENet
from processors.post_processor import PostProcessor


def _get_test_split_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the TEST.DATA config (handles list/dict conveniences)."""
    split_cfg = config.get('TEST', {})
    data = split_cfg.get('DATA') if isinstance(split_cfg, dict) else None
    if data is None:
        raise KeyError("Config missing TEST.DATA")
    if isinstance(data, list):
        if not data:
            raise ValueError("Config TEST.DATA list is empty")
        return data[0]
    if isinstance(data, dict):
        return data
    raise TypeError(f"Unexpected type for TEST.DATA: {type(data)}")


def _save_pred(
    out_dir: str,
    base: str,
    fs: int,
    pred_seq: np.ndarray,
    logger: logging.Logger,
) -> Tuple[str, str]:
    """Save prediction sequence as HDF5 + waveform PNG."""
    pred_seq = np.asarray(pred_seq).reshape(-1)
    t = np.arange(len(pred_seq), dtype=np.float32)
    if fs > 0:
        t = t / fs

    hdf5_path = os.path.join(out_dir, f"{base}_pred.hdf5")
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('time_sec', data=t, compression='gzip')
        f.create_dataset('respiration', data=pred_seq, compression='gzip')
    logger.info(f"Saved hdf5 format prediction to {hdf5_path}")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, pred_seq, label='pred', linewidth=1.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Respiration')
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{base}_pred.png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved waveform plot of prediction to {png_path}")

    return hdf5_path, png_path


def build_model(config: Dict[str, Any], logger: logging.Logger) -> torch.nn.Module:
    """Instantiate the requested model class."""
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


def run_inference(
    config: Dict[str, Any],
    model: torch.nn.Module,
    data_loader: DataLoader,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run the model over the provided loader and return per-video stats."""
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

    diff_flag = label_type == 'DiffNormalized'
    eval_method = config['INFERENCE'].get('EVALUATION_METHOD', 'FFT')

    model = model.to(device)
    if num_of_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_of_gpu)))

    post_processor = PostProcessor()
    model.eval()

    all_preds = []
    fs = None
    infant_flag = None

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing"):
            data, infant_flag_list, fs_list = batch
            data = data.to(device)

            # Adapt channels like BaseTrainer
            if model_name.lower() not in ["deepphys", "tscan"]:
                if do_optical_flow:
                    data = data[:, :, :in_channels, ...]
                else:
                    data = data[:, :, in_channels:, ...]

            # Flatten for conv input shape
            N, D, C, H, W = data.shape
            data = data.view(N * D, C, H, W)

            # Ensure data and labels length is divisible by base_len
            valid_len = (data.shape[0] // base_len) * base_len
            data = data[:valid_len]

            # Add one more frame for EfficientPhys since it does torch.diff for the input
            if model_name.lower().startswith("efficientphys") and data.shape[0] > 0:
                last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(num_of_gpu, 1, 1, 1)
                data = torch.cat((data, last_frame), 0)

            # Forward pass
            pred = model(data)

            all_preds.append(pred)
            if fs is None and len(fs_list):
                fs = fs_list[0]
                if hasattr(fs, 'item'):
                    fs = int(fs.item())
            if infant_flag is None and len(infant_flag_list):
                infant_flag = infant_flag_list[0]
                if hasattr(infant_flag, 'item'):
                    infant_flag = bool(infant_flag.item())

    pred_seq = torch.cat(all_preds, dim=0).unsqueeze(0)
    pred_seq, pred_rr = post_processor.post_process(
        pred_seq,
        fs=fs,
        diff_flag=diff_flag,
        infant_flag=infant_flag,
        use_bandpass=True,
        eval_method=eval_method
    )

    return {
        'pred_rr_bpm': float(pred_rr),
        'fs': fs,
        'pred_seq': pred_seq,
    }


def find_checkpoint(config: Dict[str, Any], logger: logging.Logger, explicit_path: Optional[str] = None) -> str:
    """Resolve which checkpoint to load."""
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


def main() -> None:
    """Entry point for running inference on videos listed in config."""
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
    video_file = config['DATA_PATH'].get('VIDEO_FILE', None)
    video_dir = config['DATA_PATH'].get('VIDEO_DIR', None)
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
        logger.info(f" === Start inference for video {vp} ===")

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

        base = os.path.splitext(os.path.basename(vp))[0]
        vp_out_dir = os.path.join(out_dir, base + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(vp_out_dir, exist_ok=True)

        metrics = run_inference(config, model, test_loader, logger)
        fs = metrics['fs']

        result = {
            'video': os.path.abspath(vp),
            'fs': fs,
            'pred_rr_bpm': float(metrics['pred_rr_bpm'])
        }

        pred_seq = metrics.get('pred_seq', np.array([]))

        if pred_seq.size:
            hdf5_path, png_path = _save_pred(
                vp_out_dir,
                base,
                fs,
                pred_seq,
                logger
            )

            result['hdf5_path'] = hdf5_path
            result['png_path'] = png_path
            logger.info(f"Saved waveform hdf5/csv/png for {vp}")

        per_video_path = os.path.join(vp_out_dir, f"{base}_result.json")
        with open(per_video_path, 'w') as f:
            json.dump(result, f, indent=2)
        result['result_json'] = per_video_path
        logger.info(f"Saved per-video result to {per_video_path}")
        logger.info(f"Inference result for video {vp}: {result['pred_rr_bpm']:.2f} BPM")
        summary.append(result)

    summary_path = os.path.join(out_dir, f"summary_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to: {summary_path}")


if __name__ == '__main__':
    main()
