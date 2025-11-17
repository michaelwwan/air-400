"""Optical flow visualization utilities for demo GIFs."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from pygifsicle import optimize
from torchvision.io import read_video

from processors.pre_processor import PreProcessor


def get_fs(video_path: str) -> int:
    """Return FPS for the provided video."""
    cap = cv2.VideoCapture(video_path)
    fs = int(round(cap.get(cv2.CAP_PROP_FPS)))
    cap.release()
    return fs


def flow_to_bgr(flow: np.ndarray) -> np.ndarray:
    """Convert calculated optical flow (H, W, 2) to BGR-format image (H, W, 3)."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def overlay_flow_on_frame(frame: np.ndarray, flow: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """Blend flow (BGR) onto an ROI frame (BGR). Returns uint8."""
    fr = frame.astype(np.float32)
    fl = flow.astype(np.float32)
    out = fr * (1.0 - alpha) + fl * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def main(
    video_path: str,
    preprocessor: PreProcessor,
    of_methods: Sequence[str],
    output_path: str,
) -> None:
    """Generate a multi-panel GIF showing ROI and optical-flow overlays."""
    visual_frames_dict: Dict[str, List[np.ndarray]] = defaultdict(list)
    max_length = 0

    frames, _, _ = read_video(video_path)
    frames = frames.numpy()  # (T, H, W, C)
    frames = frames[:150, :, :, :]  # Visualize 150 frames at most

    if preprocessor.do_downsample_before_preprocess:
        frames, _ = preprocessor._resize_frames(frames, [1280, 720])

    # Raw frames
    visual_frames_dict["original"] = list(frames)

    # ROI Crop
    # if preprocessor.do_crop_infant_region:
    roi_box, all_boxes = preprocessor.get_roi_box(frames)
    frames = preprocessor.crop_infant_boxes(frames, roi_box)
    # Convert to BGR once here and fix size 512x512
    roi_bgr = np.array([cv2.cvtColor(cv2.resize(f, (512, 512), interpolation=cv2.INTER_AREA), 
                                     cv2.COLOR_RGB2BGR) for f in frames])
    visual_frames_dict["roi"] = list(roi_bgr)
    visual_frames_dict["roi_box"] = [cb for (_, _, cb, _, _, _) in all_boxes]  # chest box
    visual_frames_dict["body_box"] = [bb for (bb, _, _, _, _, _) in all_boxes]
    visual_frames_dict["face_box"] = [fb for (_, fb, _, _, _, _) in all_boxes]

    # Optical flow methods
    roi_bgr_stack = np.stack(visual_frames_dict["roi"], axis=0)  # (T,512,512,3) uint8
    for method in of_methods:
        preprocessor.of_method = method
        flow = preprocessor.run_optical_flow(frames.copy())
        # Flow -> BGR heatmaps in a list, resized to 512x512 once
        flow_bgr_list = []
        for f in flow:
            fb = flow_to_bgr(f)  # (H,W,3) BGR
            if fb.shape[:2] != (512, 512):
                fb = cv2.resize(fb, (512, 512), interpolation=cv2.INTER_LINEAR)
            flow_bgr_list.append(fb)
        # Align lengths & stack
        L = min(len(flow_bgr_list), len(roi_bgr_stack))
        if L == 0:
            continue
        flow_bgr_stack = np.stack(flow_bgr_list[:L], axis=0)  # (L,512,512,3)
        roi_slice = roi_bgr_stack[:L]

        # Batch overlay (BGR in/out)
        overlays_bgr = overlay_flow_on_frame(roi_slice, flow_bgr_stack, alpha=0.45)  # (L,512,512,3) uint8

        # Store overlays as list of BGR
        visual_frames_dict[method] = [frame for frame in overlays_bgr]
        max_length = max(max_length, overlays_bgr.shape[0])

    # Build gif
    gif_frames = []
    all_methods = ["original", "roi"] + of_methods

    for t in range(max_length):  
        tiles = {}
        for method in all_methods:
            method_frames = visual_frames_dict[method]
            method_frame = method_frames[t] if t < len(method_frames) else method_frames[-1]  # padding when necessary
            method_frame = np.squeeze(np.array(method_frame))
            if method_frame.dtype != np.uint8:
                method_frame = np.clip(method_frame, 0, 255).astype(np.uint8)

            if method != "original":
                 method_frame = cv2.cvtColor(method_frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(method_frame)
            draw = ImageDraw.Draw(img)

            # Add ROI detected box on original frame
            if method == "original" and preprocessor.do_crop_infant_region:
                try:
                    box_font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=42)
                except:
                    box_font = ImageFont.load_default(size=42)

                bx, by, bw, bh = visual_frames_dict["body_box"][t]
                body_box = (bx, by, bx + bw, by + bh)
                draw.rectangle(body_box, outline="green", width=4)
                draw.text(xy=(body_box[0] + 10, max(0, body_box[1] + 10)), text="BODY", fill="green", font=box_font)

                fx, fy, fw, fh = visual_frames_dict["face_box"][t]
                face_box = (fx, fy, fx + fw, fy + fh)
                draw.rectangle(face_box, outline="red", width=4)
                draw.text(xy=(face_box[0] + 10, max(0, face_box[1] + 10)), text="FACE", fill="red", font=box_font)

                rx, ry, rw, rh = visual_frames_dict["roi_box"][t]
                roi_box = (rx+2, ry+2, rx + rw-2, ry + rh-2)
                draw.rectangle(roi_box, outline="blue", width=4)
                draw.text(xy=(roi_box[0] + 10, max(0, roi_box[1] + 10)), text="CHEST", fill="blue", font=box_font)

            # Add label on frame
            subject_name = os.path.basename(video_path).split('.')[0]
            label = f"{method.upper()}" if method != "original" else f"{subject_name}"
            label_size = 32 if method != "original" else 64
            label_xy = 5 if method != "original" else 10
            try:
                label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=label_size)
            except:
                label_font = ImageFont.load_default(size=label_size)
            draw.text(xy=(label_xy, label_xy), text=label, fill=(255, 255, 255), font=label_font)

            # Resize after drawing box
            if method == "original":
                img = img.resize((1024, 512), resample=Image.Resampling.BILINEAR)
            else:
                img = img.resize((512, 512), resample=Image.Resampling.BILINEAR)

            tiles[method] = np.array(img)

        rows = []
        # Row 1: [original(1024×512)] + [roi(512×512)]
        row0 = np.concatenate([tiles["original"], tiles["roi"]], axis=1)
        rows.append(row0)

        # Remaining rows: OF methods, 3 per row
        pad_tile = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(0, len(of_methods), 3):
            chunk = of_methods[i:i+3]
            row_tiles = [tiles[m] for m in chunk]
            # pad last row if < 3
            while len(row_tiles) < 3:
                row_tiles.append(pad_tile)
            rows.append(np.concatenate(row_tiles, axis=1))

        # stack all rows vertically
        frame_grid = np.concatenate(rows, axis=0)
        gif_frames.append(frame_grid)
    
    iio.imwrite(output_path, gif_frames, fps=get_fs(video_path))
    optimize(output_path)  # Smaller output file size

    print(f"Output GIF file successfully saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='video path')
    args = parser.parse_args()
    video_path = args.video

    output_dir = 'example/visual'
    os.makedirs(output_dir, exist_ok=True)

    subject_name = os.path.basename(video_path).split('.')[0]

    # Chest ROI
    print(f"Visualizing chest optical flow for {subject_name}...")
    output_path = os.path.join(output_dir,  f"{subject_name}_of.gif")
    with open("configs/base/virenet_6_fold_cv_air400_with_roi_coarse2fine_of.yaml", 'r') as f:
        config = yaml.safe_load(f)
    preprocess_config = config['TRAIN']['DATA'][0]['PREPROCESS']
    body_detector_path = config['DATA_PATH']['BODY_DETECTOR_PATH']
    face_detector_path = config['DATA_PATH']['FACE_DETECTOR_PATH']

    preprocessor = PreProcessor(preprocess_config, body_detector_path, face_detector_path)
    of_methods = ["coarse2fine", "deep", "farneback", "pca", "tvl1", "raft"]
    main(video_path, preprocessor, of_methods, output_path)

    # Body ROI
    # print(f"Visualizing body optical flow for {subject_name}...")
    # output_path_body = os.path.join(output_dir,  f"{subject_name}_of_body.gif")

    # with open("configs/experiment/virenet_6_fold_cv_air400_with_roi_coarse2fine_of_body.yaml", 'r') as f:
    #     config_body = yaml.safe_load(f)
    # preprocess_config_body = config_body['TRAIN']['DATA'][0]['PREPROCESS']
    # body_detector_path_body = config_body['DATA_PATH']['BODY_DETECTOR_PATH']
    # face_detector_path_body = config_body['DATA_PATH']['FACE_DETECTOR_PATH']

    # preprocessor_body = PreProcessor(preprocess_config_body, body_detector_path_body, face_detector_path_body)
    # main(video_path, preprocessor_body, of_methods, output_path_body)
