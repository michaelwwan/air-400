"""Data preprocessing utilities for training/inference pipelines."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import sys
from multiprocessing import cpu_count, Pool
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import pyflow
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms import v2
from tqdm import tqdm
from ultralytics import YOLO


@contextlib.contextmanager
def _suppress_pyflow_output() -> Iterator[None]:
    """Suppress verbose stdout/stderr emitted by pyflow's C++ implementation."""
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


class PreProcessor:
    """Encapsulates all frame, flow, and ROI preprocessing logic."""

    def __init__(
        self,
        preprocess_config: Dict[str, Any],
        body_detector_path: str,
        face_detector_path: str,
    ) -> None:
        """Store configuration and initialize detectors."""
        self.preprocess_config = preprocess_config
        self.body_detector_path = body_detector_path
        self.face_detector_path = face_detector_path
        self.logger = logging.getLogger(__name__)
        self._load_params()
        self._init_detector()

    def _load_params(self) -> None:
        """Load preprocess configs."""
        # Downsample to lower resolution for faster preprocessing
        self.do_downsample_before_preprocess = self.preprocess_config.get('DO_DOWNSAMPLE_BEFORE_PREPROCESS', False)
        self.downsample_size_before_preprocess = self.preprocess_config.get('DOWNSAMPLE_SIZE_BEFORE_PREPROCESS', [640, 360]) \
            if self.do_downsample_before_preprocess else [0, 0]

        self.data_normalize_type = self.preprocess_config.get('DATA_NORMALIZE_TYPE', 'Standardized')  # Type of frame normalization
        self.label_normalize_type = self.preprocess_config.get('LABEL_NORMALIZE_TYPE', 'DiffNormalized')  # Type of label normalization
        self.flow_normalize_type = self.preprocess_config.get('FLOW_NORMALIZE_TYPE', 'Standardized')  # Type of flow normalization

        self.do_chunk = self.preprocess_config.get('DO_CHUNK', True)
        self.chunk_length = self.preprocess_config.get('CHUNK_LENGTH', 180) \
            if self.do_chunk else 0  # Length of each data chunk

        # Add infant region detection parameters
        self.do_crop_infant_region = self.preprocess_config.get('DO_CROP_INFANT_REGION', False)  # Whether to crop faces
        self.dynamic_detection = self.preprocess_config.get('DYNAMIC_DETECTION', False)  # Whether to detect infant dynamically
        self.dynamic_detection_frequency = self.preprocess_config.get('DYNAMIC_DETECTION_FREQUENCY', 10) \
            if self.dynamic_detection else 0  # Frequency for dynamic infant detection
        self.infant_region = self.preprocess_config.get('INFANT_REGION', 'Chest') \
            if self.do_crop_infant_region else ''
        self.body_conf_threshold = self.preprocess_config.get('BODY_CONF_THRESHOLD', 0.25) \
            if self.do_crop_infant_region else 0.0
        self.face_conf_threshold = self.preprocess_config.get('FACE_CONF_THRESHOLD', 0.5) \
            if self.do_crop_infant_region else 0.0
        self.chest_shift_alpha = self.preprocess_config.get('CHEST_SHIFT_ALPHA', 0.2) \
            if self.do_crop_infant_region else 0.0
        self.use_larger_box = self.preprocess_config.get('USE_LARGER_BOX', False) \
            if self.do_crop_infant_region else False  # Whether to use a larger bounding box
        self.larger_box_coef = self.preprocess_config.get('LARGER_BOX_COEF', 1.0) \
            if self.use_larger_box else 0.0  # Coefficient for larger box

        # Add augmentation parameters
        self.do_grayscale = self.preprocess_config.get('DO_GRAYSCALE', False)
        self.do_augmentation = self.preprocess_config.get('DO_AUGMENTATION', False)
        self.augmentation_times = self.preprocess_config.get('AUGMENTATION_TIMES', 2) \
            if self.do_augmentation else 0
        self.do_rotation = self.preprocess_config.get('DO_ROTATION', False) \
            if self.do_augmentation else False
        # self.rotation_angles_range = self.preprocess_config.get('ROTATION_ANGLES_RANGE', [0, 180]) \
        #     if self.do_augmentation else [0, 0]
        self.do_horizontal_flip = self.preprocess_config.get('DO_HORIZONTAL_FLIP', False) \
            if self.do_augmentation else False
        self.horizontal_flip_p = self.preprocess_config.get('HORIZONTAL_FLIP_P', 0.5) \
            if self.do_augmentation else 0.0
        self.do_vertical_flip = self.preprocess_config.get('DO_VERTICAL_FLIP', False) \
            if self.do_augmentation else False
        self.vertical_flip_p = self.preprocess_config.get('VERTICAL_FLIP_P', 0.5) \
            if self.do_augmentation else 0.0
        self.do_brightness_adjustment = self.preprocess_config.get('DO_BRIGHTNESS_ADJUSTMENT', False) \
            if self.do_augmentation else False
        self.brightness_adjustment = self.preprocess_config.get('BRIGHTNESS_ADJUSTMENT', 0.2) \
            if self.do_augmentation else 0.0
        self.do_contrast_adjustment = self.preprocess_config.get('DO_CONTRAST_ADJUSTMENT', False) \
            if self.do_augmentation else False
        self.contrast_adjustment = self.preprocess_config.get('CONTRAST_ADJUSTMENT', 0.2) \
            if self.do_augmentation else 0.0

        # Add optical flow parameters
        self.do_optical_flow = self.preprocess_config.get('DO_OPTICAL_FLOW', False)
        self.of_method = self.preprocess_config.get('OF_METHOD', 'raft') \
            if self.do_optical_flow else ''
        self.of_resample_fs = self.preprocess_config.get('OF_RESAMPLE_FS', 0) \
            if self.do_optical_flow else 0
        self.pyflow_alpha = self.preprocess_config.get('PYFLOW_ALPHA', 0.012) \
            if self.do_optical_flow else 0.0  # Smoothness weight
        self.pyflow_ratio = self.preprocess_config.get('PYFLOW_RATIO', 0.75) \
            if self.do_optical_flow else 0.0  # Image pyramid scale, lower ratio → more pyramid levels
        self.pyflow_min_width = self.preprocess_config.get('PYFLOW_MIN_WIDTH', 20) \
            if self.do_optical_flow else 0  # Smallest image width to stop pyramid downscaling
        self.pyflow_n_outer_FP_iterations = self.preprocess_config.get('PYFLOW_N_OUTER_FP_ITERATIONS', 7) \
            if self.do_optical_flow else 0  # Number of outer iterations of pyramid
        self.pyflow_n_inner_FP_iterations = self.preprocess_config.get('PYFLOW_N_INNER_FP_ITERATIONS', 1) \
            if self.do_optical_flow else 0  # Number of inner iterations of pyramid
        self.pyflow_n_SOR_iterations = self.preprocess_config.get('PYFLOW_N_SOR_ITERATIONS', 30) \
            if self.do_optical_flow else 0  # Number of solver steps of pyramid
        self.pyflow_col_type = 1 if self.do_grayscale else 0  # 0: RGB, 1: GRAY

        # Downsample to lower resolution before training models
        self.do_downsample_before_training = self.preprocess_config.get('DO_DOWNSAMPLE_BEFORE_TRAINING', False)
        self.downsample_size_before_training = self.preprocess_config.get('DOWNSAMPLE_SIZE_BEFORE_TRAINING', [96, 96]) \
            if self.do_downsample_before_training else [0, 0]

    def _init_detector(self) -> None:
        """Initiate body detectors and face detectors."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.face_detector = None
        self.body_detector = None

        if not self.do_crop_infant_region:
            # No cropping, no detectors needed
            return

        try:
            self.body_detector = YOLO(self.body_detector_path).to(self.device)
            self.logger.info(f"YOLOv8 model loaded from {self.body_detector_path} on {self.device}")
        except Exception:
            self.logger.exception(f"Failed to load YOLOv8")

        try:
            self.face_detector = YOLO(self.face_detector_path).to(self.device)
            self.logger.info(f"YOLOv8n-Face model loaded from {self.face_detector_path} on {self.device}")
        except Exception:
            self.logger.exception(f"Failed to load YOLOv8n-Face model")

    def preprocess(
        self,
        frames: np.ndarray,
        labels: np.ndarray,
        raw_fs: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the entire preprocessing pipeline and return chunked tensors."""
        # 1. (Optional) downsample to lower resolution for faster preprocessing
        if self.do_downsample_before_preprocess:
            frames, _ = self._resize_frames(frames, self.downsample_size_before_preprocess)

        # 2. (Optional) Detect infant boxes and crop frames
        if self.do_crop_infant_region:
            roi_box, _ = self.get_roi_box(frames)
            frames = self.crop_infant_boxes(frames, roi_box)

        # 3. (Optional) compute optical flow on cropped frames
        flow = None
        if self.do_optical_flow:
            # Resample if required
            if self.of_resample_fs > 0 and self.of_resample_fs != raw_fs:
                frames, labels = self._resample_frames_labels(frames, labels, raw_fs)
            flow = self.run_optical_flow(frames)

        # 4. (Optional) convert frames to grayscale
        if self.do_grayscale:
            frames = self.convert_to_grayscale(frames)

        # 5. (Optional) apply augmentations on frames and flow together
        augmented_frames = [frames]
        augmented_labels = [labels]
        augmented_flow = [flow] if flow is not None else None
        if self.do_augmentation:
            for _ in range(self.augmentation_times):
                fr_augmented, fl_augmented = self._augment(frames, flow)
                augmented_frames.append(fr_augmented)
                augmented_labels.append(labels.copy())
                if flow is not None:
                    augmented_flow.append(fl_augmented)

        # Concatenate all augmentations
        frames = np.concatenate(augmented_frames, axis=0)
        labels = np.concatenate(augmented_labels, axis=0)
        if flow is not None:
            flow = np.concatenate(augmented_flow, axis=0)

        # 6. Apply normalizations for frames, flow, and labels
        frames = self._normalize_frames(frames)
        labels = self._normalize_labels(labels)
        if self.do_optical_flow:
            flow = self._normalize_flow(flow)

        # 7. (Optional) downsample to lower resolution to meet training requirements
        if self.do_downsample_before_training:
            if flow is not None:
                frames, flow = self._resize_frames(frames, self.downsample_size_before_training, flow)
            else:
                frames, _ = self._resize_frames(frames, self.downsample_size_before_training)

        # 8. Pack frames and flow together (6-channel, flow||frames, or diff||frames)

        # Ensure frames 3 channel (in case of grayscale)
        frames = self._ensure_three_channels(frames)

        if flow is not None:
            frames = np.concatenate([flow, frames], axis=-1)  # (T, H, W, 3(flow)+3(frames))
        else:
            # 3-channel diffs
            diff = np.empty_like(frames)  # (T, H, W, 3)
            diff[0] = 0.0
            diff[1:] = frames[1:] - frames[:-1]
            frames = np.concatenate([diff, frames], axis=-1)  # (T, H, W, 6)

        if frames.ndim != 4 or frames.shape[-1] != 6:
            self.logger.error(f"Expected (T,H,W,6), got {frames.shape}")
            raise ValueError(f"Expected (T,H,W,6), got {frames.shape}")

        # 9. (Optional) Chunk
        if self.chunk_length > 0:
            frames_clips, labels_clips = self._chunk(frames, labels)
        else:
            frames_clips = np.array([frames])
            labels_clips = np.array([labels])

        self.logger.debug(f"Preprocessing complete. Frame clips shape: {frames_clips.shape}, label clips shape: {labels_clips.shape}")

        return frames_clips, labels_clips

    def hash_param_dict(self) -> Tuple[Dict[str, Any], str]:
        """Return a stable dict of preprocessing parameters and a hash."""
        # Generate a param dict with stable key order
        preprocess_param_dict = {
            'DO_DOWNSAMPLE_BEFORE_PREPROCESS': self.do_downsample_before_preprocess,
            'DOWNSAMPLE_SIZE_BEFORE_PREPROCESS': self.downsample_size_before_preprocess,
            'DATA_NORMALIZE_TYPE': self.data_normalize_type,
            'LABEL_NORMALIZE_TYPE': self.label_normalize_type,
            'FLOW_NORMALIZE_TYPE': self.flow_normalize_type,
            'DO_CHUNK': self.do_chunk,
            'CHUNK_LENGTH': self.chunk_length,
            'DO_CROP_INFANT_REGION': self.do_crop_infant_region,
            'DYNAMIC_DETECTION': self.dynamic_detection,
            'DYNAMIC_DETECTION_FREQUENCY': self.dynamic_detection_frequency,
            'INFANT_REGION': self.infant_region,
            'BODY_CONF_THRESHOLD': self.body_conf_threshold,
            'FACE_CONF_THRESHOLD': self.face_conf_threshold,
            'CHEST_SHIFT_ALPHA': self.chest_shift_alpha,
            'USE_LARGER_BOX': self.use_larger_box,
            'LARGER_BOX_COEF': self.larger_box_coef,
            'DO_GRAYSCALE': self.do_grayscale,
            'DO_AUGMENTATION': self.do_augmentation,
            'AUGMENTATION_TIMES': self.augmentation_times,
            'DO_ROTATION': self.do_rotation,
            # 'ROTATION_ANGLES_RANGE': self.rotation_angles_range,
            'DO_HORIZONTAL_FLIP': self.do_horizontal_flip,
            'HORIZONTAL_FLIP_P': self.horizontal_flip_p,
            'DO_VERTICAL_FLIP': self.do_vertical_flip,
            'VERTICAL_FLIP_P': self.vertical_flip_p,
            'DO_BRIGHTNESS_ADJUSTMENT': self.do_brightness_adjustment,
            'BRIGHTNESS_ADJUSTMENT': self.brightness_adjustment,
            'DO_CONTRAST_ADJUSTMENT': self.do_contrast_adjustment,
            'CONTRAST_ADJUSTMENT': self.contrast_adjustment,
            'DO_OPTICAL_FLOW': self.do_optical_flow,
            'OF_METHOD': self.of_method,
            'OF_RESAMPLE_FS': self.of_resample_fs,
            'PYFLOW_ALPHA': self.pyflow_alpha,
            'PYFLOW_RATIO': self.pyflow_ratio,
            'PYFLOW_MIN_WIDTH': self.pyflow_min_width,
            'PYFLOW_N_OUTER_FP_ITERATIONS': self.pyflow_n_outer_FP_iterations,
            'PYFLOW_N_INNER_FP_ITERATIONS': self.pyflow_n_inner_FP_iterations,
            'PYFLOW_N_SOR_ITERATIONS': self.pyflow_n_SOR_iterations,
            'DO_DOWNSAMPLE_BEFORE_TRAINING': self.do_downsample_before_training,
            'DOWNSAMPLE_SIZE_BEFORE_TRAINING': self.downsample_size_before_training,
        }
        # Hash the param dict
        preprocess_str = json.dumps(preprocess_param_dict, sort_keys=True)
        preprocess_hash = hashlib.sha256(preprocess_str.encode()).hexdigest()
        return preprocess_param_dict, preprocess_hash

    def get_roi_box(self, frames: np.ndarray) -> Tuple[List[float], List[List[float]]]:
        """Return a fixed ROI bounding box and supporting boxes for the clip."""
        H, W = frames.shape[1], frames.shape[2]

        # Pick detection frames by frequency
        if self.dynamic_detection:
            step = max(1, self.dynamic_detection_frequency)
        else:
            step = frames.shape[0]
        detection_indices = list(range(0, frames.shape[0], step))
        # Ensure last frame included
        if not detection_indices or detection_indices[-1] != frames.shape[0] - 1:
            detection_indices.append(frames.shape[0] - 1)

        all_boxes = []
        for i in detection_indices:
            frame = frames[i]
            body_box = self.detect_body_box(frame)  # [xywh] or None
            body_bad = False
            if body_box is None:
                body_box = [0, 0, W, H]  # full frame
                body_bad = True
            else:
                if (body_box[0] <= 1e-3 and
                        body_box[1] <= 1e-3 and
                        body_box[2] >= 0.98 * W and
                        body_box[3] >= 0.98 * H):  # full frame
                    body_bad = True

            face_box = self.detect_face_box(frame)  # [xywh] or None
            face_bad = False
            if face_box is None:
                face_box = body_box.copy()  # face not detected, fall back to body box
                face_bad = True
            else:
                if (face_box[0] <= 1e-3 and
                        face_box[1] <= 1e-3 and
                        face_box[2] >= 0.98 * W and
                        face_box[3] >= 0.98 * H):  # full frame
                    face_bad = True

            # Derive a chest square box from body_box and face_box
            chest_box = self._derive_chest_box(body_box, face_box, W, H, alpha=self.chest_shift_alpha)
            # body is full-frame and face is also bad, chest is untrustworthy
            chest_bad = body_bad and face_bad
            all_boxes.append((body_box, face_box, chest_box, body_bad, face_bad, chest_bad))

        # Aggregate boxes; exclude boxes derived from full frame body boxes
        body_boxes = [bb for (bb, _, _, body_bad, _, _) in all_boxes if not body_bad]
        if not body_boxes:
            body_boxes = [bb for (bb, _, _, _, _, _) in all_boxes]

        face_boxes = [fb for (_, fb, _, _, face_bad, _) in all_boxes if not face_bad]
        if not face_boxes:
            face_boxes = [fb for (_, fb, _, _, _, _) in all_boxes]

        chest_boxes = [cb for (_, _, cb, _, _, chest_bad) in all_boxes if not chest_bad]
        if not chest_boxes:
            chest_boxes = [cb for (_, _, cb, _, _, _) in all_boxes]

        if self.infant_region.lower() == 'body':
            best_box = self._get_best_box(body_boxes, W, H)
        elif self.infant_region.lower() == 'face':
            best_box = self._get_best_box(face_boxes, W, H)
        elif self.infant_region.lower() == 'chest':
            best_box = self._get_best_box(chest_boxes, W, H, square=True)
        else:
            self.logger.exception(f"Not valid infant region type: {self.infant_region}")
            raise ValueError(f"Not valid infant region type: {self.infant_region}")

        # Optionally enlarge the box
        if self.use_larger_box and self.larger_box_coef > 1.0:
            best_box = self._enlarge_box(best_box, W, H)
        
        # Fill all_boxes for every frame for visualization
        filled_boxes = []
        detection_map = dict(zip(detection_indices, all_boxes))
        last_box = all_boxes[0]  # safe initialization
        for i in range(frames.shape[0]):
            if i in detection_map:
                last_box = detection_map[i]
            filled_boxes.append(last_box)

        return best_box, filled_boxes

    def detect_body_box(self, frame: np.ndarray) -> Optional[List[float]]:
        """Apply YOLO detection on a single frame and return an [x, y, w, h] box."""
        if self.body_detector is None:
            self.logger.warning("Body detector is not initialized.")
            return None

        H, W = frame.shape[:2]

        # Ensure uint8 RGB
        img = frame
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.ndim == 2:  # gray -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:  # BGRA/RGBA -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # Run YOLOv8 batched inference
        try:
            result = self.body_detector(
                img,
                conf=self.body_conf_threshold,
                device=self.device,
                verbose=False
            )[0]
        except Exception:
            self.logger.exception(f"Error during body detection")
            return None

        # Process results
        best_box, best_conf = None, 0.0
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # class == person
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_box = box
                    best_conf = conf
        # If no valid box detected, fall back to face detection
        if not best_box:
            self.logger.debug("No body detections found")
            return None

        # convert to xywh
        x1, y1, x2, y2 = map(float, best_box.xyxy[0])
        x, y, w, h = [x1, y1, x2 - x1, y2 - y1]

        # clamp
        x = max(0.0, min(x, W - 1.0))
        y = max(0.0, min(y, H - 1.0))
        w = max(0.0, min(w, W - x))
        h = max(0.0, min(h, H - y))
        if w <= 0.0 or h <= 0.0:
            return None

        body_box = [x, y, w, h]
        self.logger.debug(f"Detected body box (YOLOv8): {body_box}, confidence: {best_conf:.2f}")
        return body_box

    def detect_face_box(self, frame: np.ndarray) -> Optional[List[float]]:
        """Detect faces (with rotation retries) and return an [x, y, w, h] box."""

        def _rot90(img: np.ndarray, angle: int) -> np.ndarray:
            if angle == 0:
                return img
            if angle == 90:
                return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            if angle == 180:
                return cv2.rotate(img, cv2.ROTATE_180)
            if angle == 270:
                return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.logger.exception("Angle must be one of {0,90,180,270}")
            raise ValueError("Angle must be one of {0,90,180,270}")

        def _rot_to_orig(xyxy_r: Sequence[float], angle: int, W: int, H: int) -> List[float]:
            # map rotated xyxy back to original coords by transforming all 4 corners
            x1r, y1r, x2r, y2r = xyxy_r
            pts_r = [(x1r, y1r), (x1r, y2r), (x2r, y1r), (x2r, y2r)]
            pts_o = []
            for xr, yr in pts_r:
                if angle == 0:
                    xo, yo = xr, yr
                elif angle == 90:  # clockwise
                    xo, yo = yr, H - 1 - xr
                elif angle == 180:
                    xo, yo = W - 1 - xr, H - 1 - yr
                elif angle == 270:  # counter-clockwise
                    xo, yo = W - 1 - yr, xr
                else:
                    self.logger.exception("Angle must be one of {0,90,180,270}")
                    raise ValueError("Angle must be one of {0,90,180,270}")
                pts_o.append((xo, yo))
            minx = max(0.0, min(p[0] for p in pts_o))
            miny = max(0.0, min(p[1] for p in pts_o))
            maxx = min(float(W), max(p[0] for p in pts_o))
            maxy = min(float(H), max(p[1] for p in pts_o))
            return [minx, miny, maxx, maxy]

        if self.face_detector is None:
            self.logger.warning("Could not load YOLOv8n-Face detector.")
            return None

        H, W = frame.shape[:2]

        # Ensure uint8 RGB
        img = frame
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.ndim == 2:  # gray -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:  # BGRA/RGBA -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # Attempt face detection
        best_box, best_conf, best_area = None, -1.0, -1.0
        for ang in [0, 90, 180, 270]:
            try:
                img_r = _rot90(img, ang)
                result = self.face_detector(
                    img_r,
                    conf=self.face_conf_threshold,
                    device=self.device,
                    verbose=False
                )[0]
            except Exception:
                self.logger.exception(f"Face detection failed")
                continue

            if not hasattr(result, "boxes") or len(result.boxes) == 0:
                continue

            # Process results
            for box in result.boxes:
                conf = float(box.conf[0])
                xyxy_r = map(float, box.xyxy[0])
                # rotate back to original coords
                x1, y1, x2, y2 = _rot_to_orig(xyxy_r, ang, W, H)
                w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
                if w <= 0.0 or h <= 0.0:
                    continue
                area = w * h
                if conf > best_conf or (abs(conf - best_conf) < 1e-2 and area > best_area):
                    best_box = [x1, y1, x2, y2]
                    best_conf = conf
                    best_area = area

        if not best_box:
            self.logger.debug("No face detections found in all rotations")
            return None

        # convert to xywh
        x1, y1, x2, y2 = best_box
        x, y, w, h = [x1, y1, x2 - x1, y2 - y1]

        # clamp
        x = max(0.0, min(x, W - 1.0))
        y = max(0.0, min(y, H - 1.0))
        w = max(0.0, min(w, W - x))
        h = max(0.0, min(h, H - y))
        if w <= 0.0 or h <= 0.0:
            return None

        face_box = [x, y, w, h]
        self.logger.debug(f"Detected face box (YOLOv8n-Face): {face_box}, confidence: {best_conf:.2f}")
        return face_box

    def _derive_chest_box(
        self,
        body_box: Sequence[float],
        face_box: Sequence[float],
        W: int,
        H: int,
        alpha: float = 0.2,
    ) -> List[float]:
        """
        Derive a square chest ROI constrained to the torso.
        Side: short side of body box.
        Center: if w >= h (landscape): cy = infant_cy; cx = infant_cx shifts towards face_cx;
        if h >  w (portrait):  cx = infant_cx; cy = infant_cy shifts towards face_cy.

        Args:
            body_box: [x,y,w,h]
            face_box: [x,y,w,h]
            raw frame size
            raw frame size
            alpha: nudge amount of shifts towards face center (0, 1)

        Returns:
         [x, y, side, side]
        """
        bx, by, bw, bh = body_box
        bcx, bcy = bx + bw / 2.0, by + bh / 2.0
        fx, fy, fw, fh = face_box
        fcx, fcy = fx + fw / 2.0, fy + fh / 2.0

        if bw >= bh:  # landscape body box
            chest_cy = bcy  # height center
            chest_cx = bcx + (fcx - bcx) * alpha  # Offset a bit from torso center toward face
        else:  # portrait body box
            chest_cx = bcx  # width center
            chest_cy = bcy + (fcy - bcy) * alpha  # Offset a bit from torso center toward face

        # size = short side of body box
        side = max(1.0, min(bw, bh))

        # initial box
        x = chest_cx - side / 2.0
        y = chest_cy - side / 2.0

        # constrain to body box
        x = max(bx, min(x, bx + bw - side))
        y = max(by, min(y, by + bh - side))

        # Constrain to frame
        x = max(0.0, min(x, W - side))
        y = max(0.0, min(y, H - side))

        chest_box = [x, y, side, side]
        self.logger.debug(f"Detected chest box: {chest_box}")
        return chest_box

    def _enlarge_box(self, box: Sequence[float], W: int, H: int) -> List[int]:
        x, y, w, h = map(float, box)
        k = float(self.larger_box_coef)
        nx = x - (k - 1.0) / 2 * w
        ny = y - (k - 1.0) / 2 * h
        nw = k * w
        nh = k * h
        nx = max(0.0, min(nx, W - nw))
        ny = max(0.0, min(ny, H - nh))
        enlarged_box = [int(nx), int(ny), int(nw), int(nh)]
        self.logger.debug(f"Original box: {box}, Enlarged box: {enlarged_box}")
        return enlarged_box

    def _get_best_box(
        self,
        all_boxes: Sequence[Sequence[float]],
        W: int,
        H: int,
        square: bool = False,
    ) -> List[int]:
        """Aggregate detections into a single robust bounding box."""
        if not all_boxes:
            # fall back to whole frame (square for chest)
            if square:
                side = min(W, H)
                x = (W - side) // 2
                y = (H - side) // 2
                return [int(x), int(y), int(side), int(side)]
            else:
                return [0, 0, W, H]

        cx_list, cy_list, w_list, h_list = [], [], [], []
        for (x, y, w, h) in all_boxes:
            cx_list.append(x + w / 2.0)
            cy_list.append(y + h / 2.0)
            w_list.append(w)
            h_list.append(h)

        # Median aggregation to get the best box
        cx_best = np.median(cx_list)
        cy_best = np.median(cy_list)
        w_best = np.percentile(w_list, 75)  # 75th percentile for robustness
        h_best = np.percentile(h_list, 75)
        side_list = [min(w, h) for w, h in zip(w_list, h_list)]  # for square chest box
        side_best = np.percentile(side_list, 75)

        if square:
            x = cx_best - side_best / 2.0
            y = cy_best - side_best / 2.0
            # Clamp to frame
            x = max(0.0, min(x, W - side_best))
            y = max(0.0, min(y, H - side_best))
            # Convert to int
            x_i = int(round(x))
            y_i = int(round(y))
            s_i = int(round(side_best))
            x_i = max(0, min(x_i, W - s_i))
            y_i = max(0, min(y_i, H - s_i))
            best_box = [x_i, y_i, s_i, s_i]
        else:
            x = cx_best - w_best / 2.0
            y = cy_best - h_best / 2.0
            # Clamp to frame
            x = max(0.0, min(x, W - w_best))
            y = max(0.0, min(y, H - h_best))
            # Convert to int
            x_i = int(round(x))
            y_i = int(round(y))
            w_i = int(round(w_best))
            h_i = int(round(h_best))
            x_i = max(0, min(x_i, W - w_i))
            y_i = max(0, min(y_i, H - h_i))
            best_box = [x_i, y_i, w_i, h_i]

        self.logger.debug(f"Best Box: {best_box}")
        return best_box

    @staticmethod
    def crop_infant_boxes(frames: np.ndarray, roi_box: Sequence[float]) -> np.ndarray:
        """Crop frames using a fixed bounding box, padding if necessary."""
        x, y, w, h = roi_box
        cropped_frames = []
        for frame in frames:
            x1 = int(np.floor(x))
            y1 = int(np.floor(y))
            x2 = int(np.ceil(x + w))
            y2 = int(np.ceil(y + h))

            # Pad zeros if box out of bounds
            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - frame.shape[1])
            pad_bottom = max(0, y2 - frame.shape[0])

            # Clamp coordinates
            x1_clamp = max(0, x1)
            y1_clamp = max(0, y1)
            x2_clamp = min(frame.shape[1], x2)
            y2_clamp = min(frame.shape[0], y2)

            # Crop and pad
            crop = frame[y1_clamp:y2_clamp, x1_clamp:x2_clamp].copy()
            crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant')
            cropped_frames.append(crop.astype(np.float32))

        return np.stack(cropped_frames, axis=0)

    def _resize_frames(
        self,
        frames: np.ndarray,
        target_size: Sequence[int],
        flow: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Resize frames (and optionally flow) to the requested target size."""
        self.logger.debug(f"Resizing frames to {target_size}")
        target_w, target_h = target_size
        T, H, W, C = frames.shape

        # Reisze frames
        resized_frames = np.zeros((T, target_h, target_w, C), dtype=np.float32)
        for i in range(T):
            # Resize the frame
            resized_frames[i] = cv2.resize(frames[i], (target_w, target_h), interpolation=cv2.INTER_AREA)

        if flow is None:
            return resized_frames, None

        # Resize flow
        Cflow = flow.shape[-1]
        resized_flow = np.zeros((T, target_h, target_w, Cflow), dtype=np.float32)
        # Resize dx, dy with interpolation
        for i in range(T):
            # resized_flow[i] = cv2.resize(flow[i], (target_w, target_h), interpolation=cv2.INTER_AREA)

            resized_flow[i, ..., 0] = cv2.resize(
                flow[i, ..., 0],
                (target_w, target_h),
                interpolation=cv2.INTER_AREA
            ) * target_w / float(W)

            resized_flow[i, ..., 1] = cv2.resize(
                flow[i, ..., 1],
                (target_w, target_h),
                interpolation=cv2.INTER_AREA
            ) * target_h / float(H)

            resized_flow[i, ..., 2] = np.sqrt(resized_flow[i, ..., 0]**2 + resized_flow[i, ..., 1]**2 + 1e-6)

        return resized_frames, resized_flow

    @staticmethod
    def convert_to_grayscale(frames: np.ndarray) -> np.ndarray:
        """Convert frames to grayscale."""
        # Check if frames already have one channel
        if frames.shape[-1] == 1:
            return frames

        # Convert numpy frames to PyTorch tensors
        frame_tensors = torch.as_tensor(frames, dtype=torch.float32)
        frame_tensors = frame_tensors.permute(0, 3, 1, 2)  # NDHWC to NDCHW (required by torchvision.transforms.v2)

        # Define the grayscale transform
        transform = v2.Grayscale(num_output_channels=1)

        # Apply grayscale conversion to frame tensors
        frame_tensors = transform(frame_tensors)

        # Convert PyTorch tensors back to numpy frames
        return frame_tensors.permute(0, 2, 3, 1).numpy()

    def _resample_frames_labels(
        self,
        frames: np.ndarray,
        labels: np.ndarray,
        raw_fs: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample frames/labels to `self.of_resample_fs` when needed."""
        if raw_fs % self.of_resample_fs == 0:
            resample_step = raw_fs // self.of_resample_fs
            new_frames = frames[::resample_step]
            new_labels = labels[::resample_step]
        else:
            # If raw FS is not divisible by the target FS
            duration = len(frames) / raw_fs
            new_len = int(duration * self.of_resample_fs)
            old_timestamps = np.linspace(0, duration, len(frames))
            new_timestamps = np.linspace(0, duration, new_len)

            # Select closest frame indices and choose frames and labels
            # May not be strictly evenly spaced
            new_indices = np.searchsorted(old_timestamps, new_timestamps)
            new_indices = np.clip(new_indices, 0, len(frames) - 1)
            new_frames = frames[new_indices]
            new_labels = labels[new_indices]
        return new_frames, new_labels

    def _augment(
        self,
        frames: np.ndarray,
        flow: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply geometric and photometric augmentations to frames (and flow if provided).
        Geometric transforms are applied to both, flow vectors are updated correspondingly.
        Photometric transforms are applied to frames only.

        Args:
            frames: (T, H, W, 3) float32
            flow: (T, H, W, 3) or None
        """
        self.logger.debug("Applying augmentation pipeline:")

        T, H, W, C = frames.shape
        frames_aug = frames.copy()
        flow_aug = None if flow is None else flow.copy()

        # 1. Apply geometric transforms to frames
        # Discrete 90° rotations
        rotation_k = 0
        if self.do_rotation:
            rotation_k = np.random.choice([0, 1, 2, 3])  # 0, 90, 180, 270 CCW
        if rotation_k:
            frames_aug = np.rot90(frames_aug, k=rotation_k, axes=(1,2))  # rotate over H,W

        # Flips
        do_hflip = self.do_horizontal_flip and (np.random.rand() < self.horizontal_flip_p)
        do_vflip = self.do_vertical_flip and (np.random.rand() < self.vertical_flip_p)
        if do_hflip:  # Horizontal Flip
            frames_aug = np.flip(frames_aug, axis=2)  # flip W
        if do_vflip:  # Vertical Flip
            frames_aug = np.flip(frames_aug, axis=1)  # flip H

        # 2. Apply the same geometric transforms to flow and flow vectors
        if flow_aug is not None:
            # Counterclockwise Rotation
            if rotation_k:
                flow_aug = np.rot90(flow_aug, k=rotation_k, axes=(1,2))

                # Rotate vectors (dx,dy) for k*90°
                dx = flow_aug[..., 0].copy()
                dy = flow_aug[..., 1].copy()
                if rotation_k == 1:  # 90°  CCW
                    flow_aug[..., 0] = dy
                    flow_aug[..., 1] = -dx
                elif rotation_k == 2:  # 180°
                    flow_aug[..., 0] = -dx
                    flow_aug[..., 1] = -dy
                elif rotation_k == 3:  # 270°  CCW
                    flow_aug[..., 0] = -dy
                    flow_aug[..., 1] = dx

            # Flips
            if do_hflip:
                flow_aug = np.flip(flow_aug, axis=2)
                flow_aug[..., 0] *= -1.0  # dx -> -dx
            if do_vflip:
                flow_aug = np.flip(flow_aug, axis=1)
                flow_aug[..., 1] *= -1.0  # dy -> -dy

            # Recompute magnitude if exists
            if flow_aug.shape[-1] == 3:
                flow_aug[..., 2] = np.sqrt(flow_aug[..., 0]**2 + flow_aug[..., 1]**2 + 1e-6)

        # 3. Apply photometric transforms on frames only
        photo_transforms = []
        if self.do_brightness_adjustment:
            photo_transforms.append(v2.ColorJitter(brightness=self.brightness_adjustment))
        if self.do_contrast_adjustment:
            photo_transforms.append(v2.ColorJitter(contrast=self.contrast_adjustment))

        if photo_transforms:
            frame_tensors = torch.as_tensor(frames_aug, dtype=torch.float32)  # (T,H,W,3)
            frame_tensors = frame_tensors.permute(0, 3, 1, 2)  # (T, 3, H, W) (required by torchvision.transforms.v2)

            # Scale to [0, 1], jitter, and back to [0, 255]
            frame_tensors = frame_tensors / 255.0
            frame_tensors = v2.Compose(photo_transforms)(frame_tensors)
            frame_tensors = (frame_tensors * 255.0).clamp(0, 255)

            # Convert PyTorch tensors back to numpy frames
            frames_aug = frame_tensors.permute(0, 2, 3, 1).numpy().astype(np.float32)

        return frames_aug, flow_aug

    def _chunk(self, frames: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Chunk the data into smaller chunks."""
        if frames.shape[0] % self.chunk_length != 0:
            self.logger.debug(f"Some frames are dropped to make chunking work. Total frames: {frames.shape[0]}, Chunk length: {self.chunk_length}")
        clip_num = frames.shape[0] // self.chunk_length
        frames_clips = [frames[i * self.chunk_length:(i + 1) * self.chunk_length] for i in range(clip_num)]
        labels_clips = [labels[i * self.chunk_length:(i + 1) * self.chunk_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(labels_clips)

    def _normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Apply frame normalizations."""
        if self.data_normalize_type == "Raw":
            transformed_frames = frames.copy()
        elif self.data_normalize_type == "DiffNormalized":
            transformed_frames = self._diff_normalize(frames.copy())
        elif self.data_normalize_type == "Standardized":
            transformed_frames = self._standardize(frames.copy())
        else:
            self.logger.error(f"Unsupported frame transform type: {self.data_normalize_type}")
            raise ValueError(f"Unsupported frame transform type: {self.data_normalize_type}")

        return transformed_frames

    def _normalize_labels(self, labels: np.ndarray) -> np.ndarray:
        """Apply label normalizations."""
        if self.label_normalize_type == "Raw":
            transformed_labels = labels
        elif self.label_normalize_type == "DiffNormalized":
            transformed_labels = self._diff_normalize(labels)
        elif self.label_normalize_type == "Standardized":
            transformed_labels = self._standardize(labels)
        else:
            self.logger.error(f"Unsupported label transform type: {self.label_normalize_type}")
            raise ValueError(f"Unsupported label transform type: {self.label_normalize_type}")
        return transformed_labels

    def _normalize_flow(self, flow: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Apply flow normalizations (only supports raw or Z-score standardized)."""
        if flow is None:
            return None
        if self.flow_normalize_type == "Raw":
            transformed_flow = flow.copy()
        elif self.flow_normalize_type == "Standardized":
            transformed_flow = self._standardize(flow.copy())
        else:
            self.logger.error(f"Unsupported flow transform type: {self.flow_normalize_type}")
            raise ValueError(f"Unsupported flow transform type: {self.flow_normalize_type}")

        return transformed_flow

    def _diff_normalize(self, data: np.ndarray) -> np.ndarray:
        """Calculate discrete difference and normalize, with NaN prevention."""
        # Check for NaN in input
        if np.isnan(data).any():
            self.logger.debug("NaN values in input data, replacing with zeros")
            data = np.nan_to_num(data, nan=0.0)

        # Deal with frames data
        if data.ndim != 1:
            n, h, w, c = data.shape
            diff_len = n - 1
            diff_data = np.zeros((diff_len, h, w, c), dtype=np.float32)
            diff_data_padding = np.zeros((1, h, w, c), dtype=np.float32)

            # Calculate frame differences
            for j in range(diff_len - 1):
                numerator = (data[j + 1, :, :, :] - data[j, :, :, :])
                denominator = (data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
                diff_data[j, :, :, :] = numerator / denominator

        # Deal with labels data
        else:
            diff_data = np.diff(data, axis=0)
            diff_data_padding = np.zeros(1)

        # Safe normalization
        std = np.std(diff_data)
        if std < 1e-7:
            self.logger.debug("Near-zero standard deviation in diff_normalize_label")
            std = 1.0  # Use 1.0 instead of a very small value

        diff_data /= std
        diff_data = np.append(diff_data, diff_data_padding, axis=0)

        # Replace any remaining NaN or inf values
        return np.nan_to_num(diff_data, nan=0.0, posinf=0.0, neginf=0.0)

    def _standardize(self, data: np.ndarray) -> np.ndarray:
        """Z-score standardization with NaN prevention."""
        # Check for NaN in input
        if np.isnan(data).any():
            self.logger.debug("NaN values in input data for standardization, replacing with zeros")
            data = np.nan_to_num(data, nan=0.0)

        # (T, ) for labels
        if data.ndim == 1:
            mean = np.mean(data, keepdims=True)
            std = np.std(data, keepdims=True)
            std = np.where(std < 1e-7, 1.0, std)  # Use 1.0 instead of a very small value
            return np.nan_to_num((data - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

        # (T, H, W, C) for frames/flow
        mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
        std = np.std(data, axis=(0, 1, 2), keepdims=True)
        std = np.where(std < 1e-7, 1.0, std)
        return np.nan_to_num((data - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

    def run_optical_flow(self, frames: np.ndarray) -> np.ndarray:
        """Dispatch to the configured optical flow backend."""
        self.logger.info(f"Computing optical flow using method: {self.of_method.upper()}")

        if self.of_method == "raft":
            return self.compute_raft_flow(frames)

        if self.of_method in ["farneback", "tvl1", "pca", "deep"]:
            frames = frames if frames.shape[-1] == 1 else self.convert_to_grayscale(frames)
            return self.compute_dense_flow(frames)

        if self.of_method == "coarse2fine":
            return self.compute_pyflow(frames)

    @staticmethod
    def _init_raft_model(device: str) -> torch.nn.Module:
        try:
            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights, progress=False)
        except Exception:
            model = raft_large(pretrained=True, progress=False)
        return model.to(device).eval()

    def compute_raft_flow(self, frames: np.ndarray) -> np.ndarray:
        """Compute optical flow using pretrained RAFT model from torchvision."""
        # Init model on cache to avoid reloading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not hasattr(self, "_raft_model") or self._raft_model is None:
            self._raft_model = self._init_raft_model(device)
        model = self._raft_model

        # RAFT expects resolution multiples of 8 and higher than 128 * 128
        target_h = max(128, ((frames.shape[1] + 8 - 1) // 8) * 8)
        target_w = max(128, ((frames.shape[2] + 8 - 1) // 8) * 8)
        frames = np.stack([
            cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            for f in frames
        ])

        T, H, W, C = frames.shape
        flow_stack = []
        frames_tensor = torch.tensor(frames).permute(0, 3, 1, 2)  # (T, H, W, C) to (T, C, H, W)

        prev = frames_tensor[0].unsqueeze(0).float().to(device)
        for i in tqdm(range(1, T)):
            self.logger.debug(f"Computing RAFT flow for frame {i}")
            curr = frames_tensor[i].unsqueeze(0).float().to(device)
            with torch.no_grad():
                flow = model(prev, curr)[-1][0]  # (2, H, W)
            flow = flow.permute(1, 2, 0).cpu().numpy()  # (2, H, W) to (H, W, 2)
            flow_stack.append(flow)
            prev = curr

        flow = np.stack(flow_stack, axis=0)  # Shape: (T-1, H, W, 2)
        dx, dy = flow[:, :, :, 0], flow[:, :, :, 1]
        mag = np.sqrt(dx**2 + dy**2 + 1e-6)
        flow = np.stack([dx, dy, mag], axis=-1)  # (T-1, H, W, 3)
        flow_padding = np.zeros((1, H, W, 3), dtype=float)  # Add one more frame
        return np.concatenate((flow, flow_padding), axis=0, dtype=np.float32)  # Shape: (T, H, W, 3)

    def compute_dense_flow(self, frames: np.ndarray) -> np.ndarray:
        """Compute dense optical flow using OpenCV methods (Farneback/TVL1/PCA/Deep)."""
        # Squeeze channel if present
        if frames.ndim == 4 and frames.shape[-1] == 1:
            frames = frames[..., 0]  # (T,H,W,1) -> (T,H,W)

        T, H, W = frames.shape
        flow_stack = []

        method, params = self._init_dense_flow_methods()

        frames = frames.astype(np.uint8)
        prev = frames[0]

        for i in tqdm(range(1, T)):
            self.logger.debug(f"Computing {self.of_method.upper()} flow for frame {i}")
            curr = frames[i]
            # Calculate Optical Flow
            if hasattr(method, "calc"):
                flow = method.calc(prev, curr, None)
            else:
                flow = method(prev, curr, None, *params)
            flow_stack.append(flow)
            prev = curr

        flow = np.stack(flow_stack, axis=0)  # (T-1, H, W, 2)
        dx, dy = flow[:, :, :, 0], flow[:, :, :, 1]
        mag = np.sqrt(dx**2 + dy**2 + 1e-6)
        flow = np.stack([dx, dy, mag], axis=-1)  # (T-1, H, W, 3)
        flow_padding = np.zeros((1, H, W, 3), dtype=float)  # Add one more frame
        return np.concatenate((flow, flow_padding), axis=0, dtype=np.float32)  # Shape: (T, H, W, 3)

    def _init_dense_flow_methods(self) -> Tuple[Any, List[Any]]:
        params = []

        if self.of_method == "farneback":
            method = cv2.calcOpticalFlowFarneback  # 1-channel input
            params = [0.5, 3, 15, 3, 5, 1.2, 0]  # Default params for Farneback

        elif self.of_method == "tvl1":  # 1-channel input, TODO: 3-channel not sure
            method = cv2.optflow.createOptFlow_DualTVL1()

        elif self.of_method == "pca":
            method = cv2.optflow.createOptFlow_PCAFlow()  # 1-channel input

        elif self.of_method == "deep":
            method = cv2.optflow.createOptFlow_DeepFlow()  # 1-channel input

        else:
            self.logger.error(f"Unsupported Optical Flow algorithm: {self.of_method}")
            raise NotImplementedError(f"Unsupported Optical Flow algorithm: {self.of_method}")
        return method, params

    def compute_pyflow(self, frames: np.ndarray) -> np.ndarray:
        """Compute optical flow using coarse2fine method in pyflow library."""
        T, H, W, C = frames.shape
        flow_stack = []

        # Frame values must be resized to 0.0 - 1.0
        frames = [
            np.clip(f / 255.0, 0.0, 1.0).astype(float)
            for f in frames
        ]

        # Ensure grayscale frame has the required shape (H, W, C)
        if self.pyflow_col_type == 1:
            frames = [f[:, :, 0:1] for f in frames]

        # Prepare shared args for PyFlow
        pyflow_args = (
            self.pyflow_alpha,
            self.pyflow_ratio,
            self.pyflow_min_width,
            self.pyflow_n_outer_FP_iterations,
            self.pyflow_n_inner_FP_iterations,
            self.pyflow_n_SOR_iterations,
            self.pyflow_col_type,
        )

        # Prepare frame pairs along with pyflow args
        input_triples = [(frames[i - 1], frames[i], pyflow_args) for i in range(1, T)]
        if T < 2 or not input_triples:
            self.logger.debug(f"PyFlow: Not enough valid frame pairs (T={T})")
            return np.zeros((1, H, W, 3), dtype=np.float32)

        # Use multiprocessing to compute optical flow for all pairs
        n_workers = min(len(input_triples), cpu_count())
        with Pool(processes=n_workers) as pool:
            try:
                flow_stack = list(tqdm(
                    pool.imap(
                        compute_pyflow_single_pair,
                        input_triples,
                        chunksize=len(input_triples) // n_workers
                    ),
                    total=len(input_triples)
                ))
            except Exception:
                self.logger.exception(f"Failed to compute PyFlow")

        flow = np.stack(flow_stack, axis=0)  # (T-1, H, W, 2)
        dx, dy = flow[:, :, :, 0], flow[:, :, :, 1]
        mag = np.sqrt(dx**2 + dy**2 + 1e-6)
        flow = np.stack([dx, dy, mag], axis=-1)  # (T-1, H, W, 3)
        flow_padding = np.zeros((1, H, W, 3), dtype=float)  # Add one more frame
        return np.concatenate((flow, flow_padding), axis=0, dtype=np.float32)  # (T, H, W, 3)

    def _ensure_three_channels(self, data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        c = data.shape[-1]
        if c == 3:
            return data
        if c == 1:
            return np.repeat(data, 3, axis=-1)  # grayscale → RGB
        self.logger.error(f"Unexpected channel count: {c}")
        raise ValueError(f"Unexpected channel count: {c}")

    @staticmethod
    def flow_to_bgr(flow: np.ndarray) -> np.ndarray:
        """Convert optical flow (dx, dy) to BGR format (H, W, 3)."""
        dx = flow[..., 0].astype(np.float32)
        dy = flow[..., 1].astype(np.float32)

        # magnitude and angle
        mag = np.sqrt(dx**2 + dy**2)  # (T,H,W)
        T, H, W = mag.shape
        ang = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)  # [0, 2π)
        v_all =  cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Hue from angle; OpenCV HSV expects H in [0,180)
        h_all = (ang * 180.0 / np.pi / 2.0).astype(np.uint8)
        s_all = np.full((T, H, W), 255, dtype=np.uint8)

        bgr_out = np.empty((T, H, W, 3), dtype=np.uint8)

        for t in range(T):
            hsv = np.stack([h_all[t], s_all[t], v_all[t]], axis=-1)  # (H,W,3) uint8
            bgr_out[t] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # BGR uint8

        return bgr_out.astype(np.float32)  # [0,255] float32


def compute_pyflow_single_pair(args: Tuple[np.ndarray, np.ndarray, Tuple[Any, ...]]) -> np.ndarray:
    """Compute flow for a pair of frames using PyFlow (multiprocessing safe)."""
    prev, curr, pyflow_args = args
    with _suppress_pyflow_output():
        u, v, _ = pyflow.coarse2fine_flow(prev, curr, *pyflow_args)
    return np.stack((u, v), axis=-1)
