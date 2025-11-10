import logging
import os
import cv2
import numpy as np
from torch.utils.data import Dataset

from processors.pre_processor import PreProcessor


class InferenceDataset(Dataset):
    def __init__(self, config, video_path):

        super().__init__()

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Use the provided path for this dataset instance
        self.video_path = video_path
        self.body_detector_path = self.config['DATA_PATH']['BODY_DETECTOR_PATH']
        self.face_detector_path = self.config['DATA_PATH']['FACE_DETECTOR_PATH']
        # Resolve split data config (supports list or dict)
        data_cfg = config['TEST']['DATA'][0] if isinstance(config['TEST']['DATA'], list) else config['TEST']['DATA']

        self.infant_flag = data_cfg['INFANT_FLAG']
        self.data_format = data_cfg['DATA_FORMAT']
        self.preprocess_config = data_cfg['PREPROCESS']

        self.body_detector_path = config['DATA_PATH']['BODY_DETECTOR_PATH']
        self.face_detector_path = config['DATA_PATH']['FACE_DETECTOR_PATH']

        self.pre_processor = PreProcessor(self.preprocess_config, self.body_detector_path, self.face_detector_path)

        frames, fps = self._read_video(self.video_path)
        self.fs = fps
        labels = np.zeros((frames.shape[0],), dtype=np.float32)  # dummy labels

        frames_clips, labels_clips = self.pre_processor.preprocess(frames, labels, self.fs)
        self.frames_clips = frames_clips
        self.labels_clips = labels_clips
        self.filename = os.path.splitext(os.path.basename(self.video_path))[0]

    def __len__(self):
        return self.frames_clips.shape[0]

    def __getitem__(self, idx):
        data = self.frames_clips[idx]  # (D,H,W,6)
        label = self.labels_clips[idx]  # (D,)

        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError(f"Unsupported data format: {self.data_format}")

        data = np.float32(data)
        label = np.float32(label)

        return data, label, self.filename, f"{idx:03d}", self.infant_flag, self.fs

    def _read_video(self, video_file):
        self.logger.debug(f"Reading video file {video_file}")
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        fs = int(round(VidObj.get(cv2.CAP_PROP_FPS)))
        success, frame = VidObj.read()
        frames = []

        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0
            frames.append(frame)
            success, frame = VidObj.read()
        VidObj.release()
        if len(frames) == 0:
            raise RuntimeError(f"No frames decoded from: {video_file}")
        return np.asarray(frames), fs
