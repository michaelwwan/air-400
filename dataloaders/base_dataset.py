import glob
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from processors.pre_processor import PreProcessor


class BaseDataset(Dataset, ABC):
    """Abstract base class for AIR 125, AIR 400, and COHFACE datasets."""

    def __init__(self, config, split):
        """
        Initialize the dataset.

        Args:
            config: Configuration dictionary
            split (str): 'TRAIN', 'VALID', or 'TEST'
        """

        super().__init__()

        self.config = config
        self.logger = logging.getLogger(__name__)
        self.split = split

        self._load_params()

        # Initialize pre processor
        self.pre_processor = PreProcessor(self.preprocess_config, self.body_detector_path, self.face_detector_path)
        self.preprocess_param_dict, self.preprocess_hash = self.pre_processor.hash_param_dict()

        # Initialize input and label paths
        self.inputs = []
        self.labels = []
        self.fs_list = []

        # Create output and cache directories
        self._create_directories()

        # Get raw data directories
        self.data_dirs = self.get_raw_data(self.data_path)

        # Split data by subject
        self.subjects = self.get_unique_subjects(self.data_dirs)
        self.subjects_for_split = self._get_split_subjects()

        # Get the appropriate data directories for this split
        self.dirs_for_split = [d for d in self.data_dirs if d["subject"] in self.subjects_for_split]

        # Load and check current config metadata file or create a new one
        self._load_metadata()

        # Change FS accordingly if resample flow FS in preprocessing
        of_resample_fs = self.preprocess_param_dict['OF_RESAMPLE_FS']
        if of_resample_fs > 0:
            for d in self.dirs_for_split:
                d['fs'] = of_resample_fs

        # Preprocess or load data
        self._auto_preprocess_dataset()

        self.logger.info(f"{self.dataset} Dataset {split} Split: {len(self.subjects_for_split)} preprocessed subjects ({len(self.dirs_for_split)} clips in {len(self.inputs)} chunks) loaded")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.inputs)

    def __getitem__(self, index):
        """Return a data sample by index."""
        # Load data and label
        data = np.load(self.inputs[index])
        label = np.load(self.labels[index])

        # Handle data format
        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass  # Already in this format
        else:
            self.logger.error(f"Unsupported data format: {self.data_format}")
            raise ValueError(f"Unsupported data format: {self.data_format}")

        # Convert to float32
        data = np.float32(data)
        label = np.float32(label)

        # Extract filename and chunk_id
        item_path = self.inputs[index]
        filename, chunk_id = self._get_filename(item_path)

        fs = self.fs_list[index]

        return data, label, filename, chunk_id, self.dataset, fs

    @staticmethod
    def get_unique_subjects(data_dirs):
        """Get unique subject IDs."""
        return sorted(list(set([d["subject"] for d in data_dirs])))

    def _load_params(self):
        data_config = self.config[self.split]['DATA']
        self.dataset = data_config['DATASET']  # 'AIR_125', 'AIR_400', or 'COHFACE'
        self.data_path = self.config['DATA_PATH'][self.dataset]  # Path to the raw data directory
        self.split_type = data_config['SPLIT_TYPE']  # 'custom' or 'ratio'
        self.split_subjects = data_config['SPLIT_SUBJECTS'] \
            if self.split_type == 'custom' else []  # Custom List of subject IDs for current split
        self.split_ratio_start, self.split_ratio_end = self._calculate_split_range()  # Start and end of current split range
        self.data_format = data_config['DATA_FORMAT']  # PyTorch format (batch, channels, depth, height, width) ('NDHWC', 'NDCHW', 'NCDHW')
        self.body_detector_path = self.config['DATA_PATH']['BODY_DETECTOR_PATH']
        self.face_detector_path = self.config['DATA_PATH']['FACE_DETECTOR_PATH']
        self.preprocess_config = data_config['PREPROCESS']

    def _create_directories(self):
        """Create model and cache directories for the dataset."""

        self.cache_dir = os.path.join(
            str(self.config['DATA_PATH']['CACHE_DIR']),
            self.dataset,
            str(self.preprocess_hash)
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        # Save preprocess configs in a file within each cache_dir
        self.meta_path = os.path.join(self.cache_dir, 'meta.json')

    def _calculate_split_range(self):
        """Calculate split ranges (start, end) for current split."""
        # Use full dataset as split range
        if self.split_type not in ['custom', 'ratio']:
            self.logger.error("Split_TYPE must be either 'custom' or 'ratio'")
            raise ValueError("Split_TYPE must be either 'custom' or 'ratio'")

        # No need to split when specifying subjects
        if self.split_type == 'custom':
            return None, None

        if self.split_type == 'ratio':
            # Traverse all splits to calculate split ranges
            split_params = defaultdict(dict)
            for phase_split in ['TRAIN', 'VALID', 'TEST']:
                phase_cfg = self.config[phase_split]['DATA']
                if phase_cfg is None or phase_cfg['DATASET'] != self.dataset or phase_cfg['SPLIT_TYPE'] != 'ratio':
                    continue
                split_params[phase_split] = phase_cfg['SPLIT_RATIO']

            # Check split of current dataset
            sum_split_ratios = sum(split_params.values())
            if abs(sum_split_ratios - 1.0) > 1e-6:
                self.logger.warning(f"Sum of split ratios for dataset {self.dataset} not equal to 1.0: {sum_split_ratios}")

            # Calculate split range for current split
            cursor = 0.0
            for phase_split, phase_ratio in split_params.items():
                if phase_split == self.split:
                    break
                cursor += phase_ratio

            return cursor, cursor + self.config[self.split]['DATA']['SPLIT_RATIO']

    def _get_split_subjects(self):
        """Split subjects into train, validation, and test sets."""
        # Split by custom subject list
        if self.split_type == 'custom':
            if not isinstance(self.split_subjects, list):
                self.logger.error("SLIT_SUBJECTS must be a list of subjects")
                raise TypeError("SLIT_SUBJECTS must be a list of subjects")
            # Validate that all specified subjects exist in the dataset
            all_subjects = set(self.subjects)
            for subject in self.split_subjects:
                if subject not in all_subjects:
                    self.logger.warning(f"Subject {subject} specified in custom split ({self.split_type}) not found in dataset")
            return self.split_subjects

        # Random split based on split_ratio
        if self.split_type == 'ratio':
            subjects = np.array(self.subjects)
            n = len(subjects)

            # Shuffle subjects
            np.random.shuffle(subjects)

            # Calculate start and end indices for current split
            start_idx = int(n * self.split_ratio_start)
            end_idx = int(n * self.split_ratio_end)

            return subjects[start_idx:end_idx]

    def _auto_preprocess_dataset(self):
        """Preprocess all data for the current split."""
        self.logger.info(f"\n====Preprocessing====")
        self.logger.info(f"Preprocessing {len(self.dirs_for_split)} clips in {self.split} split of {self.dataset} dataset:")
        self.logger.info(f"{[d['index'] for d in self.dirs_for_split]}")

        need_preprocess = []
        preprocessed = []

        for data_dir in self.dirs_for_split:
            # Search in metadata to check if subject has not been preprocessed
            if data_dir['index'] not in self.metadata['PREPROCESSED_INDICES']:
                need_preprocess.append(data_dir)

        if need_preprocess:
            self.logger.info(f"Found {len(need_preprocess)} clips need to be preprocessed. Preprocessing:")
            self.logger.info(f"{[d['index'] for d in need_preprocess]}")
            self._preprocess_subjects(need_preprocess)
        
        for data_dir in self.dirs_for_split:
            # Search in metadata to check if subject has been preprocessed
            if data_dir['index'] in self.metadata['PREPROCESSED_INDICES']:
                preprocessed.append(data_dir)
        
        preprocessed.sort(key=lambda d: d['index'])
        if preprocessed:
            self.logger.info(f"Found {len(preprocessed)} preprocessed clips. Loading from cached dir:")
            self.logger.info(f"{[d['index'] for d in preprocessed]}")
            self._load_preprocessed_subjects(preprocessed)

    def _load_metadata(self):
        """
        Load config from existing metadata file and compare with current config.
        If metadata file does not exist, create a new one to save current config.
        """
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)
            if self.metadata['HASH'] != self.preprocess_hash:
                self.logger.error("Config stored in meta file mismatches current config.")
                raise ValueError("Config stored in meta file mismatches current config.")
        else:
            self.metadata = {
                'HASH': self.preprocess_hash,
                'PREPROCESS': self.preprocess_param_dict,
                'PREPROCESSED_INDICES': []
            }

            with open(self.meta_path, "w") as f:
                json.dump(self.metadata, f, indent=2)

    def _load_preprocessed_subjects(self, preprocessed):
        """Load preprocessed subjects for current preprocess config."""
        def _get_chunk_id(path):
            m = re.search(r'_input_(\d+)\.npy$', os.path.basename(path))
            return int(m.group(1)) if m else -1

        for data_dir in preprocessed:
            file_idx = data_dir['index']
            # Match pattern for all input files from this subject
            input_pattern = os.path.join(self.cache_dir, f"{file_idx}_input_*.npy")
            input_files = sorted(glob.glob(input_pattern), key=lambda x: _get_chunk_id(x))
            label_files = [f.replace("input", "label") for f in input_files]

            self.inputs.extend(input_files)
            self.labels.extend(label_files)
            self.fs_list.extend(len(input_files) * [data_dir['fs']])

    def _preprocess_subjects(self, need_preprocess):
        """Preprocess and save subjects with current preprocess config."""
        for data_dir in tqdm(need_preprocess):
            file_idx = data_dir['index']
            fs = data_dir['fs']

            # Read video and wave files
            frames, labels = self._read_video_wave_files(data_dir)

            # Resample RR at video FPS
            target_length = frames.shape[0]
            labels = self._resample_ppg(labels, target_length)

            # Preprocess data
            frames_clips, labels_clips = self.pre_processor.preprocess(frames, labels, fs)

            # Save preprocessed data
            self._save_chunks(frames_clips, labels_clips, file_idx)

    def _save_chunks(self, frames_clips, labels_clips, file_idx):
        """Save preprocessed data chunks."""
        for i in range(len(labels_clips)):
            input_path = os.path.join(self.cache_dir, f"{file_idx}_input_{i:03d}.npy")
            label_path = os.path.join(self.cache_dir, f"{file_idx}_label_{i:03d}.npy")

            try:
                np.save(input_path, frames_clips[i])
                np.save(label_path, labels_clips[i])
            except Exception:
                self.logger.exception(f"Failed to save preprocessed chunk {i} for subject {file_idx}")

        self.logger.debug(f"Saved {len(labels_clips)} clips for subject {file_idx} to cache dir: {self.cache_dir}")

        # Update metadata to record preprocessed data chunks
        self.metadata['PREPROCESSED_INDICES'].append(file_idx)
        with open(self.meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    @staticmethod
    def _get_filename(item_path):
        """Extract filename and chunk_id."""
        item_path_filename = os.path.basename(item_path)
        split_idx = item_path_filename.rindex('_input_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + 7:].split('.')[0]
        return filename, chunk_id

    def _read_video(self, video_file):
        """Read video file and return frames (T,H,W,3)."""
        self.logger.debug(f"Reading video file {video_file}")
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = []

        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0
            frames.append(frame)
            success, frame = VidObj.read()

        return np.asarray(frames)

    def _read_wave(self, label_file):
        """Read wave file and return respiration signal."""
        self.logger.debug(f"Reading wave file {label_file}")
        f = h5py.File(label_file, 'r')
        # For AIR dataset, you might need to adjust the key based on your HDF5 structure
        # Assuming the key is 'respiration', same as COHFACE
        resp = f['respiration'][:]
        return resp

    @staticmethod
    def _resample_ppg(input_signal, target_length):
        """Resample PPG signal to match video length."""
        return np.interp(
            np.linspace(1, input_signal.shape[0], target_length),
            np.linspace(1, input_signal.shape[0], input_signal.shape[0]),
            input_signal
        )

    def _read_video_wave_files(self, data_dir):
        """Read video to get frames, read wave to get labels."""
        frames = self._read_video(data_dir['video_path'])
        labels = self._read_wave(data_dir['label_path'])
        return frames, labels

    # Abstract methods that must be implemented by subclasses
    @classmethod
    @abstractmethod
    def get_raw_data(cls, data_path):
        """Get raw data directories."""
        pass
