"""AIR-400 dataset reader."""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, Sequence

import cv2

from dataloaders.base_dataset import BaseDataset


class AIR400Dataset(BaseDataset):
    """
    AIR-400 dataset layout (video at root, labels under out/).

    Dataset structure:
    /
    S01/
    |   |-- 1.mp4
    |   |-- 2.mp4
    |   |-- 3.mp4
    |...
    |   |-- n.mp4
    |   |-- out/
    |       |-- 1.hdf5
    |       |-- 2.hdf5
    |       |-- 3.hdf5
    |       |..
    |       |-- n.hdf5
    S02/
    ...
    """

    @classmethod
    def get_raw_data(cls, data_path: str) -> Sequence[Dict[str, Any]]:
        """Enumerate raw clip metadata for AIR-400."""
        logger = logging.getLogger(cls.__name__)

        subject_dirs = glob.glob(os.path.join(data_path, "S*"))
        if not subject_dirs:
            logger.error("AIR 400 data paths empty!")
            raise ValueError("AIR 400 data paths empty!")

        logger.info(f"Found {len(subject_dirs)} total subjects for AIR_400 dataset")

        dirs = []
        for subject_dir in subject_dirs:
            # Extract subject ID (e.g., 'S01' -> 'S01')
            subject = os.path.basename(subject_dir)
            
            # Skip non-subject directories
            if not os.path.isdir(subject_dir) or not subject.startswith('S'):
                logger.debug(f"Skipping non-subject item: {subject}")
                continue
            
            # Get all MP4 files in the subject directory
            mp4_files = glob.glob(os.path.join(subject_dir, "*.mp4"))
            
            for mp4_file in mp4_files:
                # Extract recording ID (number without extension)
                recording_id = os.path.splitext(os.path.basename(mp4_file))[0]

                # Check if corresponding HDF5 file exists
                hdf5_file = os.path.join(subject_dir, "out", f"{recording_id}.hdf5")
                if not os.path.exists(hdf5_file):
                    logger.debug(f"Skipping {mp4_file}: No corresponding HDF5 file found")
                    continue

                # Extract frame rate
                try:
                    cap = cv2.VideoCapture(mp4_file)
                    fs = int(round(cap.get(cv2.CAP_PROP_FPS)))
                    cap.release()
                except Exception:
                    logger.exception(f"Failed to read FPS for {mp4_file}")
                    continue

                dirs.append({
                    "index": f"{subject}_{int(recording_id):03d}",  # Unique identifier
                    "subject": subject,  # Subject ID (e.g., 'S01')
                    "video_path": mp4_file,  # Path to video file
                    "label_path": hdf5_file,  # Path to label file
                    "fs": fs
                })

        if not dirs:
            logger.error("No valid AIR 400 dataset files found!")
            raise ValueError("No valid AIR 400 dataset files found!")
            
        return dirs
