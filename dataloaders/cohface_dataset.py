# Import the simplified COHFACE dataloader we created earlier
# This file contains the COHFACEDataset class that was previously implemented
import logging
import os
import glob

import cv2

from dataloaders.base_dataset import BaseDataset


class COHFACEDataset(BaseDataset):
    """
    PyTorch Dataset for the COHFACE dataset.
    
    Dataset structure:
    RawData/
    |   |-- 1/
    |      |-- 0/
    |          |-- data.avi
    |          |-- data.hdf5
    |      |...
    |      |-- 3/
    |          |-- data.avi
    |          |-- data.hdf5
    |...
    |   |-- n/
    |      |-- 0/
    |          |-- data.avi
    |          |-- data.hdf5
    |      |...
    |      |-- 3/
    |          |-- data.avi
    |          |-- data.hdf5
    """

    @classmethod
    def get_raw_data(cls, data_path):
        """Get raw data directories."""
        logger = logging.getLogger(cls.__name__)

        data_dirs = glob.glob(os.path.join(data_path, "*"))
        if not data_dirs:
            logger.error("COHFACE data paths empty!")
            raise ValueError("COHFACE data paths empty!")

        logger.info(f"Found {len(data_dirs)} total subjects for COHFACE dataset")

        dirs = []
        for data_dir in data_dirs:
            # Only process directories with numeric names (subjects)
            subject = os.path.split(data_dir)[-1]
            
            # Skip non-numeric directories
            if not os.path.isdir(data_dir) or not subject.isdigit():
                logger.debug(f"Skipping non-subject item: {subject}")
                continue
                
            for i in range(4):  # Each subject has 4 recordings
                avi_file = os.path.join(data_dir, str(i), "data.avi")
                hdf5_file = os.path.join(data_dir, str(i), "data.hdf5")

                # Check if corresponding HDF5 file exists
                if not os.path.exists(hdf5_file):
                    logger.debug(f"Skipping {avi_file}: No corresponding HDF5 file found")
                    continue

                # Extract frame rate
                try:
                    cap = cv2.VideoCapture(avi_file)
                    fs = int(round(cap.get(cv2.CAP_PROP_FPS)))
                    cap.release()
                except Exception:
                    logger.exception(f"Failed to read FPS for {avi_file}")
                    continue

                dirs.append({
                    "index": f"{subject}_{i}",
                    "subject": subject,
                    "video_path": avi_file,
                    "hdf5_path": hdf5_file,
                    "fs": fs
                })
        
        if not dirs:
            logger.error("No valid COHFACE data directories found!")
            raise ValueError("No valid COHFACE subject directories found!")
            
        return dirs
