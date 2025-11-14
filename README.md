<h1 align="center">Overcoming Small Data Limitations in Video-Based Infant Respiration Estimation & AIR-400 Dataset</h1>

<p align="center">
<b>WACV 2026</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/0000.00000">
    <img src="https://img.shields.io/badge/arXiv-0000.00000-b31b1b.svg?style=flat-square">
  </a>
  <a href="https://github.com/michaelwwan/air-400">
    <img src="https://img.shields.io/badge/Github-michaelwwan/air--400-black.svg?logo=github&style=flat-square">
  </a>
  <a href="https://drive.google.com/drive/u/1/folders/12BCJ2TNjAquMHTr3A60p2sQJ9Gp7CRDt">
    <img src="https://img.shields.io/badge/Dataset-Google%20Drive-blue.svg?logo=google-drive&style=flat-square">
  </a>
  <a href="https://drive.google.com/drive/u/1/folders/1kjSAF9Dt24D670cwBgc-uXCz8WYTaulq">
    <img src="https://img.shields.io/badge/Model-Checkpoint-orange.svg?logo=google-cloud&style=flat-square">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square">
  </a>
</p>

---

This is the official repository of our **WACV 2026** paper:

> Song, L.\*, Bishnoi, H.\*, Manne, S.K.R., Ostadabbas, S., Taylor, B.J., Wan, M., "Overcoming Small Data Limitations in Video-Based Infant Respiration Estimation" (*equal contribution). Under review, Preprint available on arXiv.

Here we provide our model code, training checkpoints, and annotated dataset to support automatic estimation of infant respiration waveforms and respiration rate from natural video footage, with the help of spatiotemporal computer vision models and infant-specific region-of-interest tracking. 

## 📋 Table of Contents
- [Requirements & Setup](#-requirements--setup)
- [Quickstart: Inference](#-quickstart-inference)
- [Annotated Infant Respiration Dataset (AIR-400)](#-annotated-infant-respiration-dataset-air-400)
- [Reproducing Paper Results](#-reproducing-paper-results)
- [Citation](#-citation)
- [License](#-license)

#### 🎥 Sample Dataset Preprocessing
<video src="https://github.com/user-attachments/assets/4b7b4703-9163-4812-b380-84225971f0e7" autoplay loop muted playsinline></video>

## 📦 Requirements & Setup 

<a href="https://anaconda.org/anaconda/conda/files?version=25.1.1">
  <img src="https://img.shields.io/badge/Conda-25.1.1-44A833.svg?logo=anaconda&style=flat-square">
</a>
<a href="https://www.python.org/downloads/release/python-3918">
  <img src="https://img.shields.io/badge/Python-3.9.18-blue.svg?logo=python&style=flat-square">
</a>

#### 1. Set up the environment
```bash
conda env create -f environment.yml
```

#### 2. Compile [pyflow](https://github.com/pathak22/pyflow) library and import it as a module
```bash
git clone https://github.com/pathak22/pyflow.git
(cd pyflow && python setup.py build_ext -i && mv pyflow.cpython-*.so ..)
```

## ⚡ Quickstart: Inference

#### Sample Inference Output

Predicted respiration waveform from a sample infant video after preprocessing and model inference.

<video src="https://github.com/user-attachments/assets/7898794e-3223-4c57-9565-410497a466c9" autoplay loop muted playsinline></video>
<img src="https://github.com/user-attachments/assets/b8c66ec4-a379-4d50-94ff-c08689228af7" alt="Demo Waveform Plot" width="70%" />

Use `infer.py` to preprocess input video(s) and run a trained model for respiration rate estimation.

#### 1. Preparation
- Download [trained model](https://drive.google.com/drive/folders/1kjSAF9Dt24D670cwBgc-uXCz8WYTaulq?usp=drive_link) and [ROI detector](https://drive.google.com/drive/folders/1k0BHMGXAXIdmOYyt3iGzbUBH_sEGVcAk?usp=drive_link) files.
- Fill the YAML `DATA_PATH` fields. 
  - Set paths for cache directory and output directory.
  - Set valid detector paths (YOLO weights) if ROI cropping is enabled. Otherwise, set `DO_CROP_INFANT_REGION: False`.
  - Set input video path.

```yaml
DATA_PATH:
  OUTPUT_DIR: /absolute/path/to/output_dir/
  BODY_DETECTOR_PATH: /absolute/path/to/yolov8m.pt
  FACE_DETECTOR_PATH: /absolute/path/to/yolov8n-face.pt
  # Provide exactly one of the following:
  VIDEO_FILE: /absolute/path/to/video.mp4
  # VIDEO_DIR: /absolute/path/to/videos/
```

#### 2. Example run

```bash
python infer.py \
  --config configs/inference/virenet_coarse2fine_infer.yaml \
  --checkpoint checkpoints/virenet_coarse2fine_body.pth \
```

#### 3. Outputs
- Per-video JSON under `OUTPUT_DIR/inference/{video}_{datetime}` with prediction result JSON file and generated artifacts (HDF5 format time series and PNG format waveform plots).
- A summary JSON across all processed videos (`summary_*.json`).
- Logs saved under `OUTPUT_DIR/logs/`.


## 📚 Annotated Infant Respiration Dataset (AIR-400)

The [**AIR-400** dataset](https://drive.google.com/drive/folders/12BCJ2TNjAquMHTr3A60p2sQJ9Gp7CRDt?usp=drive_link) consists of two parts:

- **AIR_125** — legacy dataset (8 subjects)

- **AIR_400** — newly collected dataset (10 subjects)

Each subject directory contains synchronized **video files (.mp4)** and **breathing signal annotations (.hdf5)**.

In the `AIR_125` folder, each subject directory (`S01`, `S02`, ... `S08`) includes paired video and annotation files:
```
AIR_125/
    S01/
    │-- 001.mp4
    │-- 001.hdf5
    │-- 002.mp4
    │-- 002.hdf5
    │   ...
    │-- n.mp4
    │-- n.hdf5
    │
    S02/
    │-- 001.mp4
    │-- 001.hdf5
    │   ...
    ...

```

In the AIR_400 folder, annotation files are stored separately inside each subject's `out/` directory:
```
AIR_400/
    S01/
    │-- 001.mp4
    │-- 002.mp4
    │-- 003.mp4
    │   ...
    │-- n.mp4
    │
    │-- out/
    │    │-- 001.hdf5
    │    │-- 002.hdf5
    │    │-- 003.hdf5
    │    │   ...
    │    │-- n.hdf5
    │
    S02/
    │-- 001.mp4
    │-- ...
    │-- out/
    │    │-- 001.hdf5
    │    ...
    ...

```

## 🔬 Reproducing Paper Results

#### 1. (Optional) Sign W&B and login to record training results
```bash
export WANDB_API_KEY=<your_api_key>
wandb login
```
Set `USE_WANDB: True` in yaml file.

#### 2. Download [AIR-400 dataset](https://drive.google.com/drive/folders/12BCJ2TNjAquMHTr3A60p2sQJ9Gp7CRDt?usp=drive_link) and [ROI detector](https://drive.google.com/drive/folders/1k0BHMGXAXIdmOYyt3iGzbUBH_sEGVcAk?usp=drive_link) files.

#### 3. Fill the YAML `DATA_PATH` fields.

```yaml
DATA_PATH:
  AIR_125: [air-125-dir-path]
  AIR_400: [air-400-dir-path]
  COHFACE: [cohface-dir-path]
  CACHE_DIR: [your-cache-dir]
  OUTPUT_DIR: [your-output-dir]
  BODY_DETECTOR_PATH: [yolov8-path]
  FACE_DETECTOR_PATH: [yolov8-face-path]
```

#### 4. Preprocess the data
Specify required config yaml file path in `run.sh`. Then uncomment `--preprocess` after `python main.py --config "$CONFIG"` to enable preprocess-only mode. Run this approach first to make sure dataset is preprocessed correctly before following training and testing.
```bash
./run.sh
```

#### 5. Start training and testing process
Comment out `--peprocess` after `python main.py --config "$CONFIG"` in `run.sh` to start training and testing process.
```bash
./run.sh
```

## 📝 Citation

```bibtex
@misc{song_bishnoi_overcoming_2025,
	title = {Overcoming {Small} {Data} {Limitations} in {Video}-{Based} {Infant} {Respiration} {Estimation}},
	url = {https://arxiv.org/abs/0000.00000},
	author = {Song, Liyang and Bishnoi, Hardik and Manne, Sai Kumar Reddy and Ostadabbas, Sarah and Taylor, Brianna J and Wan, Michael},
	year = {2025},
}
```

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
