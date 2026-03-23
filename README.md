<h1 align="center">Overcoming Small Data Limitations in Video-Based Infant Respiration Estimation & AIR-400 Dataset</h1>

<p align="center">
<b>WACV 2026</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.06888"><img src="https://img.shields.io/badge/arXiv-2512.06888-b31b1b.svg?logo=arXiv&logoColor=white&style=flat-square"></a>
  <a href="https://wacv.thecvf.com/virtual/2026/poster/508"><img src="https://img.shields.io/badge/WACV-2026-e19c6a.svg?style=flat-square"></a>
  <a href="https://github.com/michaelwwan/air-400"><img src="https://img.shields.io/badge/Github-michaelwwan/air--400-black.svg?logo=github&logoColor=white&style=flat-square"></a>
</p>
<p align="center">
  <a href="https://coe.northeastern.edu/Research/AClab/AIR-400/"><img src="https://img.shields.io/badge/Dataset-AClab%20Drive-1f6feb.svg?logo=google-drive&logoColor=white&style=flat-square"></a>
  <a href="https://coe.northeastern.edu/Research/AClab/AIR-400/Model%20Checkpoints/"><img src="https://img.shields.io/badge/Model-Checkpoint-orange.svg?logo=google-cloud&logoColor=white&style=flat-square"></a>
  <a href="https://wacv.thecvf.com/media/PosterPDFs/WACV%202026/508.png?t=1771517236.1543777"><img src="https://img.shields.io/badge/Poster-WACV%202026-145a6c.svg?style=flat-square"></a>
  <a href="https://youtu.be/qhQWZ8Oco80"><img src="https://img.shields.io/badge/YouTube-Presentation-red.svg?logo=YouTube&logoColor=white&style=flat-square"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square"></a>
</p>

This is the official repository of our **WACV 2026** paper:

> Song, L.\*, Bishnoi, H.\*, Manne, S.K.R., Ostadabbas, S., Taylor, B.J., Wan, M., "**Overcoming Small Data Limitations in Video-Based Infant Respiration Estimation**" (*equal contribution). 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). [[arXiv](https://arxiv.org/abs/2512.06888)]

Here we provide our **code**, **training checkpoints**, and the **AIR-400 annotated dataset** for estimating **infant respiration waveforms and respiration rate** from natural video footage, using spatiotemporal computer vision models and infant-specific region-of-interest tracking. 

## ✨ Highlights

- Introduces **AIR-400**, a large-scale annotated dataset for video-based infant respiration estimation.
- Supports estimation of both **respiration waveform** and **respiration rate** from natural infant videos.
- Includes **training / inference code**, **model checkpoints**, and **ROI detectors**.
- Provides a **WACV 2026 poster** and **recorded presentation** for quick overview.

<p align="center">
  <a href="https://wacv.thecvf.com/media/PosterPDFs/WACV%202026/508.png?t=1771517236.1543777" title="WACV Poster">
    <img src="https://github.com/user-attachments/assets/ab3b7890-e5b8-4fa4-8e27-090a6a7e7062" alt="WACV Poster" width="80%"/>
  </a>
</p>
<p align="center">
<i>WACV Poster</i>
</p>

<p align="center">
  <a href="https://youtu.be/qhQWZ8Oco80" title="[WACV 2026] Overcoming Small Data Limitations in Video-Based Infant Respiration Estimation">
    <img src="https://img.youtube.com/vi/qhQWZ8Oco80/maxresdefault.jpg" alt="Recorded Presentation on YouTube" width="60%">
  </a>
</p>
<p align="center">
<i>Recorded Presentation on YouTube</i>
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/e2aacfc2-fa0d-4e4d-b03f-65e555bd81a2" alt="Sample Dataset Preprocessing" width="60%"></img>
</p>
<p align="center">
<i>Sample Dataset Preprocessing</i>
</p>

---

## 📋 Table of Contents
- [Requirements & Setup](#-requirements--setup)
- [Quickstart: Inference](#-quickstart-inference)
- [Annotated Infant Respiration Dataset (AIR-400)](#-annotated-infant-respiration-dataset-air-400)
- [Reproducing Paper Results](#-reproducing-paper-results)
- [Citation](#-citation)
- [License](#-license)


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

<p align="center">
<img src="https://github.com/user-attachments/assets/ed03120a-5591-453a-aef3-b58b8348dd50" alt="Sample Inference Waveform" width="60%"></img>
</p>
<p align="center">
<img src="https://github.com/user-attachments/assets/b8c66ec4-a379-4d50-94ff-c08689228af7" alt="Sample Waveform Plot" width="50%" />
</p>
<p align="center">
<i>Sample Inference Output</i>
</p>



#### 1. Preparation
- Download a [trained model](https://coe.northeastern.edu/Research/AClab/AIR-400/Model%20Checkpoints/) and [ROI detector](https://coe.northeastern.edu/Research/AClab/AIR-400/ROI%20Detectors/) files. Download our [demo video](https://coe.northeastern.edu/Research/AClab/AIR-400/demo-air-400-s05-23.mp4), or provide your own as input.
- Fill the `DATA_PATH` fields of config YAML in `configs/inference` folder. 
  - Set path for **output** directory.
  - Set valid **detector** paths (YOLO weights) if ROI cropping is enabled. Otherwise, set `DO_CROP_INFANT_REGION: False`.
  - Set **input** video file or video folder path.

```yaml
DATA_PATH:
  OUTPUT_DIR: /absolute/path/to/output_dir/
  BODY_DETECTOR_PATH: /absolute/path/to/yolov8m.pt
  FACE_DETECTOR_PATH: /absolute/path/to/yolov8n-face.pt
  # Provide exactly one of the following:
  VIDEO_FILE: /absolute/path/to/video.mp4
  # VIDEO_DIR: /absolute/path/to/videos/
```

#### 2. Start inference process

Use `run_infer.sh` to preprocess input video(s) and run a trained model for respiration rate estimation. Specify required **config YAML** file path and **model checkpoint** file path in `run_infer.sh`.

Example run:

```bash
./run_infer.sh
```

#### 3. Expected outputs
- **Per-video JSON** under `OUTPUT_DIR/inference/{video}_{datetime}` with prediction result JSON file and generated artifacts (HDF5 format time series and PNG format waveform plots).
- A **summary JSON** across all processed videos (`summary_{datetime}.json`).
- Logs saved under `OUTPUT_DIR/logs/`.


## 📚 Annotated Infant Respiration Dataset (AIR-400)

The [**AIR-400** dataset](https://coe.northeastern.edu/Research/AClab/AIR-400/) consists of two parts:

- **AIR-125** — original dataset (125 videos from 8 subjects, labeled S01 through S08, with S06, S07, and S08 provided as public web links)

- **AIR-400** — expanded dataset (275 videos from 10 additional subjects from the same study, labeled S01 through S10; these subject IDs are independent from those in AIR-125)

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
Set `USE_WANDB: True` in YAML file.

#### 2. Download [AIR-400 dataset](https://coe.northeastern.edu/Research/AClab/AIR-400/) and [ROI detector](https://coe.northeastern.edu/Research/AClab/AIR-400/ROI%20Detectors/) files.

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
Specify required **config YAML** file path in `run.sh`. Then *uncomment* `--preprocess` after `python main.py --config "$CONFIG"` to enable **preprocess-only** mode. Run this approach first to make sure dataset is preprocessed correctly before following training and testing.
```bash
./run.sh
```

#### 5. Start training and testing process
Comment out `--preprocess` after `python main.py --config "$CONFIG"` in `run.sh` to start training and testing process.
```bash
./run.sh
```

## 📝 Citation

```bibtex

@inproceedings{song_bishnoi_overcoming_2026,
	booktitle = {2026 {IEEE}/{CVF} {Winter} {Conference} on {Applications} of {Computer} {Vision} ({WACV})},
	publisher = {IEEE},
	title = {Overcoming {Small} {Data} {Limitations} in {Video}-{Based} {Infant} {Respiration} {Estimation}},
	author = {Song, Liyang and Bishnoi, Hardik and Manne, Sai Kumar Reddy and Ostadabbas, Sarah and Taylor, Brianna J and Wan, Michael},
	year = {2026},
}
```

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
