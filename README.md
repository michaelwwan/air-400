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
  <a href="https://drive.google.com/drive/folders/1-bYcnAFy15y_sff9-izpPSGS-cinzEut?usp=share_link">
    <img src="https://img.shields.io/badge/Dataset-Google%20Drive-blue.svg?logo=google-drive&style=flat-square">
  </a>
  <a href="https://drive.google.com/drive/folders/1ohYbeIJG85cpop3yhBtXaCfQ3ooWZMsk?usp=sharing">
    <img src="https://img.shields.io/badge/Model-Checkpoint-orange.svg?logo=google-cloud&style=flat-square">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square">
  </a>
</p>

This is the official repository of our **WACV 2026** paper:

> Song, L.\*, Bishnoi, H.\*, Manne, S.K.R., Ostadabbas, S., Taylor, B.J., Wan, M., "**Overcoming Small Data Limitations in Video-Based Infant Respiration Estimation**" (*equal contribution). 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV).

Here we provide our **model code**, **training checkpoints**, and **annotated dataset** to support automatic estimation of **infant respiration waveforms and respiration rate** from natural video footage, with the help of spatiotemporal computer vision models and infant-specific region-of-interest tracking. 

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
- Download a [trained model](https://drive.google.com/drive/folders/1ohYbeIJG85cpop3yhBtXaCfQ3ooWZMsk?usp=sharing) and [ROI detector](https://drive.google.com/drive/folders/1PQo7md-hW1x76l_GaBnWH8_H8U7rxpOt?usp=share_link) files. Download our [demo video](https://drive.google.com/file/d/1GLIE4sI8xc06mi0-9h6F6yGMxd39caOc/view?usp=share_link), or provide your own as input.
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

#### 3. Outputs
- **Per-video JSON** under `OUTPUT_DIR/inference/{video}_{datetime}` with prediction result JSON file and generated artifacts (HDF5 format time series and PNG format waveform plots).
- A **summary JSON** across all processed videos (`summary_{datetime}.json`).
- Logs saved under `OUTPUT_DIR/logs/`.


## 📚 Annotated Infant Respiration Dataset (AIR-400)

The [**AIR-400** dataset](https://drive.google.com/drive/folders/1-bYcnAFy15y_sff9-izpPSGS-cinzEut?usp=share_link) consists of two parts:

- **AIR-125** — original dataset (125 videos from 8 subjects, labeled S01 through S08, with S06, S07, and 08 provided as public web links)

- **AIR-400** — expanded dataset (275 videos from 10 additional subjects from the same study, labeled S01 through S10, but not the same as the ones from AIR-125)

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

#### 2. Download [AIR-400 dataset](https://drive.google.com/drive/folders/1-bYcnAFy15y_sff9-izpPSGS-cinzEut?usp=share_link) and [ROI detector](https://drive.google.com/drive/folders/1PQo7md-hW1x76l_GaBnWH8_H8U7rxpOt?usp=sharing) files.

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
Comment out `--peprocess` after `python main.py --config "$CONFIG"` in `run.sh` to start training and testing process.
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
