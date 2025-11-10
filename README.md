# Video-Based Infant Respiration Estimation & AIR-400 Dataset

This is the official repository of our paper:

> Song, L.\*, Bishnoi, H.\*, Manne, S.K.R., Ostadabbas, S., Taylor, B.J., Wan, M., “Overcoming Small Data Limitations in Video-Based Infant Respiration Estimation" (*equal contribution). Under review, [arXiv preprint].

````
@misc{song_bishnoi_overcoming_2025,
	title = {Overcoming {Small} {Data} {Limitations} in {Video}-{Based} {Infant} {Respiration} {Estimation}},
	url = {https://arxiv.org/abs/0000.00000},
	author = {Song, Liyang and Bishnoi, Hardik and Manne, Sai Kumar Reddy and Ostadabbas, Sarah and Taylor, Brianna J and Wan, Michael},
	year = {2025},
}
````

Here we provide our model code, training checkpoints, and annotated dataset to support automatic estimation of infant respiration waveforms and respiration rate from natural video footage, with the help of spatiotemporal computer vision models and infant-specific region-of-interest tracking. 

<video src="https://github.com/user-attachments/assets/4b7b4703-9163-4812-b380-84225971f0e7" controls autoplay loop muted></video>

## Requirements & Setup

#### 1. Set up the environment
```bash
conda env create -f environment.yml
```

#### 2. Compile [pyflow](https://github.com/pathak22/pyflow) library and import it as a module
```bash
git clone https://github.com/pathak22/pyflow.git
(cd pyflow && python setup.py build_ext -i && mv pyflow.cpython-*.so ..)
```

## Quickstart: Inference

Use `infer.py` to preprocess input video(s) and run a trained model for respiration rate estimation.

1. Preparation
- Download [trained model](https://drive.google.com/drive/folders/1kjSAF9Dt24D670cwBgc-uXCz8WYTaulq?usp=drive_link) and [ROI detector](https://drive.google.com/drive/folders/1k0BHMGXAXIdmOYyt3iGzbUBH_sEGVcAk?usp=drive_link) files.
- Fill the YAML `DATA_PATH` fields. 
  - Set paths for cache directory and output directory.
  - Set valid detector paths (YOLO weights) if ROI cropping is enabled. Otherwise, set `DO_CROP_INFANT_REGION: False`.
  - Set input video path.

```yaml
DATA_PATH:
  CACHE_DIR: /absolute/path/to/cache_dir/
  OUTPUT_DIR: /absolute/path/to/output_dir/
  BODY_DETECTOR_PATH: /absolute/path/to/yolov8m.pt
  FACE_DETECTOR_PATH: /absolute/path/to/yolov8n-face.pt
  # Provide exactly one of the following:
  VIDEO_FILE: /absolute/path/to/video.mp4
  VIDEO_DIR: /absolute/path/to/videos/
```

2. Example run:

```bash
python infer.py \
  --config configs/inference/virenet_infer_example.yaml \
  --checkpoint /path/to/model_dir/VIRENet_best.pth \
```

3. Outputs:
- Per-video JSON under `OUTPUT_DIR/inference/` with RR stats per chunk and mean/std.
- Logs saved under `OUTPUT_DIR/logs/
- A summary JSON across all processed videos.
- (Optional) Waveform CSV and PNG can be enabled in future; current version focuses on RR stats.

## Annotated Infant Respiration Dataset (AIR-400)

(Link to dataset, with basic description of how it is organized.) 

## Reproducing Paper Results

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
