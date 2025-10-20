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

[INCLUDE DEMO VIDEO]

## Requirements & Setup

#### 1. Set up the environment
```bash
conda env create -f environment.yml
```

#### 2. Compile [pyflow](https://github.com/pathak22/pyflow) library and import it as a module
```bash
git clone https://github.com/pathak22/pyflow.git
cd pyflow/
python setup.py build_ext -i
```
Move the compiled `pyflow.cpython-**.so` file to the root directory of this repo, so `pyflow` can be imported directly as a module.

#### 3. Sign W&B and login to record training results
```bash
export WANDB_API_KEY=<your_api_key>
wandb login
```

## Quickstart: Inference

(Simple instructions for someone to run a pretrained model on their own video data.)

(Link to pretrained models on Google Drive.)

## Annotated Infant Respiration Dataset (AIR-400)

(Link to dataset, with basic description of how it is organized.) 

## Reproducing Paper Results

#### 1. Get [AIR-400 dataset]() folder and [ROI detectors]() ready

#### 2. Update data paths in the config yaml files

#### 3. Preprocess the data
Specify required config yaml file path in `run.sh`. Then uncomment `--preprocess` after `python main.py --config "$CONFIG"` to enable preprocess-only mode. Run this approach first to make sure dataset is preprocessed correctly before following training and testing.
```bash
./run.sh
```

#### 4. Start training and testing process
Comment out `--peprocess` after `python main.py --config "$CONFIG"` in `run.sh` to start training and testing process.
```bash
./run.sh
```







