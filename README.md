# Co-SemDepth
Official implementation for the joint network presented in the paper "Co-SemDepth: Fast Joint Semantic Segmentation and Depth Estimation on Aerial Images" 
![alt text](https://github.com/Malga-Vision/Co-SemDepth/blob/main/joint_arch.png?raw=true)
## Overview
Co-SemDepth is a lightweight joint deep architecture for monocular depth estimation and semantic segmentation given an input of image sequences captured in outdoor environments by a camera moving with 6 degrees of freedom (6 DoF). 

Please refer to these repositories for the single [M4Depth](https://github.com/michael-fonder/M4Depth) and [M4Semantic](https://github.com/Malga-Vision/M4Semantic) architectures.
## Citation

## Dependencies
Starting from a fresh Anaconda environment, you can install the required depndencies to run our code with:
```shell
conda install -c conda-forge tensorflow-gpu=2.7 numpy pandas
```

### Datasets

#### Mid-Air [[1](#ref_1)]

To download the Mid-Air dataset necessary for training and testing our architecture, do the following:
> 1. Go on the [download page of the Mid-Air dataset](https://midair.ulg.ac.be/download.html)
> 2. Select the "Left RGB", "Semantic seg." and "Stereo Disparity" image types
> 3. Move to the end of the page and press "Get download links"

When you have the file, execute this script to download and extract the dataset:
```shell
bash  scripts/0a-get_midair.sh path/to/desired/dataset/location path/to/download_config.txt
```

Apply the semantic classes mapping on MidAir by running the following script:
```shell
python scripts/data_class_mapping.py
```
## Reproducing paper results

### Training from scratch
To train on MidAir:
```shell
bash  scripts/1a-train-midair.sh path/to/desired/weights/location
```

### Evaluation and Pretrained weights
The pre-trained weights can be downloaded from [here](https://drive.google.com/file/d/1N5Av4b_eVzGohn0OmgRhWlpN1YuIVcb9/view?usp=sharing) and extracted inside the weights folder.

To evaluate on MidAir:
```shell
bash  scripts/2-evaluate.sh midair path/to/weights/location
```
### Other operations

### Processing outputs

## Prediction on your own images

## Baseline methods performance reproduction

## Training on MidAir+TopAir
TopAir dataset can be downloaded from [here](https://huggingface.co/datasets/yaraalaa0/TopAir). Then you need to specify its path in the file [datasets_location.json](https://github.com/Malga-Vision/Co-SemDepth/blob/main/datasets_location.json)

To train on MidAir and TopAir datasets together, run:
```shell
python joint_train.py --mode=train --seq_len=4 --db_seq_len=4 --arch_depth=5 --ckpt_dir="path/to/weights/location/" --log_dir="path/to/weights/location/summaries" --records_path=data/ --val_records_path=data/
```

## License

