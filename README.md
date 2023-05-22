Code for our CVPR'23 paper: "[Polynomial Implicit Neural Representations For Large Diverse Datasets](https://arxiv.org/pdf/2303.11424.pdf)"
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polynomial-implicit-neural-representations/image-generation-on-imagenet-128x128)](https://paperswithcode.com/sota/image-generation-on-imagenet-128x128?p=polynomial-implicit-neural-representations)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polynomial-implicit-neural-representations/image-generation-on-imagenet-256x256)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=polynomial-implicit-neural-representations)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polynomial-implicit-neural-representations/image-generation-on-imagenet-512x512)](https://paperswithcode.com/sota/image-generation-on-imagenet-512x512?p=polynomial-implicit-neural-representations)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polynomial-implicit-neural-representations/image-generation-on-ffhq-256-x-256)](https://paperswithcode.com/sota/image-generation-on-ffhq-256-x-256?p=polynomial-implicit-neural-representations)


The libraries are burrowed from the [StyleGAN-XL](https://github.com/autonomousvision/stylegan_xl.git) repository. Big thanks to the authors for the wonderful code.




# Requirements 
- 64-bit Python 3.8 and PyTorch 1.9.0 (or later)
- CUDA toolkit 11.1 or later.
- GCC 7 or later compilers.
- Use the following commands with Miniconda3 to create and activate your Python environment:<br>
    &nbsp;conda env create -f environment.yml<br>
    &nbsp;conda activate polyinr<br>

# Data Preparation 
  python dataset_tool.py --source=./data/location --dest=./data/dataname_256.zip --resolution=256x256 --transform=center-crop



# Training the intial resolutuion

python train.py --outdir=./training-runs/dataname  --data=./data/dataname_32.zip --gpus=4 --batch=64 --mirror=1 --snap 10 --batch-gpu 8 --kimg 10000


# Training the super-resolution 

python train.py --outdir=./training-runs/dataname --data=./data/dataname_64.zip --gpus=4 --batch=64 --mirror=1 --snap 10 --batch-gpu 8 --kimg 10000 \
  --superres --path_stem training-runs/dataname/00000-gmgan-dataname_32-gpus8-batch64/best_model.pkl


# To generate samples run
first download the imagenet (256x256) checkpoint from the below link.
https://drive.google.com/file/d/1aYbsRpOHh0_ruBrRZz03GxQ7EvwVfudK/view?usp=share_link

python gen_images.py --outdir=out --trunc=0.6 --seeds=1-20 --batch-sz 1 --class 135 --network=path/to/best_model.pkl

