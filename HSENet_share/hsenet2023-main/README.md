# Installation

## 1. Create new virtual environment. Tested with Python 3.10  
python -m venv /path/to/new/virtual/environment

## 2. Install pytorch

pip3 install torch torchvision torchaudio

## 3.  Install detectron2. Current version is 0.6
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

If you failed to install detectron2 in the conventional way, try this approach below:
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
CC=g++ python setup.py install


# How to run evaluation
## Register your dataset
### 1. Go into train_net file and register your evaluation dataset.
### 2. Modify the test dataset in the config file

### Run evaluation command

python hsenet_gaion_train_net.py --eval-only --config ./configs/ihp_hsenet_V_39_FPN_3x.yaml MODEL.WEIGHTS ./weights/model_final.pth 

