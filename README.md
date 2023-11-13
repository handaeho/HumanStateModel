# WTC_deepview

## Prerequisite programs
```
# conda create –name <env_name> python=3.7
conda create –name pytorch1.6 python=3.7
-> if python3.8 needed,
conda create –name pytorch1.6 python=3.8

# activate the created environment
source activate pytorch1.6

# install pytorch & torchvision
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
-> if CUDA 11.3 needed,
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# install opencv-python
pip install opencv-python

# install detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
