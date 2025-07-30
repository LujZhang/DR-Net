

This is the official repository of the **Dilated Regions Network (DR-Net)**. For technical details, please refer to:

 <a href=https://ieeexplore.ieee.org/abstract/document/10488461/> **Weakly-supervised Point Cloud Semantic Segmentation Based on Dilated Region** <br /></a>

### (1) Setup

This code has been tested with Python 3.5, Tensorflow 1.11, CUDA 9.0 and cuDNN 7.4.1 on Ubuntu 16.04/Ubuntu 18.04.

- Clone the repository

```
git clone --depth=1 https://github.com/LujZhang/DR-Net && cd DR-Net
```

- Setup python environment

```
conda create -n drnet python=3.5
source activate drnet
pip install -r helper_requirements.txt
sh compile_op.sh
```

### (2) Training (S3DIS as example)

First, follow the RandLA-Net [instruction](https://github.com/QingyongHu/RandLA-Net) to prepare the dataset.

- Start training with weakly supervised setting:
```
python main_S3DIS.py --mode train --gpu 0 --labeled_point 0.1%
```
- Evaluation:
```
python main_S3DIS.py --mode test --gpu 0 --labeled_point 0.1%
```
### Acknowledge

- Our code refers to <a href="https://github.com/QingyongHu/RandLA-Net">RandLA-Net</a> and <a href="https://github.com/QingyongHu/SQN">SQN</a>, and thank a lot to these contribution.




