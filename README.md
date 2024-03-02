

This is the official repository of the **Dilated Regions Network (DR-Net)**. For technical details, please refer to:

**Weakly-supervised Point Cloud Semantic Segmentation Based on Dilated Region** <br />

### (1) Setup

This code has been tested with Python 3.5, Tensorflow 1.11, CUDA 9.0 and cuDNN 7.4.1 on Ubuntu 16.04/Ubuntu 18.04.

- Clone the repository

```
git clone --depth=1 https://github.com/QingyongHu/SQN && cd SQN
```

- Setup python environment

```
conda create -n sqn python=3.5
source activate sqn
pip install -r helper_requirements.txt
sh compile_op.sh
```

### (2) Training (Semantic3D as example)

First, follow the RandLA-Net [instruction](https://github.com/QingyongHu/RandLA-Net) to prepare the dataset, and then
manually change the
dataset [path](https://github.com/QingyongHu/SQN/blob/f75eb51532a5319c0da5320c20f58fbe5cb3bbcd/main_Semantic3D.py#L17) here.

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




