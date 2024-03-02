

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


## Related Repos

1. [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://github.com/QingyongHu/RandLA-Net) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/RandLA-Net.svg?style=flat&label=Star)
2. [SoTA-Point-Cloud: Deep Learning for 3D Point Clouds: A Survey](https://github.com/QingyongHu/SoTA-Point-Cloud) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SoTA-Point-Cloud.svg?style=flat&label=Star)
3. [3D-BoNet: Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds](https://github.com/Yang7879/3D-BoNet) ![GitHub stars](https://img.shields.io/github/stars/Yang7879/3D-BoNet.svg?style=flat&label=Star)
4. [SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration](https://github.com/QingyongHu/SpinNet) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SpinNet.svg?style=flat&label=Star)
5. [SensatUrban: Learning Semantics from Urban-Scale Photogrammetric Point Clouds](https://github.com/QingyongHu/SensatUrban) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SensatUrban.svg?style=flat&label=Star)
6. [Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds](https://github.com/yifanzhang713/IA-SSD) ![GitHub stars](https://img.shields.io/github/stars/yifanzhang713/IA-SSD.svg?style=flat&label=Star)




