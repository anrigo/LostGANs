Improving LostGANv2 occlusion handling capabilities through depth-awareness.
Fork of [LostGANs](https://github.com/WillSuen/LostGANs)

# Setup
Conda env
```bash
# New environment with pytorch 1.10.1
conda env create -f lost2-env.yml

# Old environment with pytorch 1.0 (not needed anymore)
conda env create -f lost-env.yml
```

Install g++ and cuda toolkit (nvcc), for Debian-based systems
```bash
apt install g++ build-essential nvidia-cuda-toolkit
```

Only needed if you have to run the project on the IvI cluster:
```bash
# change gcc to a version compatible with pytorch
source /opt/rh/devtoolset-8/enable

# add nvcc to path
PATH=$PATH:/usr/local/cuda-10.2/bin/

# check that you can access nvcc
nvcc --version
```

Setup roi_layers
```bash
python setup.py build develop
```
## Data Preparation

Download COCO dataset to `datasets/coco`
```bash
bash scripts/download_coco.sh
```
Download VG dataset to `datasets/vg`
```bash
bash scripts/download_vg.sh
python scripts/preprocess_vg.py
```


# LostGANs: Image Synthesis From Reconfigurable Layout and Style
This is implementation of our paper [**Image Synthesis From Reconfigurable Layout and Style**](https://arxiv.org/abs/1908.07500) and [**Learning Layout and Style Reconfigurable GANs for Controllable Image Synthesis**](https://arxiv.org/abs/2003.11571)


## Network Structure
![network_structure](./figures/network_structure.png)

## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions.
#### 1. Download pretrained model
Download pretrained [models](https://drive.google.com/drive/folders/1peI9d4PI7jJZJzFTcr-5mwZqnrNsX_3p?usp=sharing) to `pretrained_model/`

#### 2. Train models
```
python train.py --dataset coco --out_path outputs/
```

#### 3. Run pretrained model
```
python test.py --dataset coco --model_path pretrained_model/G_coco.pth --sample_path samples/coco/
```


## Results
###### Compare different models
![compare](./figures/generated_images.png)
###### Multiple samples generated from same layout
![various_out](./figures/various_outs.png)
###### Synthesized images and learned masks for given layout
![mask](./figures/mask.png)

## Contact
Please feel free to report issues and any related problems to Wei Sun (wsun12 at ncsu dot edu) and Tianfu Wu (tianfu_wu at ncsu dot edu).


## Reference
* Synchronized-BatchNorm-PyTorch: [https://github.com/vacancy/Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
* Image Generation from Scene Graphs: [https://github.com/google/sg2im](https://github.com/google/sg2im)
* Faster R-CNN and Mask R-CNN in PyTorch 1.0: [https://github.com/facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
