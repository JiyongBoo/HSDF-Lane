<br />
<p align="center">
  
  <h3 align="center"><strong>HSDF-Lane: Height-Aligned Signed Distance Field with Semantic Lane Prior for 3D Lane Detection</strong></h3>

<p align="center">
  <a href="" target='_blank'>
    <!-- <img src="https://img.shields.io/badge/arXiv-%F0%9F%93%83-yellow"> -->
    <img src="https://img.shields.io/badge/arXiv-Paper-red">
  </a>
  <a href="https://jiyongboo.github.io/HSDF-Lane-project-page/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-Page-blue">
  </a>
</p>


This is the official implementation of **HSDF-Lane** (ECCV 2026).

---

### Dataset Preparation
Please follow [data preparation](./docs/data_preparation.md) to download dataset.

---

### Installation

#### 1. Clone this repository:

```bash
git clone https://github.com/JiyongBoo/HSDF-Lane
```

#### 2. Create a virtual environment:

```bash
cd ./HSDF-Lane
conda create -n hsdflane python=3.8.20
pip install -r requirement.txt
conda install -c conda-forge cudatoolkit-dev=11.1
```

#### 3. Clone the required dependency (Deformable-DETR) **in the same parent directory**:

```bash
cd ..
git clone https://github.com/fundamentalvision/Deformable-DETR.git
```

Place both repositories under the same parent directory as follows:

```
<your_workspace>/
├── dataset/
├── Deformable-DETR/
└── HSDF-Lane/
```

#### 4. Compile CUDA operators:

Build the CUDA operators from the Deformable-DETR ops directory:

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

cd ./Deformable-DETR/models/ops
sh ./make.sh
```

#### 5. Additional Requirements:
```bash
pip install setuptools==69.5.1 wheel ninja Cython
```

---

### Eval

#### Pretrained Checkpoints
Download the pretrained model from the following link:
[https://huggingface.co/boo0828/HSDF-Lane](https://huggingface.co/boo0828/HSDF-Lane)

```bash
cd <your_workspace>/HSDF-Lane/
huggingface-cli download boo0828/HSDF-Lane <ckpt_name> --local-dir ./
```

| Dataset | ckpt_name | Metrics | 
| - | - | - | 
| OpenLane-1000 | [hsdflane.pth](https://huggingface.co/boo0828/HSDF-Lane/resolve/main/hsdflane.pth?download=true) | F1=66.3% | 
| OpenLane-1000 (FPN version) | [hsdflane_fpn.pth](https://huggingface.co/boo0828/HSDF-Lane/resolve/main/hsdflane_fpn.pth?download=true) | F1=66.9% | 
| Apollo-standard,rare | [hsdflane_apollo_standard.pth](https://huggingface.co/boo0828/HSDF-Lane/resolve/main/hsdflane_apollo_standard.pth?download=true) | F1=98.8% | 
| Apollo-illus_chg| [hsdflane_apollo_illus_chg.pth](https://huggingface.co/boo0828/HSDF-Lane/resolve/main/hsdflane_apollo_illus_chg.pth?download=true) | F1=97.9% | 


#### Validation

With the dataset and checkpoint in place, run:

```bash
cd <your_workspace>/HSDF-Lane/
python tools/val_openlane.py \\
    --config ./tools/hsdflane_config.py \\
    --checkpoint ./hsdflane.pth 
```

This runs evaluation on the OpenLane validation set.

To evaluate the model on the Apollo dataset, run:

```bash
cd <your_workspace>/HSDF-Lane/
python tools/val_apollo.py \\
    --config ./tools/hsdflane_apollo_config.py \\
    --checkpoint ./hsdflane_apollo_standard.pth 
```

### Train

To train HSDF-Lane from scratch:

```bash
cd <your_workspace>/HSDF-Lane/
# OpenLane
python tools/train_openlane.py \\
    --config ./tools/hsdflane_config.py
# Apollo
python tools/train_apollo.py \\
    --config ./tools/hsdflane_apollo_config.py
```

---

### Citation

```
TBD
```

---

### Acknowledgments

This repository builds upon

* [**HeightLane**](https://github.com/parkchaesong/HeightLane)
* [**SC-Lane**](https://github.com/parkchaesong/SC-Lane)

We are grateful to the authors of these works for providing the foundation for our project.
