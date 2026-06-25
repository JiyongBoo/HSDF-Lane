# Data Preparation

## OpenLane

#### Step 1: Download OpenLane Dataset
Follow the instructions from the [OpenLane Dataset README](https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md) to download the full dataset.

```bash
cd <your_workspace>/dataset && mkdir openlane && cd openlane
ln -s ${OPENLANE_PATH}/images .
ln -s ${OPENLANE_PATH}/lane3d_1000/training .
ln -s ${OPENLANE_PATH}/lane3d_1000/validation .
```
After downloading, your directory structure should look like:
```
<your_workspace>/dataset/openlane/
├── images/
├── training/
└── validation/
```

#### Step 2: Download Height Map Data

Download the height map data from the following link:
[https://huggingface.co/datasets/boo0828/HSDF-Lane_heightmap](https://huggingface.co/datasets/boo0828/HSDF-Lane_heightmap)

```bash
cd <your_workspace>/dataset/openlane
huggingface-cli download boo0828/HSDF-Lane_heightmap openlane_heightmap.tar --local-dir .
tar -xvf openlane_heightmap.tar
```
After downloading and extracting the data, your final directory structure should look like this:

```
<your_workspace>/dataset/openlane/
├── images/
├── training/
├── validation/
├── heightmap_training/
└── heightmap_validation/
```

## Apollo

#### Step 1: Download Apollo Dataset
Follow the instructions from the [Apollo Dataset README](https://github.com/yuliangguo/3D_Lane_Synthetic_Dataset) to download the full dataset.

```bash
cd <your_workspace>/dataset 
# Directly download into the '<your_workspace>/dataset' folder and unzip
unzip Apollo_Sim_3D_Lane_Release.zip
```
After downloading, your directory structure should look like:
```
<your_workspace>/dataset/Apollo_Sim_3D_Lane_Release/
├── depth/
├── images/
├── labels/
├── segmentation/
├── img_list.txt
└── laneline_label.json

```

#### Step 2: Download Height Map Data

Download the height map data from the following link:
[https://huggingface.co/datasets/boo0828/HSDF-Lane_heightmap](https://huggingface.co/datasets/boo0828/HSDF-Lane_heightmap)

```bash
cd <your_workspace>/dataset/openlane
huggingface-cli download boo0828/HSDF-Lane_heightmap apollo_heightmap.tar --local-dir .
tar -xvf apollo_heightmap.tar
```
After downloading and extracting the data, your directory structure should be updated as follows:

```
<your_workspace>/dataset/Apollo_Sim_3D_Lane_Release/
├── depth/
├── images/
├── labels/
├── segmentation/
├── map/                # Extracted height map data
├── img_list.txt
└── laneline_label.json
```

#### Step 3: Make Dataset Split

The Apollo dataset is categorized into three distinct splits: 'standard', 'rare_subset', and 'illus_chg'. Run the following script to generate these splits automatically:

```bash
cd <your_workspace>/HSDF-Lane/loader/bev_road
python create_apollo_splits.py --seed 42
```
Once the script completes, your final directory structure will look like this:
```
<your_workspace>/dataset/Apollo_Sim_3D_Lane_Release/
├── depth/
├── images/
├── labels/
├── segmentation/
├── map/                # Extracted height map data
├── splits/
|     ├── illus_chg/
|     ├── rare_subset/
|     └── standard/   
├── train.json
├── val.json
├── img_list.txt
└── laneline_label.json
```