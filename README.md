
# Uni3DETR
Code release for our NeurIPS 2023 paper
**Uni3DETR: Unified 3D Detection Transformer**

Zhenyu Wang, Yali Li, Xi Chen, Hengshuang Zhao, Shengjin Wang

<div align="center">
  <img src="docs/Uni3DETR.png"/>
</div><br/>

This project provides an implementation for our NeurIPS 2023 paper "[Uni3DETR: Unified 3D Detection Transformer](https://arxiv.org/)" based on [mmDetection3D](https://github.com/open-mmlab/mmdetection3d). Uni3DETR provides a unified structure for both indoor and outdoor 3D object detection.

## Preparation
This project is based on [mmDetection3D](https://github.com/open-mmlab/mmdetection3d), which can be constructed as follows.
* Install mmDetection3D [v1.0.0rc5](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc5) following [the instructions](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc5/docs/getting_started.md).
* Copy our project and related files to installed mmDetection3D:
```bash
cp -r projects mmdetection3d/
cp -r extra_tools mmdetection3d/
```
* Prepare the dataset following -----------

## Training
```bash
bash extra_tools/dist_train.sh ${CFG_FILE} ${NUM_GPUS}
```

## Evaluation
```bash
bash extra_tools/dist_test.sh ${CFG_FILE} ${CKPT} ${NUM_GPUS} --eval=bbox
```

## Uni3DETR models
We provide results on SUN RGB-D, ScanNet, KITTI, nuScenes with pretrained models (for Tab. 1, Tab. 2, Tab. 3 of our paper).
|  Dataset                                    | mAP (%) | download | 
|---------------------------------------------|:-------:|:-------:|
| **indoor** |
| [SUN RGB-D](projects/configs/uni3detr/uni3detr_sunrgbd.py) | 67.0 | [GoogleDrive](https://drive.google.com/drive/folders/1ljh6quUw5gLyHbQiY68HDGtY6QLp_d6e?usp=sharing) |
| [ScanNet](projects/configs/uni3detr/uni3detr_scannet_large.py) | 71.7 | [GoogleDrive](https://drive.google.com/drive/folders/1ljh6quUw5gLyHbQiY68HDGtY6QLp_d6e?usp=sharing) |
| **outdoor** |
| [KITTI (3 classes)](projects/configs/uni3detr/uni3detr_kitti_car.py) | 86.57 (moderate car) | [GoogleDrive](https://drive.google.com/drive/folders/1ljh6quUw5gLyHbQiY68HDGtY6QLp_d6e?usp=sharing) |
| [KITTI (car)](projects/configs/uni3detr/uni3detr_kitti_3classes.py) | 86.74 (moderate car) | [GoogleDrive](https://drive.google.com/drive/folders/1ljh6quUw5gLyHbQiY68HDGtY6QLp_d6e?usp=sharing) |
| [nuScenes](projects/configs/uni3detr/uni3detr_nuscenes.py) | 61.7 | [GoogleDrive](https://drive.google.com/drive/folders/1ljh6quUw5gLyHbQiY68HDGtY6QLp_d6e?usp=sharing) |

