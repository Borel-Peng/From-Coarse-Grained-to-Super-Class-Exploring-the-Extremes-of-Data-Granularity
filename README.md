# Superclass Learning with Representation Enhancement

This is the official repo of From Coarse-Grained to Super-Class: Exploring the
Extremes of Data Granularity.

<!-- The paper: -->

We provide a training example of our model: SCLRE (SuperClass Learning with Representation Enhancement). You can generate the reorganized CIFAR100-4 dataset and train it. After training, you can check the result with the help of tensorboard. Some main package versions used in the program:
## 0. Dependencies
```
python: 3.8.12
pytorch: 1.11.0
torchvision: 0.12.0
CUDA Version: 11.4
tensorboard-logger: 0.1.0
tensorboard: 2.7.0
tqdm: 4.62.3
CLIP: 1.0
```
## 1. Generate Superclass Dataset
In filefolder `ptgenerate/`, we provide a mapping list `CIFAR100-4.csv` to map common classes to superclasses according to the rules in the supplementary metarial. You can run the following command:
```
python cifar100.py
```
to get two reorganized data tensors `train.pt` and `test.pt` generated. They are placed at `./data/re_cifar100/4_categories/`.
## 2. Train with SCLRE
After generation, in the main directory, run
```
python main.py
```
to train with SCLRE. The settings of GPUs are in `main.py`, and you can change it based on your hardware.
The training process will be shown in the terminal with the help of `tqdm`.
## 3. Check the Result
After training, the result will be recorded with the help of `tensorboard`. By default, it will be restored in `./runs/`, and you can run
```
tensorboard --logdir=runs/
```
to check the training result. We evaluate our model in accuracy, recall, precision and F1 score. We also record the loss in the training process.

## 4. Change the Hyperparameters
All the hyperparameters are listed in `hyperparameter.py`, and the meanings of them are also listed in the comments in this file. You can change the hyperparameters and re-train with Step2. 

To change the architecture of the model, set `attention=False` to drop CIA, set `contrastive=False` to drop contrastive adjustment loss, set `TARGET=False` to drop targeted adjustment loss.

Note that when the backbone is set as a CNN smaller than ResNet50, some network params in `model/TC2.py` need to be reset because these backbones generate representations with smaller dimensions. So we recommend to set this hyperparameter only as `ResNet50`, `ResNet101` and `ResNet152`.

## 5. Train with CLIP
Before training with CLIP, you need to change the hpyerparameters in `hyperparameter.py` to set the backbone of CLIP(`ViT-B/32` or `RN50`), the batch size and the name of the dataset. Then run
```
python main_clip.py
```
## 6. Visualization
We adapt the t-SNE algorithm to visualize the results of `CIFAR10-2` in the file `visualization.ipynb`. You can run the code in the jupyter notebook to visualize the results. 

## 6. Acknowledgements
This paper is an extension of the conference paper in [CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Superclass_Learning_With_Representation_Enhancement_CVPR_2023_paper.pdf).

So the code is based on its official [code](https://github.com/ZyGan1999/Superclass-Learning-with-Representation-Enhancement).