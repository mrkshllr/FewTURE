# Installation and Datasets  (used in FewTURE)

## Prerequisites
Please install [PyTorch](https://pytorch.org/)  as appropriate for your system. This codebase has been developed with 
python 3.8.12, PyTorch 1.11.0, CUDA 11.3 and torchvision 0.12.0 with the use of an 
[anaconda](https://docs.conda.io/en/latest/miniconda.html) environment.

To create an appropriate conda environment (after you have successfully installed conda), run the following command:
```
conda create --name fewture --file requirements.txt
```
Activate your environment via
```
conda activate fewture
```
----------


## Datasets
### <i>mini</i> ImageNet
To download the miniImageNet dataset, you can use the script [download_miniimagenet.sh](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_miniimagenet.sh) in the `datasets` folder.

The miniImageNet dataset (Vinyals et al., 2016; Ravi & Larochelle, 2017) consists of a specific 100 class subset of Imagenet (Russakovsky et al., 2015) with 600 images
for each class. The data is split into 64 training, 16 validation and 20 test classes.

### <i>tiered</i> ImageNet
To download the tieredImageNet dataset, you can use the script [download_tieredimagenet.sh](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_tieredimagenet.sh) in the `datasets` folder.

Similar to the previous dataset, the tieredImageNet (Ren et al., 2018) is a subset of classes
selected form the bigger ImageNet dataset (Russakovsky et al., 2015), however with a substantially larger set of classes and
different structure in mind. It comprises a selection of 34 super-classes with a total of 608 categories,
totalling in 779,165 images that are split into 20,6 and 8 super-classes to achieve better separation
between training, validation and testing, respectively.

### CIFAR-FS
To download the CIFAR-FS dataset (Bertinetto et al., 2018), you can use the script [download_cifar_fs.sh](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_cifar_fs.sh) in the `datasets` folder.

The CIFAR-FS dataset contains the 100 categories with 600 images per category
from the CIFAR100 dataset (Krizhevsky et al., 2009) which are split into 64 training, 16 validation and 20 test classes.

### FC100
To download the FC-100 dataset, you can use the script [download_fc100.sh](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_fc100.sh) in the `datasets` folder.

The FC-100 dataset (Oreshkin et al., 2018) is also derived from CIFAR100 (Krizhevsky et al., 2009) but follows a splitting strategy
similar to tieredImageNet to increase difficulty through higher separation, resulting in 60 training, 20
validation and 20 test classes.

