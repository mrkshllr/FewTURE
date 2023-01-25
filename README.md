# FewTURE - Rethinking Generalization in Few-Shot Classification
Official PyTorch implementation of the paper **Rethinking Generalization in Few-Shot Classification**.

:mortar_board: :page_facing_up: Find our paper: [[arXiv]](https://arxiv.org/abs/2206.07267) &nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;&nbsp;&nbsp;&nbsp; 
:bookmark: Reference: [[BibTeX]](https://github.com/mrkshllr/FewTURE#citing-fewture)

## Updates :tada:
- September 15, 2022: **FewTURE is accepted at NeurIPS 2022!** :fire: 
- August 30, 2022: Release of our code -- **Try out FewTURE!**  :sparkles: :computer: :arrow_left:
- June 15, 2022: Release of our preprint on [arXiv](https://arxiv.org/abs/2206.07267)

## TL;DR :eyes:
<div align="center">
  <img width="95%" alt="FewTure concept" src=".github/concept.png">
</div>

**FewTURE** is a novel approach to tackle the challenge of supervision collapse introduced by single image-level labels 
in few-shot learning. Splitting the input samples into patches and
encoding these via the help of Vision Transformers allows us to establish semantic
correspondences between local regions across images and independent of their
respective class. The most informative patch embeddings for the task at hand are
then determined as a function of the support set via online optimization at inference
time, additionally providing visual interpretability of ‘what matters most’ in the
image. We build on recent advances in unsupervised training of networks via
masked image modelling to overcome the lack of fine-grained labels and learn the
more general statistical structure of the data while avoiding negative image-level
annotation influence, aka supervision collapse.

**FewTURE** achieves strong performance on several few-shot classification benchmarks (Accuracy on unseen test set):


|  Dataset  |         5-Way 1-Shot          |   5-Way 5-Shot   |   
|:--------|:-----------------------------:|:----------------:|
| <i>mini</i>ImageNet | **72.40** ± <font size=1>0.78 | **86.38** ± <font size=1>0.49 |
|  <i>tiered</i>ImageNet  |       **76.32** ± <font size=1>0.87        | **89.96** ± <font size=1>0.55 | 
| CIFAR-FS |       **77.76** ± <font size=1>0.81        | **88.90** ± <font size=1>0.59 | 
| FC100 |       **47.68** ± <font size=1>0.78        | **63.81** ± <font size=1>0.75 |

<br>

**FewTURE** learns '<i>what matters most</i>' at inference time via online optimization:
<div align="center">
    <img width="95%" alt="Importance weights 5shot" src=".github/token_weights_5shot.png">
</div>


## Installation and Datasets
For detailed instruction how to set up your environment, install required packages and download the datasets, please refer to the [installation instructions](https://github.com/mrkshllr/FewTURE/blob/main/INSTALL.md).

## Training FewTURE
For a glimpse at the documentation of all arguments available for training, please check for **self-supervised training**:
```
python train_selfsup_pretrain.py --help
```
and for **meta fine-tuning**:
```
python train_metatrain_FewTURE.py --help
```


### Self-Supervised Pretraining via Masked Image Modelling
To start the self-supervised pre-training procedure using a **ViT-small** architecture on one node with 4 GPUs using a total **batch size** of **512** for **1600 epochs** on **miniImageNet**, run:
```
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345
export WORLD_SIZE=4
echo "Using Master_Addr $MASTER_ADDR:$MASTER_PORT to synchronise, and a world size of $WORLD_SIZE."

echo "Start training..."
torchrun --nproc_per_node 4 train_selfsup_pretrain.py --use_fp16 True --arch vit_small --epochs 1600 --batch_size_per_gpu 128 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --dataset miniimagenet --image_size 224 --data_path <path-to-dataset> --saveckp_freq 50 --shared_head true --out_dim 8192  --local_crops_number 10 --global_crops_scale 0.25 1 --local_crops_scale 0.05 0.25 --pred_ratio 0 0.3 --pred_ratio_var 0 0.2
echo "Finished training!"
```
If you want to instead train a swin Transformer architecture, choose `--arch swin_tiny`. 
Since swin models generally lead to higher memory consumption, consider using multiple nodes 
for training (we use 2 in our paper with a batch size of 64 per GPU), or experiment with reducing the overall batch size. We also recommend reducing the number of epochs (e.g. 300 - 800).


_Note_: Make sure to provide the path to the folder where you stored the datasets via the `--data_path <path-to-dataset>` argument.

###Self-Supervised Pretrained Models
You can use the following links to download the checkpoints of our pretrained models. <br> Note that all provided models have been trained in a self-supervised manner using **ONLY** the training split of the denoted few-shot image classification datasets and **NO** additional data.

<table>
  <tr>
    <th>Pretraining Dataset</th>
    <th> Architecture</th>
    <th> Epochs</th>
    <th colspan="3">Download</th>
  </tr>
  <tr>
    <td rowspan="2"><i>mini</i>ImageNet</td>
    <td align="center">vit-small</td>
    <td align="center">1600</td>
    <td><a href="https://drive.google.com/file/d/16ed4kmJ4cAZaXKzpRV9N5EQrTFArEAqs/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1eiQPhaY7w0YmjFOYyrAIRRUKfRNLDoyV/view?usp=share_link">args</a></td>
    <td><a href="https://drive.google.com/file/d/1FdjHFXf6CnveW8ki_iP0MwewGmQmdtnK/view?usp=share_link">logs</a></td>
  </tr>
  <tr>
    <td align="center">swin-tiny</td>
    <td align="center">&nbsp;800</td>
    <td><a href="https://drive.google.com/file/d/1iWeDYLKfVSd06OMmaTYfhKAoV4lNuizB/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1FHaf8Omus-tD1CeDATpDnmqh6C8nphKO/view?usp=share_link">args</a></td>
    <td><a href="https://drive.google.com/file/d/1OCimTYAY90KxK-XznX2IRrJTMv7CRPQm/view?usp=share_link">logs</a></td>
  </tr>
  <tr>
    <td rowspan="2"><i>tiered</i>ImageNet</td>
    <td align="center">vit-small</td>
    <td align="center">&nbsp;&nbsp;800*</td>
    <td><a href="https://drive.google.com/file/d/1Fr17-oE6-WOJ3O1VYgZl6S417wx7PfFP/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1IZ8CqI2BJXQJXiBI45TzgRJ8VP-VmfRZ/view?usp=share_link">args</a></td>
    <td><a href="https://drive.google.com/file/d/1N7P99sXZQab1lP_BHHDReKy5KKw0FqeC/view?usp=share_link">logs</a></td>
  </tr>
  <tr>
    <td align="center">swin-tiny</td>
    <td align="center">&nbsp;800</td>
    <td><a href="https://drive.google.com/file/d/16vDA5dVigqXuzAf7mSGka6xWZDcJuQDH/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1fVUJvKMUAVkiNHUPURzznB8lb1aGwt0L/view?usp=share_link">args</a></td>
    <td><a href="https://drive.google.com/file/d/1FFYOKHnKhaVrzTPYhwScxf8rlpOcDiuH/view?usp=share_link">logs</a></td>
  </tr>
  <tr>
    <td rowspan="2">CIFAR-FS</td>
    <td align="center">vit-small</td>
    <td align="center">1600</td>
    <td><a href="https://drive.google.com/file/d/1Qqk3nECzdXVHbkq2pGucKZd_Jh_4TFBT/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1eyUrtLefNr8aAoQ7kB5TJYR5PLsTFONZ/view?usp=share_link">args</a></td>
    <td><a href="https://drive.google.com/file/d/1Y9Kp7AEy2nIfq-VT2h7KEQXhF7XEHvDo/view?usp=share_link">logs</a></td>
  </tr>
  <tr>
    <td align="center">swin-tiny</td>
    <td align="center">&nbsp;800</td>
    <td><a href="https://drive.google.com/file/d/1MQpk_CQLS0jRIZp47w4IyYC2lWTqftcr/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1fD3A5fmbYiAyiRoAtrYe_Rt-GF0qB0Dl/view?usp=share_link">args</a></td>
    <td><a href="https://drive.google.com/file/d/1l-xAOtv57ZmK1z1yyQ65yIWIRlkxj58n/view?usp=share_link">logs</a></td>
  </tr>
  <tr>
    <td rowspan="2">FC100</td>
    <td align="center">vit-small</td>
    <td align="center">1600</td>
    <td><a href="https://drive.google.com/file/d/1CCr0QVT79qYGVxWgMLJOOQ5xOfMZfOPH/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1kOKxHJh6AsjCnVWhKkDnCyH-l4MI6thn/view?usp=share_link">args</a></td>
    <td><a href="https://drive.google.com/file/d/1QNkcSEswdtLqWp1R-RsBKPLn-bdPgsK6/view?usp=share_link">logs</a></td>
  </tr>
  <tr>
    <td align="center">swin-tiny</td>
    <td align="center">&nbsp;800</td>
    <td><a href="https://drive.google.com/file/d/19p-hxFrfhrAbMWHZN0mfVwYXmdfMucyQ/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1ZUfERQ3qpQBDqRBu2hRLsnS7EpbLIRSU/view?usp=share_link">args</a></td>
    <td><a href="https://drive.google.com/file/d/1fGlxg2mhK29Dbv_pGXRXXnQ_g3-xSR3P/view?usp=share_link">logs</a></td>
  </tr>
</table>

*_Note_: Due to the comparably big size of the <i>tiered</i>ImageNet training set, we ran pretraining for only 800 epochs for ViT to reduce the computational load.

### Meta Fine-tuning 
To start the meta fine-tuning procedure using a previously pretrained **ViT-small** architecture using 
one GPU for **100 epochs** on the **miniImageNet** training dataset using **5 steps** to adapt the **token importance weights** via online-optimisation at inference time, run:
- For a 5-way 5-shot scenario:
```
python3 train_metatrain_FewTURE.py --epochs 100 --data_path <path-to-dataset> --arch vit_small --n_way 5 --k_shot 5 --optim_steps_online 5  --chkpt_epoch 1599 --mdl_checkpoint_path <path-to-checkpoint-of-pretrained-model>
```
- For a 5-way 1-shot scenario:
```
python3 train_metatrain_FewTURE.py --epochs 100 --data_path <path-to-dataset> --arch vit_small --n_way 5 --k_shot 1 --optim_steps_online 5  --chkpt_epoch 1599 --mdl_checkpoint_path <path-to-checkpoint-of-pretrained-model>
```
To instead meta fine-tune a pretrained hierarchical swin architecture, use the `--arch swin_tiny`. 

_Note_: Replace the `--data_path <path-to-dataset>` and possibly `--mdl_checkpoint_path <path-to-checkpoint-of-pretrained-model>` with the corresponding paths where you stored the model on your machine! 



## Evaluating FewTure
To evaluate a meta-trained **ViT-small** architecture on the **miniImageNet** test dataset, run:
- For a 5-way 5-shot scenario:
```
python3 eval_FewTURE.py --data_path <path-to-dataset> --arch vit_small --n_way 5 --k_shot 5 --trained_model_type metaft --optim_steps_online 5  --mdl_checkpoint_path <path-to-checkpoint-of-metaft-model>
```
- For a 5-way 1-shot scenario:
```
python3 eval_FewTURE.py --data_path <path-to-dataset> --arch vit_small --n_way 5 --k_shot 1 --trained_model_type metaft --optim_steps_online 5  --mdl_checkpoint_path <path-to-checkpoint-of-metaft-model>
```


## Acknowledgement

This repository is built using components of the [iBOT](https://github.com/bytedance/ibot/) and 
[DINO](https://github.com/facebookresearch/dino) repositories for pre-training,
and the [DeepEMD](https://github.com/icoz69/DeepEMD) and 
[CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot) repositories for loading the few-shot datasets.


## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/mrkshllr/FewTURE/blob/main/LICENSE) file.

## Citing FewTURE
If you find this repository useful, please consider giving us a star :star: and cite our [work](https://arxiv.org/abs/2206.07267):
```
@inproceedings{
hillerma2022fewture,
title={Rethinking Generalization in Few-Shot Classification},
author={Markus Hiller and Rongkai Ma and Mehrtash Harandi and Tom Drummond},
booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=p_g2nHlMus}
}
```
If you have any questions regarding our work, please feel free to reach out!
