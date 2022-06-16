# FewTURE - Rethinking Generalization in Few-Shot Classification
Official PyTorch implementation of the paper **Rethinking Generalization in Few-Shot Classification**.

:mortar_board: :page_facing_up: Find our paper: [[arXiv]](https://arxiv.org/abs/2206.07267) &nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;&nbsp;&nbsp;&nbsp; 
:bookmark: Reference: [[BibTeX]](https://github.com/mrkshllr/FewTURE#citing-fewture)

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



## Updates :tada:
- June 2022: Code coming soon (within the next few days!) :hourglass_flowing_sand: :computer:
- June 2022: Release of our preprint on [arXiv](https://arxiv.org/abs/2206.07267)


## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/mrkshllr/FewTURE/LICENSE) file.

## Citing FewTURE
If you find this repository useful, please consider giving us a star :star: and cite our [work](https://arxiv.org/abs/2206.07267):
```
@article{hillerma2022fewture,
  title={Rethinking Generalization in Few-Shot Classification},
  author={Hiller, Markus and Ma, Rongkai and Harandi, Mehrtash and Drummond, Tom},
  journal={arXiv preprint arXiv:2206.07267},
  year={2022}
}
```
If you have any questions regarding our work, please feel free to reach out!