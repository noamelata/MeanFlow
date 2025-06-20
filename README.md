## Unofficial Pytorch implementation of "Mean Flows for One-step Generative Modeling" by Geng et al. 

This repository contains an unofficial PyTorch implementation of the paper [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447) by Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, Kaiming He. 

The paper introduces a new generative model called Mean Flow, which is a flow-based model that can generate high-quality images in a single step.

This codebase is based on [DiT](https://github.com/facebookresearch/DiT/) which is the official implementation of the paper [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748).

## Installation

Please install the following repositories:
`pytorch torchvision numpy tqdm diffusers accelerate timm`

For training, please download [ImageNet](https://www.image-net.org/download.php).

For FID evaluations please download the ImageNet reference stats:
```bash
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```
We have used [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID evaluation.


## Sampling

Run
```bash
torchrun --standalone --nproc_per_node=<num gpus> sample.py --ckpt <checkpoint path>
```

## Checkpoints
| DiT Model                                                             | Image Resolution | CFG | EPOCH | Steps | FID-50K |
|-----------------------------------------------------------------------|------------------|-----|-------|-------|--------|
| [B/4](https://drive.google.com/file/d/19mqYNTDH_wLb-_Y5E9kc8LwRSnt-UFKv/view?usp=sharing)  | 256x256          | 3.0 | 80    | 1     | 14.89  |

## Training

Run
```bash
torchrun --standalone --nproc_per_node=<gpus> train.py --data-path <path to ImageNet train data> --cfg <CFG>
```

Alternatively, you can pre-compute the VAE encoded dataset with:
```bash
torchrun --standalone --nproc_per_node=<gpus> encode_imagenet.py --data-path <path to ImageNet train data> --out-path <new path for encoded data>
```

And then run:
```bash
torchrun --standalone --nproc_per_node=<gpus> train.py --data-path <new path for encoded data> --encoded-data --cfg <CFG>
```

The training hyperparams correspond to optimal hyperparameters found in Sec. 5 of the paper.
To run with a different set of hyperparams, use `train.py -h` to view available options and syntax.

## References

```bibtex
@article{geng2025mean,
  title={Mean flows for one-step generative modeling},
  author={Geng, Zhengyang and Deng, Mingyang and Bai, Xingjian and Kolter, J Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2505.13447},
  year={2025}
}
@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748},
}
```

Our implementation is based on [DiT](https://github.com/facebookresearch/DiT/), inhereting its license ([`LICENSE.txt`](LICENSE.txt)) and copyright ([`ACKNOWLEDGEMENTS.txt`](ACKNOWLEDGEMENTS.txt)).
