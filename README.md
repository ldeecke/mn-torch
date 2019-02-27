## Mode normalization

This repository contains code for the normalization approach described in *"Mode Normalization"*, L. Deecke, I. Murray, H. Bilen, [	arXiv:1810.05466](https://arxiv.org/abs/1810.05466) (2018).

Execute `train.py` to train a ResNet20 on CIFAR10 from scratch, with all its batch normalizations (BN) replaced with mode normalization (MN) – thereby jointly normalizing samples similar to each other with individual means and standard deviations.

To train a different architecture, run `train.py --model resnet56`. To learn on CIFAR100, pass `--dataset cifar100`. For an overview over commands use `train.py --help`.

This repository implements two modes of operation for MN, described below. 

#### 1. Replace all normalizations

In the default setting `--mn full`, all of the model's BNs are replaced with MN.

A predefined job is located in the `jobs/full` folder. In this seeded example a ResNet56 is trained on CIFAR100, with a final test error of **28.75%**.

> While the code has a `--seed` parameter in place, this does not necessarily guarantee portability across devices, c.f. this [note](https://pytorch.org/docs/stable/notes/randomness.html) on the official PyTorch website.

#### 2. Replace initial BN

By setting `--mn init`, only the initial BN is replaced with MN. Early on in the network the amount of variation is arguably highest, and the runtime increase from replacing a single BN unit is tiny.

In `jobs/init` we include a trial for this alternative setup. We replaced the initial BN in a ResNet20 with MN and two modes on CIFAR10, obtaining a test error of **7.73%**.

## Reference

```
@inproceedings{Deecke19,
	author       = "Deecke, Lucas and Murray, Iain and Bilen, Hakan",
	title        = "Mode Normalization",
	booktitle    = "Proceedings of the 7th International Conference on Learning Representations",
	year         = "2019"
}
```
