<!--# PHDimGeneralization
Official implementation of "Intrinsic Dimension, Persistent Homology and Generalization in Neural Networks", NeurIPS 2021.
-->
# [Intrinsic Dimension, Persistent Homology and Generalization in Neural Networks]([https://arxiv.org/abs/1901.01255](https://arxiv.org/abs/2111.13171)), NeurIPS 2021.
[Tolga Birdal](http://tolgabirdal.github.io/), Aaron Lou, Leonidas Guibas, Umut Simsekli\
Stanford University | INRIA

## Overview

This package provides computation of ph dimension of neural network trajectories. In particular, computation is done in ```topology.py```. The code to produce the analysis experiments are given in ```train_analysis.py```, and the code to produce the regularization experiments are given in ```train_reqularize.py```.

## Requirements

The baseline code requires [PyTorch](https://pytorch.org/), which can be installed directly through a software package manager like pip or conda. However, the topological PH requirements are a bit more complex.

### CPU (non-Differentiable)

The function ```calculate_ph_dim```, which computes topology on CPU and is not differentiable, requires [Ripser](https://ripser.scikit-tda.org/en/latest/). This can be installed using

```
pip install Cython
pip install Ripser
```

### GPU (Differentiable)

The function ```calculate_ph_dim_gpu```, which computes topology on GPU and is differentiable, requires [TorchPH](https://c-hofer.github.io/torchph/). This is more difficult to install (due to various dependencies including C++ version). We recommend take a look at the [installation page](https://c-hofer.github.io/torchph/install/index.html).

```shell
@article{birdal2021intrinsic,
  title={Intrinsic dimension, persistent homology and generalization in neural networks},
  author={Birdal, Tolga and Lou, Aaron and Guibas, Leonidas J and Simsekli, Umut},
  journal={Advances in neural information processing systems},
  volume={34},
  pages={6776--6789},
  year={2021}
}
```
