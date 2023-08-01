# StableGP

This library implements a robust version of Sparse Variational Gaussian Process regression in PyTorch.

![](example.png)

## Example

```
pip install stablegp
```

```python
import torch
from stablegp import SGPR, SEKernel

X = torch.linspace(0, 15, 100).reshape(-1, 1)
Y = torch.sin(X)

kernel = SEKernel(num_dimensions=X.shape[1])
model = SGPR(X, Y, kernel, num_inducing=10)

model.fit()
pred_mean, pred_var = model.predict_f(X)
```

## Background

Gaussian process regression (GPR) is a model that tries to approximate the function $f(x)$ based on noisy observations $y = f(x) + \epsilon$ where $\epsilon$ is Gaussian noise. It is a powerful model since (1) it provides uncertainty estimates of each prediction (2) it's non-parametric and can fit many datasets well. However, for $N$ training observations a GP requires $\mathcal{O}(N^2)$ memory and $\mathcal{O}(N^3)$ computational complexity.

Sparse Gaussian process regression (SGPR) approximates the full model by using $M < N$ datapoints to approximate the whole dataset which reduces memory and computational complexities to $\mathcal{O}(NM)$ and $\mathcal{O}(NM^2)$ respectively. Despite these benefits, SGPR can be difficult to apply in practice to due occurence of numerical errors. 

This library aims to make SGPR easy by providing a stable implementation that has been tested in more than 30 UCI datasets.  


## Things to be aware of 

1. **Hyperparameters**: The number of inducing points `num_inducing` is the only hyperparameter that requires tuning. Using more inducing points is guaranteed to lead to a better model at the cost of higher computation time and memory usage so the optimal value depends on your goals - fast training/inference vs predictive accuracy. 

2. **Data preprocessing**: You should standardise your data to mean 0 and unit variance e.g. via SciPy [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). This reduces the chance of numerical errors. 

3. **Precision**: PyTorch uses float32 by default but GPs work best with float64 which can be enabled by adding `torch.set_default_dtype(torch.float64)` at the start of your script or by casting individual tensors, e.g. `X_train = X_train.to(torch.float64)`. Using float32 may lead to numerical instabilities and worse performance, however, float32 is still worth a try due to halved memory usage and because modern GPUs are much faster with float32. 

4. **Training**: we suggest using the training loop provided in `SGPR.fit()` since it has been tested and known to work well.

5. **Limitations**: currently we only support regression with a Gaussian likelihood and the squared exponential kernel. 


## Acknowledgements 

This library is heavily based on the work of [GPFlow](https://github.com/GPflow/GPflow) and [RobustGP](https://github.com/markvdw/RobustGP) which can be cited via

```
@ARTICLE{GPflow2017,
  author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
	Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
	Ghahramani, Zoubin and Hensman, James},
    title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
  journal = {Journal of Machine Learning Research},
  year    = {2017},
  month = {apr},
  volume  = {18},
  number  = {40},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v18/16-537.html}
}

@article{burt2020gpviconv,
  author  = {David R. Burt and Carl Edward Rasmussen and Mark van der Wilk},
  title   = {Convergence of Sparse Variational Inference in Gaussian Processes Regression},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {131},
  pages   = {1-63},
  url     = {http://jmlr.org/papers/v21/19-1015.html}
}
```