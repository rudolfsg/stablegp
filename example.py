
import torch
from stablegp.models import SGPR, SEKernel
import matplotlib.pyplot as plt

seed = 0
torch.manual_seed(seed)
torch.set_default_dtype(torch.float64)

# Generate dataset
N = 500

X = torch.linspace(10, 25, N).reshape(-1, 1)
Y_noiseless = torch.sin(X)
Y = Y_noiseless + 0.5 * torch.randn(N).reshape(-1, 1)

@torch.no_grad()
def plot(model, title):
    pred_mean, pred_var = model.predict_f(X, diag=True)
    pred_mean = pred_mean.reshape(-1)
    pred_std = torch.sqrt(pred_var).reshape(-1)

    plt.plot(X, Y_noiseless, label="Noise-free function", linestyle="dotted", linewidth=4, c='blue')
    plt.scatter(X, Y, label="Observations", facecolors='none', edgecolors='b', alpha=.6)
    plt.plot(X, pred_mean, label="Mean prediction", c='orange', linewidth=4)
    plt.fill_between(
        X.ravel(),
        pred_mean - 1.96 * pred_std,
        pred_mean + 1.96 * pred_std,
        alpha=0.3,
        label=r"95% confidence interval",
        color='orange'
    )
    plt.title(title)


kernel = SEKernel(num_dimensions=X.shape[1])
model = SGPR(
    X,
    Y,
    kernel,
    num_inducing=10,
)

fig, axes = plt.subplots(ncols=2, figsize=(12,4))
plt.sca(axes[0])
plot(model, title=f"Before optimization: ELBO={model.elbo().item():.2f}")

model.fit()

plt.sca(axes[1])
pred_mean, pred_var = model.predict_f(X, diag=True)
plot(model, title=f"After optimization: ELBO={model.elbo().item():.2f}")

plt.legend()
plt.tight_layout()
plt.savefig('example.png')