import torch
from stablegp.models import SGPR, SEKernel
import numpy as np
from torch.nn.functional import mse_loss
import time

seed = 0
torch.manual_seed(seed)
torch.set_default_dtype(torch.float64)

# Generate dataset
N = 1000
X = torch.linspace(10, 25, N).reshape(-1, 1)
Y_noiseless = torch.sin(X)
Y = Y_noiseless + 0.7 * torch.randn(N).reshape(-1, 1)

X_test = X
Y_test = Y

# Define initial model
kernel = SEKernel(num_dimensions=X.shape[1])
model = SGPR(
    X,
    Y,
    kernel,
    num_inducing=10,
)

# Store model evaluation metrics through a callback
eval_data = []


@torch.no_grad()
def create_callback():
    def callback(iteration):
        if iteration % 5 != 0:  # evaluate every 5 iterations
            return
        pred_mean, pred_var = model.predict_f(X_test)
        data = dict(
            iteration=iteration,
            noise_var=model.noise_var.item(),
            signal_var=model.kernel.signal_var.item(),
            lengthscale=model.kernel.lengthscale.cpu().numpy().tolist(),
            M=model.inducing_variable.shape[0],
            elbo=model.elbo().item(),
            upper_bound=model.upper_bound().item(),
            nlpd=-1 * model.predict_log_density(X_test, Y_test).mean().item(),
            rmse=np.sqrt(mse_loss(pred_mean, Y_test).item()),
            timestamp=time.time(),
        )
        eval_data.append(data)

    return callback


# Fit the model
model.fit_automatic(callback=create_callback())
