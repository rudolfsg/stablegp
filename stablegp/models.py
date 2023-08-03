from torch.nn import Parameter
import torch.nn.functional as F
from torch.linalg import solve_triangular
import torch
from stablegp.optimizer import LBFGS
import time 

default_jitter = 1e-10
default_positive_minimum = 1e-5

class _Variable:
    def __init__(self, value, lower=None, upper=None) -> None:
        self.lower = lower
        self.upper = upper
        value = torch.tensor(value) if not isinstance(value, torch.Tensor) else value
        self.param = Parameter(self.inverse_transform(value), requires_grad=True)
    
    def inverse_transform(self):
        pass
    
    @torch.no_grad()
    def set_value(self, value):
        value = torch.tensor(value) if not isinstance(value, torch.Tensor) else value
        value = self.inverse_transform(value)
        self.param.copy_(value)


class PositiveVariable(_Variable):
    def transform(self):
        return self.lower + F.softplus(self.param)

    def inverse_transform(self, value):
        x = value - self.lower
        return x + torch.log(-torch.expm1(-x))


class SEKernel(torch.nn.Module):
    def __init__(
        self,
        num_dimensions: int,
        lengthscale_lower: float = default_positive_minimum,
        signal_var_lower: float = default_positive_minimum,
    ):
        """
        Squared exponential kernel with one lengthscale per dimension (ARD) and lower bounds on parameters to ensure positivity.
        """
        super().__init__()
        self.raw_signal_var = PositiveVariable(1, lower=signal_var_lower)
        self.raw_lengthscale = PositiveVariable(torch.ones(num_dimensions), lower=lengthscale_lower)

        self.register_parameter("signal_var_param", self.raw_signal_var.param)
        self.register_parameter("lengthscale_param", self.raw_lengthscale.param)

    @property
    def signal_var(self):
        return self.raw_signal_var.transform()

    @property
    def lengthscale(self):
        return self.raw_lengthscale.transform()
    
    def set_signal_var(self, value):
        self.raw_signal_var.set_value(value)
    
    def set_lengthscale(self, value):
        self.raw_lengthscale.set_value(value)

    def __call__(self, a, b=None, diag=False):
        if diag:
            assert b is None
            return torch.tile(self.signal_var, (a.shape[0],))
        # NB: casting to f64 can be crucial if using f32, see https://github.com/pytorch/pytorch/issues/37734
        dt = a.dtype
        if b is None:  # better mem usage if we avoid setting b=a
            dist = torch.cdist(a.to(torch.float64)  / self.lengthscale, a.to(torch.float64)  / self.lengthscale, p=2).to(dt)
        else:
            dist = torch.cdist(a.to(torch.float64) / self.lengthscale, b.to(torch.float64) / self.lengthscale, p=2).to(dt)

        K = self.signal_var * torch.exp(-0.5 * dist.pow(2))
        return K


class SGPR(torch.nn.Module):
    def __init__(
        self,
        X: torch.tensor,
        Y: torch.tensor,
        kernel: SEKernel,
        num_inducing: int = 10,
        jitter: float = default_jitter,
        max_jitter_attempts: int = 10,
        noise_var_lower: float = default_positive_minimum
    ):
        super().__init__()

        self.X = X
        self.Y = Y
        self.num_inducing = num_inducing
        self.kernel = kernel

        self.jitter = jitter
        self.max_jitter_attempts = max_jitter_attempts

        self.inducing_variable = Parameter(
            torch.ones((num_inducing, X.shape[1]), dtype=torch.get_default_dtype()),
            requires_grad=False,
        )

        self.raw_noise_var = PositiveVariable(0.01, lower=noise_var_lower)
        self.register_parameter("noise_var_param", self.raw_noise_var.param)

    @property
    def noise_var(self):
        return self.raw_noise_var.transform()
    
    def set_noise_var(self, value):
        self.raw_noise_var.set_value(value)

    def set_inducing_variable(self, x):
        self.inducing_variable.copy_(x)

    def mean_function(self, args):
        return 0
    
    @staticmethod
    def _chol(A):
        # https://github.com/pytorch/pytorch/issues/34272
        L, info = torch.linalg.cholesky_ex(A)
        if info.gt(0).any():
            raise torch.linalg.LinAlgError("Cholesky failed: matrix is not positive semi-definite")

        # Cholesky may return nan without error
        if not torch.isfinite(L).all():
            raise torch.linalg.LinAlgError(
                f"Cholesky failed with non-finite values: {torch.isinf(L).sum().item():.2E} inf, {torch.isnan(L).sum().item():.2E} NaNs"
            )
        return L 

    def cholesky(self, A, safe=False):
        if not safe:
            return self._chol(A)
        
        jitter = self.jitter

        for i in range(self.max_jitter_attempts):
            try:
                L = self._chol(A + torch.eye(A.shape[0]) * jitter)
                
                if jitter > self.jitter:
                    print(f"Cholesky success with jitter {jitter:.2E}")
                return L
            except torch.linalg.LinAlgError:
                if i < self.max_jitter_attempts - 1:
                    jitter *= 10
        raise torch.linalg.LinAlgError(
            f"Cholesky failed with jitter {jitter:.2E} after {self.max_jitter_attempts} attempts"
        )
    
    def forward(self):
        num_data = self.Y.shape[0]
        err = self.Y - self.mean_function(self.X)
        Kdiag = self.kernel(self.X, diag=True)
        kuf = self.kernel(self.inducing_variable, self.X)
        kuu = self.kernel(self.inducing_variable, self.inducing_variable)
        L = self.cholesky(kuu, safe=True)
        sigma = torch.sqrt(self.noise_var)

        A = solve_triangular(L, kuf, upper=False) / sigma
        AAT = A @ A.t()
        B = AAT + torch.eye(self.num_inducing)
        LB = self.cholesky(B)
        Aerr = A @ err
        c = solve_triangular(LB, Aerr, upper=False) / sigma

        bound = -0.5 * num_data * torch.log(2 * torch.tensor(torch.pi))
        bound += -torch.sum(torch.log(torch.diag(LB)))
        bound -= 0.5 * num_data * torch.log(self.noise_var)
        bound += -0.5 * torch.sum(err**2) / self.noise_var
        bound += 0.5 * torch.sum(c**2)

        Qff = (A * A).sum(dim=0) * self.noise_var
        trace_term = 0.5 * (Kdiag - Qff).sum() / self.noise_var

        if trace_term < 0:
            print("Warning: Trace term negative, should be positive")
            if (Qff < (1 + 1e-3) * Kdiag).all():
                trace_term = torch.tensor(0)
            else:
                return torch.tensor(torch.nan, requires_grad=True)
            
        bound -= trace_term

        return -1 * bound

    def elbo(self):
        return -1 * self()

    def upper_bound(self):
        Kdiag = self.kernel(self.X, diag=True)
        kuu = self.kernel(self.inducing_variable, self.inducing_variable)
        kuf = self.kernel(self.inducing_variable, self.X)

        I = torch.eye(self.num_inducing)

        L = self.cholesky(kuu, safe=True)
        A = solve_triangular(L, kuf, upper=False)

        AAT = A @ A.t()
        B = I + AAT / self.noise_var
        LB = self.cholesky(B)

        c = torch.sum(Kdiag) - torch.sum(A.pow(2))

        if c < 0:
            print(f"Warning: upper bound max eigenval={c.detach().item():.4E}, replacing with 0")
        
        c = torch.maximum(c, torch.tensor(0))
        cn = self.noise_var + c

        const = -0.5 * self.X.shape[0] * torch.sum(torch.log(2 * torch.pi * self.noise_var))
        logdet = -torch.sum(torch.log(torch.diag(LB)))

        err = self.Y - self.mean_function(self.X)
        LC = self.cholesky(I + AAT / cn)
        v = solve_triangular(LC, (A @ err) / cn, upper=False)
        quad = -0.5 * torch.sum((err).pow(2)) / cn + 0.5 * torch.sum(v.pow(2))

        return const + logdet + quad

    def predict_f(self, X_test, diag=False):
        err = self.Y - self.mean_function(self.X)
        kuf = self.kernel(self.inducing_variable, self.X)
        kuu = self.kernel(self.inducing_variable, self.inducing_variable)
        Kus = self.kernel(self.inducing_variable, X_test)

        sigma = torch.sqrt(self.noise_var)

        L = self.cholesky(kuu, safe=True)  # cache alpha, qinv
        A = solve_triangular(L, kuf / sigma, upper=False)
        B = A @ A.t() + torch.eye(self.num_inducing)  # cache qinv
        LB = self.cholesky(B)  # cache alpha
        Aerr = A @ (err / sigma)
        c = solve_triangular(LB, Aerr, upper=False)
        tmp1 = solve_triangular(L, Kus, upper=False).t()
        tmp2 = solve_triangular(LB, tmp1.t(), upper=False).t()
        mean = tmp2 @ c
        if not diag:
            var = self.kernel(X_test, X_test) + tmp2 @ tmp2.t() - tmp1 @ tmp1.t()
        else:
            mean = mean.reshape(-1)
            var = (
                self.kernel(X_test, diag=True)
                + torch.sum(tmp2.pow(2), dim=1)
                - torch.sum(tmp1.pow(2), dim=1)
            )

        transform = torch.diag if not diag else lambda x: x
        # NB: number of values filled may depend on whether diag is used
        mask = transform(var) <= 0
        if mask.sum() > 0:
            fill_value = transform(var)[~mask].min().detach().item()
            min_value = transform(var)[mask].min().detach().item()
            print(f"Warning: {mask.sum().detach().item()} variances are negative up to {min_value:.4E}, replacing with {fill_value:.4E}")
            var[transform(mask)] = torch.tensor(fill_value)

        return mean + self.mean_function(X_test), var

    def predict_log_density(self, X_test, Y_test, diag=True):
        if not diag:
            raise NotImplementedError
        mean, var = self.predict_f(X_test, diag=diag)
        var += self.noise_var
        density = -0.5 * (
            torch.log(2 * torch.tensor(torch.pi))
            + torch.log(var)
            + (mean - Y_test.reshape(-1)).pow(2) / var
        )
        return density

    def fit(self, max_epochs=20, optimise_iv=True, reinit_retries=5):
        start_time = time.time()
        optimizer = LBFGS(
            [v for v in self.parameters() if v.requires_grad],
            line_search_fn="strong_wolfe",
            history_size=10,
            lr=1,
            max_iter=25 * (self.X.shape[1] + 2),  # Iterate 25 times per hyperparam,
            max_eval=15000,
            tolerance_grad=1e-5,
            ignore_previous_state=True
        )

        def closure():
            optimizer.zero_grad()
            loss = self()
            loss.backward()
            return loss

        iv = greedy_selection(self.X, self.num_inducing, self.kernel)
        self.set_inducing_variable(iv)

        max_epochs = 20
        reinit_iv = True 
        for epoch in range(max_epochs):

            optimizer.step(closure)

            with torch.no_grad():
                old_elbo = self.elbo()

            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch + 1}/{max_epochs}, ELBO: {old_elbo.item()}")

            if not reinit_iv:
                break 

            # Check if reinit improves ELBO
            old_iv = self.inducing_variable.clone()

            with torch.no_grad():
                for _ in range(reinit_retries):
                    iv = greedy_selection(self.X, self.num_inducing, self.kernel)
                    self.set_inducing_variable(iv)
                    elbo = self.elbo()
                    if torch.isnan(elbo):
                        print("Warning: NaN ELBO after IV reset, retrying...")
                        self.set_inducing_variable(old_iv)
                    else:
                        break
                if torch.isnan(elbo):
                    self.set_inducing_variable(old_iv)
                    raise RuntimeError(f"Setting new IV failed after {reinit_retries} retries")

            if elbo < old_elbo:
                self.set_inducing_variable(old_iv)

                if optimise_iv:
                    reinit_iv = False
                    self.inducing_variable.requires_grad = True
                    print("Optimising inducing points")
                    continue 

                print(
                    f"Terminating due to ELBO decrease: {old_elbo.item()} -> {elbo.item()}"
                )
                break
            else:
                print(f"Reinit ELBO improvement: {old_elbo.item()} -> {elbo.item()}")

            if epoch == max_epochs - 1:
                print("Terminating due to max_epochs")
        print(f"Optimization finished in {(time.time() - start_time):.1f}s")

@torch.no_grad()
def greedy_selection(training_inputs, M, kernel: SEKernel):
    N = training_inputs.shape[0]
    perm = torch.randperm(N)  
    training_inputs = training_inputs[perm]

    indices = torch.zeros(M, dtype=torch.int) + N
    di = kernel(training_inputs, diag=True) + 1e-12  # jitter
    indices[0] = torch.argmax(di)  

    if M == 1:
        Z = training_inputs[indices]
        return Z
    ci = torch.zeros((M - 1, N)) 
    for m in range(M - 1):
        j = int(indices[m])
        new_Z = training_inputs[j : j + 1] 
        dj = torch.sqrt(di[j])  
        cj = ci[:m, j] 
        Lraw = kernel(training_inputs, new_Z)
        L = torch.round(torch.squeeze(Lraw), decimals=20)  
        L[j] += 1e-12  # jitter
        ei = (L - (cj.reshape(-1, 1).T @ ci[:m]).squeeze()) / dj
        ci[m, :] = ei
        di -= ei**2
        di = torch.clip(di, 0, None)
        indices[m + 1] = torch.argmax(di)  

    Z = training_inputs[indices]
    indices = perm[indices]
    return Z


class GPR(torch.nn.Module):
    def __init__(
        self,
        X,
        Y,
        kernel,
        noise_var_lower=default_positive_minimum
    ):
        super().__init__()

        self.X = X
        self.Y = Y
        self.kernel = kernel

        self.raw_noise_var = PositiveVariable(0.01, lower=noise_var_lower)
        self.register_parameter("noise_var_param", self.raw_noise_var.param)

    @property
    def noise_var(self):
        return self.raw_noise_var.transform()
    
    def set_noise_var(self, value):
        self.raw_noise_var.set_value(value)

    def mean_function(self, args):
        return 0

    def forward(self):
        num_obs = self.X.shape[0]
        ks = self.kernel(self.X, self.X) + torch.eye(num_obs) * self.noise_var
        L = torch.linalg.cholesky(ks)
        err = self.Y - self.mean_function(self.X)
        alpha = torch.linalg.solve_triangular(L, err, upper=False)
        p = -0.5 * torch.sum(alpha.pow(2), dim=0)
        p -= 0.5 * num_obs * torch.log(2 * torch.tensor(torch.pi))
        p -= torch.sum(torch.log(torch.diag(L)))
        p = torch.sum(p)
        return -1 * p

    def log_marginal_likelihood(self):
        return -1 * self()

    def predict_f(
        self,
        X_test,
        diag=False,
    ):
        err = self.Y - self.mean_function(self.X)

        Knn = self.kernel(X_test, diag=diag)
        Kmn = self.kernel(self.X, X_test)
        Kmm = self.kernel(self.X, self.X) + torch.eye(self.X.shape[0]) * self.noise_var

        Lm = torch.linalg.cholesky(Kmm)
        A = torch.linalg.solve_triangular(Lm, Kmn, upper=False) 

        if not diag:
            var = Knn - A.t() @ A
        else:
            var = Knn - torch.sum(A.pow(2), dim=0)

        A = torch.linalg.solve_triangular(Lm.t(), A, upper=True)

        mean = A.t() @ err + self.mean_function(X_test)
        if diag:
            mean = mean.reshape(-1)
            var = var.t()
        return mean, var

    def predict_log_density(self, X_test, Y_test, diag=True):
        if not diag:
            raise NotImplementedError
        mean, var = self.predict_f(X_test, diag=diag)
        var += self.noise_var
        density = -0.5 * (
            torch.log(2 * torch.tensor(torch.pi))
            + torch.log(var)
            + (mean - Y_test.reshape(-1)).pow(2) / var
        )
        return density
