import numpy as np
from myGP_torch import *
from pyswarms.single.global_best import GlobalBestPSO
from util import linearly_spaced_combinations

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import Normalize

from botorch.acquisition import AcquisitionFunction
# set device globally

class SumVarianceAcquisition(AcquisitionFunction):
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    def forward(self, X):
        if X.dim() == 2:
            X = X.unsqueeze(-2)

        posterior = self.model.posterior(X)
        variance = posterior.variance

        if variance.dim() > 2:
            variance = variance.sum(dim=-1)
        variance = variance.sum(dim=-1)

        return variance


class Context_BayesOpt:
    r"""Contextual BayesOpt class."""

    def __init__(self, variable_dims, context_dims, print_level=False,
                 kernel_OM="RBF", kernel_SM="DLK", NN_params=None, training_params=None):
        self.nz = variable_dims
        self.nt = context_dims
        self.print_level = print_level
        self.kernel_OM = kernel_OM
        self.kernel_SM = kernel_SM
        self.best_SOL = None
        self.best_OBJ = -np.inf
        self.NN_params = NN_params
        self.training_params = training_params

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reset_sol(self):
        self.best_SOL = None
        self.best_OBJ = -np.inf

    def set_feasible_set(self, var_bounds, ctx_bounds, n_pts=101):
        self.ZZ = linearly_spaced_combinations(var_bounds, n_pts)
        self.TT = linearly_spaced_combinations(ctx_bounds, n_pts)
        self.var_set_size = len(self.ZZ)
        self.ctx_set_size = len(self.TT)

    def evaluate_cand(self, Z, J):
        if J > self.best_OBJ:
            self.best_OBJ = J
            self.best_SOL = Z

    def run_Inner(self, true_fcn, T, j_max, j_init, beta, beta_rate, tol, patience):
        train_X = torch.zeros(0, self.nz, dtype=torch.double, device=self.device)
        train_J = torch.zeros(0, 1, dtype=torch.double, device=self.device)
        
        prev_Z = None
        no_improvement_count = 0
        bounds = torch.tensor([self.ZZ[0], self.ZZ[-1]], dtype=torch.double, device=self.device)

        for j in range(j_max):
            if j < j_init:
                Z = self.ZZ[np.random.randint(0, self.var_set_size)]
            else:
                beta *= beta_rate
                ucb = UpperConfidenceBound(self.GP_o, beta=beta)
                Z_cand, _ = optimize_acqf(
                    acq_function=ucb, bounds=bounds,
                    q=1, num_restarts=10, raw_samples=20,
                    options={"dtype": torch.double, "device": self.device}
                )
                Z = Z_cand.cpu().numpy().flatten()

            if self.print_level:
                print("Bayes Opt proposes to sample ", Z)

            J = true_fcn.evaluate_true(Z, T)
            J_val = torch.tensor(J, dtype=torch.double, device=self.device).view(1, 1)
            self.evaluate_cand(Z, J)

            # Update training set
            train_X = torch.cat([train_X, torch.tensor(Z, dtype=torch.double, device=self.device).view(1, -1)], dim=0)
            train_J = torch.cat([train_J, J_val], dim=0)
            if j >= j_init - 1:
                try:
                    kernel = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel(ard_num_dims=self.nz)
                    ).to(self.device)
                    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
                    likelihood.noise = 1e-2
                    likelihood.noise_covar.raw_noise.requires_grad_(False)
                    new_gp = SingleTaskGP(
                        train_X=train_X,
                        train_Y=train_J,
                        covar_module=kernel,
                        likelihood=likelihood,
                        input_transform=Normalize(d=self.nz),
                        outcome_transform=Standardize(m=1)).to(self.device)
                    mll = ExactMarginalLogLikelihood(new_gp.likelihood, new_gp)
                    fit_gpytorch_mll(mll)
                    self.GP_o = new_gp  # only replace the model if training succeeds

                except Exception as e:
                    print("GP fitting failed at iteration", j)
                    break

            # Convergence check
            if prev_Z is not None:
                solution_change = np.max(np.abs(Z - prev_Z))
                if solution_change < tol:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
            if no_improvement_count >= patience:
                print(f"Converged after {j + 1} iterations.")
                break

            prev_Z = Z

    def run_Outer(self, true_fcn, k_max=30, k_init=5, j_max=30, j_init=5,
                  beta=1e1, beta_rate=1.0, tol=1e-2, patience=3):
        train_Z = torch.zeros(0, self.nz, dtype=torch.float32, device=self.device)
        train_T = torch.zeros(0, self.nt, dtype=torch.float32, device=self.device)
        bounds = torch.tensor([self.TT[0], self.TT[-1]], dtype=torch.float32, device=self.device)

        for k in range(k_max):
            print("Iteration #", 1 + k)

            if k < k_init:
                T = self.TT[np.random.randint(0, self.ctx_set_size)]
            else:
                AdaptSamp = SumVarianceAcquisition(self.GP_s.model)
                T_cand, _ = optimize_acqf(
                    acq_function=AdaptSamp, bounds=bounds,
                    q=1, num_restarts=10, raw_samples=500,
                    options={"sample_with_replacement": True, "dtype": torch.float32, "device": self.device}
                )
                T = T_cand.cpu().numpy().flatten()

            if self.print_level:
                print("Sample the context:", T)

            self.reset_sol()
            self.run_Inner(true_fcn, T, j_max, j_init, beta, beta_rate, tol, patience)
            Z = self.best_SOL

            if self.print_level:
                print("Inner loop returns solution:", Z)

            train_Z = torch.cat([train_Z, torch.tensor(Z, dtype=torch.float32, device=self.device).view(1, -1)], dim=0)
            train_T = torch.cat([train_T, torch.tensor(T, dtype=torch.float32, device=self.device).view(1, -1)], dim=0)

            if k >= k_init - 1:
                self.GP_s = moGP_torch(train_T, train_Z, self.kernel_SM,
                                       NN_params=self.NN_params, training_params=self.training_params)

        return train_Z.cpu(), train_T.cpu()