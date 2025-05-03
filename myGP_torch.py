import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, MultitaskKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.utils.errors import NotPSDError


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Single-output GP model with default Matern kernel
    """
    def __init__(self, train_X, train_Y, likelihood, kernel=None):
        super().__init__(train_X, train_Y, likelihood)
        input_dim = train_X.shape[1]
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel is None:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=input_dim))
        else:
            self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MultitaskExactGPModel(gpytorch.models.ExactGP):
    """
    Multi-output GP model with default Matern kernel
    """
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=train_y.shape[-1]
        )
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)  # important

    def posterior(self, X): # will be used within BOTorch
        self.eval()  
        return self(X)

class NN_Kernel(torch.nn.Sequential):
    """
    Neural layer used in deep learning kernel
    """
    def __init__(self, in_dim, out_dim, depth=3, neurons=64):
        layers = []
        layers.append(torch.nn.Linear(in_dim, neurons))
        layers.append(torch.nn.ReLU())

        for _ in range(depth - 1):
            layers.append(torch.nn.Linear(neurons, neurons))
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(neurons, out_dim))
        super().__init__(*layers)


class DKL_GPModel(gpytorch.models.ExactGP):
    """
    Multi-output GP model with default deep learning layer and RBF kernel
    """
    def __init__(self, train_x, train_y, likelihood, NN_kernel):
        super().__init__(train_x, train_y, likelihood)
        self.NN_kernel = NN_kernel
        self.num_outputs = train_y.shape[1]
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=self.num_outputs)
        self.covar_module = MultitaskKernel(
            ScaleKernel(RBFKernel()), num_tasks=self.num_outputs, rank=2)

    def forward(self, x):
        x = self.NN_kernel(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

    def posterior(self, X): # will be used within BOTorch
        self.eval()  
        return self(X)
    
class soGP_torch:
    """
    Single-output GP model
    """
    def __init__(self, train_X, train_Y, kernel = "RBF", training_params=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.X = torch.tensor(train_X, dtype=torch.float32).to(self.device)
        self.Y = torch.tensor(train_Y, dtype=torch.float32).squeeze().to(self.device)
        self.N = self.Y.shape[0]
        self.n = self.X.shape[1]
        self.kernel = kernel
        self.training_params = training_params if training_params else {"max_iter": 1000, "optimizer": "Adam", "num_restarts": 5}

        self._train_gp(self.X, self.Y)

    def _train_gp(self, X, Y):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.likelihood.noise = 1e-2
        self.likelihood.noise_covar.raw_noise.requires_grad_(False)
        
        if self.kernel.lower() == "rbf":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.n))
        elif self.kernel.lower() == "matern32":
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=self.n)
                )
        self.model = ExactGPModel(X, Y, self.likelihood, kernel).to(self.device)

        max_iter = self.training_params["max_iter"] 
        opti = self.training_params["optimizer"] 
        num_restarts = self.training_params["num_restarts"]
        best_model_so_far = None
        best_loss = float('inf')

        # Perform multiple restarts
        for restart in range(num_restarts):
            self.model.train()
            self.likelihood.train()

            if opti == "Adam":
                optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-3, weight_decay=1e-3)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
                def closure():
                    optimizer.zero_grad()
                    output = self.model(X)
                    loss = -mll(output, Y)
                    loss.backward()
                    optimizer.step()
                try:
                    for i in range(max_iter):
                        closure()
                except NotPSDError: # may need a better way to handle NotPSDError
                    continue   

            elif opti == "LBFGS":
                optimizer = torch.optim.LBFGS(self.model.parameters(), max_iter=max_iter)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
                def closure():
                    optimizer.zero_grad()
                    output = self.model(X)
                    loss = -mll(output, Y)
                    loss.backward()
                    return loss
                optimizer.step(closure)
                closure()

            # Track the best model based on the log marginal likelihood
            with torch.no_grad():
                self.model.eval()
                self.likelihood.eval()
                output = self.model(X)
                loss = -mll(output, Y).item()  # Track the loss as a scalar
                if loss < best_loss:
                    best_loss = loss
                    best_model_so_far = self.model

        # Set the best model from restarts as the final model
        self.model = best_model_so_far

    def update_data(self, X, Y, max_size=None):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).squeeze().to(self.device)

        if max_size is not None and len(Y) >= max_size:
            X = X[-max_size:]
            Y = Y[-max_size:]

        self.X = X
        self.Y = Y
        self.N = self.Y.shape[0]
        self.model.set_train_data(inputs=self.X, targets=self.Y, strict=False)
        self._train_gp(self.X, self.Y)


    def predict(self, x):
        self.model.eval()
        self.likelihood.eval()

        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(x))
            mean = preds.mean
            var = preds.variance

        return mean.cpu().numpy(), var.cpu().numpy()

class moGP_torch:
    """
    Multi-output GP model
    """
    def __init__(self, train_X, train_Y, kernel="DLK", NN_params=None, training_params=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.X = torch.tensor(train_X, dtype=torch.float32).to(self.device)
        self.Y = torch.tensor(train_Y, dtype=torch.float32).to(self.device)
        self.N, self.ni = self.X.shape
        self.no = self.Y.shape[1]
        self.kernel = kernel
        self.training_params = training_params if training_params else {"max_iter": 1000, "optimizer": "Adam", "num_restarts": 5}
        self.NN_params = NN_params if NN_params else {"depth": 3, "neurons": 64}
        self._train_gp(self.X, self.Y)
 
    def _train_gp(self, X, Y):      
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.no).to(self.device)
        # fix the noise
        self.likelihood.task_noises = 1e-2*torch.ones(self.no).to(self.device)  # set all tasks noise to 1e-2
        self.likelihood.raw_task_noises.requires_grad_(False)  # freeze noise

        if self.kernel.lower() == "rbf":
            base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.ni))
            # Use LCM kernel for multi-output
            kernel = gpytorch.kernels.LCMKernel(
                base_kernels=[base_kernel], num_tasks=self.no)
            self.model = MultitaskExactGPModel(X, Y, self.likelihood, kernel).to(self.device)
        elif self.kernel.lower() == "matern32":
            base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=self.ni))
            # Use LCM kernel for multi-output
            kernel = gpytorch.kernels.LCMKernel(
                base_kernels=[base_kernel], num_tasks=self.no)
            self.model = MultitaskExactGPModel(X, Y, self.likelihood, kernel).to(self.device)
        elif self.kernel.lower() == "dlk":
            depth = self.NN_params["depth"]
            neurons= self.NN_params["neurons"]
            deep_kernel = NN_Kernel(self.ni, self.no, depth=depth, neurons=neurons).to(self.device)
            self.model = DKL_GPModel(X, Y, self.likelihood, deep_kernel).to(self.device)

        max_iter = self.training_params["max_iter"] 
        opti = self.training_params["optimizer"] 
        num_restarts = self.training_params["num_restarts"]
        best_model_so_far = None
        best_loss = float('inf')
        
        # Perform multiple restarts
        for restart in range(num_restarts):
            self.model.train()
            self.likelihood.train()

            if opti == "Adam":
                optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-3, weight_decay=1e-3)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
                def closure():
                    optimizer.zero_grad()
                    output = self.model(X)
                    loss = -mll(output, Y)
                    loss.backward()
                    optimizer.step()
                try: # may need a better way to handle NotPSDError
                    for i in range(max_iter):
                        closure()
                except NotPSDError:
                    continue

            elif opti == "LBFGS":
                optimizer = torch.optim.LBFGS(self.model.parameters(), max_iter=max_iter)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
                def closure():
                    optimizer.zero_grad()
                    output = self.model(X)
                    loss = -mll(output, Y)
                    loss.backward()
                    return loss
                optimizer.step(closure)
                closure()

            # Track the best model based on the log marginal likelihood
            with torch.no_grad():
                self.model.eval()
                self.likelihood.eval()
                output = self.model(X)
                loss = -mll(output, Y).item()  # Track the loss as a scalar
                if loss < best_loss:
                    best_loss = loss
                    best_model_so_far = self.model

        # Set the best model from restarts as the final model
        self.model = best_model_so_far

    def update_data(self, train_X, train_Y, max_size=None):
        X = torch.tensor(train_X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(train_Y, dtype=torch.float32).to(self.device)

        if max_size is not None and Y.shape[0] >= max_size:
            X = X[-max_size:]
            Y = Y[-max_size:]

        self.X = X
        self.Y = Y
        self.model.set_train_data(inputs=self.X, targets=self.Y, strict=False)

        self._train_gp(self.X, self.Y)

    def predict(self, x):
        self.model.eval()
        self.likelihood.eval()

        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(x))
            mean = preds.mean 
            var = preds.variance

        return mean.cpu().numpy(), var.cpu().numpy()
