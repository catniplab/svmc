import torch
from torch.distributions import MultivariateNormal
from torch.optim import Adam

from . import SVMC
from .dynamics import NullDynamics
from .metric import gaussian_loss
from .resample import resample


class SVMCGP(SVMC):
    def __init__(self, n_pf, n_optim, d_latent, d_obs, log_like, rx, rz, lr=0.001, _lambda=0,
                 scheme='multinomial', gp_diffusion=1e-2):
        """
        :param log_like: log likelihood function with parameters that may need to be optimized
        :param rx: proposal distribution with parameters that need to be optimized
        :param lr: learning rate to use for optimizer
        """
        super().__init__(n_pf, n_optim, d_latent, d_obs, log_like, NullDynamics(), rx, lr, _lambda, scheme)
        self.propose = rx
        self.rz = rz
        self.optim = Adam(self.parameters(), lr=lr)  # optimizer
        self.gp_diffusion = torch.tensor(gp_diffusion)

    def compute_weight(self, Y, particles, W, M, U=None, beta=1, update=True):
        X, *_ = particles

        with torch.no_grad():
            ancestors = resample(W, M, self.scheme)
            # ancestors = torch.multinomial(torch.as_tensor(W), M)
            x_prev = X[ancestors, :]

        # Obtain samples from proposal distribution
        cat = torch.cat((x_prev, Y.repeat(M, 1)), 1)
        if U is not None:
            cat = torch.cat((cat, U.repeat(M, 1)), 1)

        # print(x_prev.shape, cat.shape)

        mu, var = self.propose(cat)
        x_samples = mu + torch.sqrt(var) * torch.randn(M, self.d_latent)

        # Compute unnormalized log weights
        log_weight = self.log_like(Y, x_samples)  # p(y|x)
        log_weight += gaussian_loss(x_samples, mu, torch.log(var))  # r(x)

        # E[p(x|z)]
        fmean, fcov = self.rz.predict(x_prev, u=U.repeat(M, 1))
        fmean = fmean.t().reshape(-1)
        log_weight += MultivariateNormal(loc=fmean, covariance_matrix=fcov).log_prob(x_samples.t().reshape(-1))
        
        return log_weight, (x_samples, x_prev.detach()), var

    def nongradient(self, particles, U=None):
        with torch.no_grad():
            x_samples, x_prevs = particles
            self.rz.update(x_prevs, x_samples, u=U)
