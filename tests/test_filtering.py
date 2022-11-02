import torch
import math
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from tqdm import tqdm
from svmc.svmc import SVMC
from svmc.proposal import MlpProposal, BootstrapProposal, MlpOptimal
from svmc.likelihood import ISOGaussian
from svmc.dynamics import BootstrapDynamics
from svmc.utils import compute_mse, compute_elbo, compute_ess
torch.manual_seed(12345)
npr.seed(54321)


# In[]
def one_step_fhn(x_prev, a=0.8, b=0.7, tau=12.5, dt=0.1, I=0.7):
    vel = torch.zeros(x_prev.shape)
    vel[:, 0] = x_prev[:, 0] - x_prev[:, 0] ** 3 / 3 - x_prev[:, 1] + I
    vel[:, 1] = (x_prev[:, 0] + a - b * x_prev[:, 1]) / tau
    return x_prev + dt * vel

# In[]
"Generate latent states from FitzHugh-Nagumo system"
d_latent = 2
var_latent = 0.1
T = 2_000

x = torch.zeros((T + 1, d_latent))
x[:, 0] = 2 * torch.rand(1) - 1
for t in range(T):
    x[t + 1, :] = one_step_fhn(x[[t]]) + math.sqrt(var_latent) * torch.randn(1, d_latent)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x[:, 0], x[:, 1])
plt.show()

# In[]
"Generate observations, y"
d_obs = 10
C = torch.randn(d_latent, d_obs)
D = 0.1 * torch.randn(1, d_obs)
var_obs = 0.1

y = x[1:] @ C + D + math.sqrt(var_obs) * torch.randn(T, d_obs)

fig = plt.figure()
ax = fig.add_subplot(111)
[ax.plot(y[:, d]) for d in range(d_obs)]
plt.show()

# In[]
"set up likelihood"
log_like = ISOGaussian(d_obs, d_latent, log_flag=False)
log_like.input_to_output.weight.data = C.T
log_like.input_to_output.weight.requires_grad = False  # we aren't learning likelihood

log_like.input_to_output.bias.data = D
log_like.input_to_output.bias.requires_grad = False  # should we learn emission distribution

log_like.tau.data.fill_(math.log(var_obs))
log_like.tau.requires_grad = False  # should we learn emission distribution

# In[]
"set up dynamics"
log_dynamics = BootstrapDynamics(d_latent, d_latent, dynamics=one_step_fhn)
log_dynamics.tau.data.fill_(math.log(var_latent))
log_dynamics.tau.requires_grad = False

# In[]
"general hyper parameters that will be used by all particle filters"
n_pf = 1_000  # number of particles used for filtering
n_optim = 4  # number of particles used for computing ELBO
iter_optim = 15  # number of sgd steps used at each time step
lr = 0.001  # learning rate of optimizer
weight_decay = 0.  # strength of weight decay


# In[]
def run_pf(pf, iter_optim=0):
    x_particles = torch.zeros(T + 1, n_pf, d_latent)  # used to store all particles generated in a run
    x_particles[0] = torch.randn(n_pf, d_latent)  # generate initial particles randomly

    normalized_weights = torch.zeros(T + 1, n_pf)  # used to store all normalized importance weights
    normalized_weights[0] = torch.ones(n_pf) / n_pf

    log_weights = torch.zeros(T, n_pf)  # used to store importance weights

    for t in tqdm(range(T)):
        particles, log_w, w_norm = pf.filter(y[t, :],
                                             x_particles[t],
                                             normalized_weights[t].numpy(),
                                             iter_optim)

        x_particles[t + 1] = particles.detach()
        normalized_weights[t + 1] = torch.from_numpy(w_norm)
        log_weights[t] = torch.from_numpy(log_w)
    return x_particles, normalized_weights, log_weights

# In[]
"Create bootstrap particle filter. This will serve as our baseline"
bootstrap_pf = SVMC(n_pf=n_pf,
                    n_optim=0,
                    d_latent=d_latent,
                    d_obs=d_obs,
                    log_like=log_like,
                    log_dynamics=log_dynamics,
                    proposal=BootstrapProposal(d_in=d_latent,
                                               d_out=d_latent,
                                               d_obs=d_obs,
                                               var=var_latent * torch.ones(1),
                                               dynamics=one_step_fhn),
                    lr=0.001,
                    scheme='stratified',
                    weight_decay=0.)

# In[]
x_particles_bpf, normalized_weights_bpf, log_weights_bpf = run_pf(bootstrap_pf)

# In[]
"compute test metrics"
bpf_ess = compute_ess(normalized_weights_bpf)
bpf_elbo = compute_elbo(torch.exp(log_weights_bpf))
bpf_mse = compute_mse(x, x_particles_bpf, normalized_weights_bpf)

# In[]
"svmc with mlp proposal"
svmc_pf = SVMC(n_pf=n_pf,
               n_optim=n_optim,
               d_latent=d_latent,
               d_obs=d_obs,
               log_like=log_like,
               log_dynamics=log_dynamics,
               proposal=MlpProposal(d_in=d_latent + d_obs,
                                    d_hidden=50,
                                    d_out=d_latent,
                                    n_layers=2,
                                    log_flag=False,
                                    relu_flag=False),
               lr=lr,
               scheme='stratified',
               weight_decay=weight_decay)

# In[]
x_particles_svmc, normalized_weights_svmc, log_weights_svmc = run_pf(svmc_pf, iter_optim)

# In[]
"compute test metrics"
svmc_ess = compute_ess(normalized_weights_svmc)
svmc_elbo = compute_elbo(torch.exp(log_weights_svmc))
svmc_mse = compute_mse(x, x_particles_svmc, normalized_weights_svmc)

# In[]
"svmc with optimal proposal"
svmc_opt_pf = SVMC(n_pf=n_pf,
                   n_optim=n_optim,
                   d_latent=d_latent,
                   d_obs=d_obs,
                   log_like=log_like,
                   log_dynamics=log_dynamics,
                   proposal=MlpOptimal(d_in=d_latent + d_obs,
                                       d_hidden=50,
                                       d_out=d_latent,
                                       dynamics=one_step_fhn,
                                       n_layers=2,
                                       log_flag=False,
                                       relu_flag=False),
                   lr=lr,
                   scheme='stratified',
                   weight_decay=weight_decay)

# In[]
x_particles_opt, normalized_weights_opt, log_weights_opt = run_pf(svmc_opt_pf, iter_optim)

# In[]
"compute test metrics"
opt_ess = compute_ess(normalized_weights_opt)
opt_elbo = compute_elbo(torch.exp(log_weights_opt))
opt_mse = compute_mse(x, x_particles_opt, normalized_weights_opt)

# In[]
fig = plt.figure(figsize=(24, 8))

ax = fig.add_subplot(131)
ax.plot(bpf_ess[1:], label='BPF')
ax.plot(svmc_ess[1:], label='SVMC')
ax.plot(opt_ess[1:], label='SVMC Opt')
ax.legend()


ax = fig.add_subplot(132)
ax.plot(bpf_elbo, label='BPF')
ax.plot(svmc_elbo, label='SVMC')
ax.plot(opt_elbo, label='SVMC Opt')
ax.legend()

ax = fig.add_subplot(133)
ax.plot(bpf_mse[1:], label='BPF')
ax.plot(svmc_mse[1:], label='SVMC')
ax.plot(opt_mse[1:], label='SVMC Opt')
ax.legend()

fig.tight_layout()
plt.show()
