import torch
import torch.nn as nn
from torch.nn import Parameter

from .base import Proposal


class MLPProposal(Proposal):
    # 2 layer perceptron
    def __init__(self, d_in, d_hidden, d_out, log=True):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.log = log
        self.add_module("input_to_hidden",  nn.Linear(d_in, d_hidden))
        self.add_module("hidden_to_mean",  nn.Linear(d_hidden, d_out))  # proposal mean
        self.add_module("hidden_to_tau",  nn.Linear(d_hidden, d_out))  # proposal variance

    def get_proposal_params(self, input):
        hidden_layer = self.input_to_hidden(input)
        transform_layer = torch.relu(hidden_layer)
        mean = self.hidden_to_mean(transform_layer) + input[:, :self.d_out]
        var = torch.exp(torch.min(self.hidden_to_tau(transform_layer), torch.tensor(10.)))  # to avoid constrained optimization
        if self.log:
            var = torch.log(1 + var)  # to avoid constrained optimization
        return mean, var

    def sample(self, input, no_samples):
        mean, var = self.get_proposal_params(input)  # obtain mean and variance
        return mean + torch.sqrt(var) * torch.randn(no_samples,
                                                    self.d_out)  # return samples from N(mu(theta), var(theta)I)

    def forward(self, input):
        return self.get_proposal_params(input)


class DNNProposal(Proposal):
    # MLP with K layers
    def __init__(self, d_in, d_out, hiddens, log=False, bias=False, activation='relu', bn=False):
        super().__init__()
        self.numHiddenLayers = len(hiddens)  # number of hidden layers in the network
        self.bn = bn
        self.d_in = d_in
        self.d_out = d_out
        modules_mean = []
        modules_var = []
        for idx in range(self.numHiddenLayers):
            if idx == 0:
                modules_mean.append(nn.Linear(d_in, hiddens[idx], bias=bias))
                modules_var.append(nn.Linear(d_in, hiddens[idx], bias=bias))
            else:
                modules_mean.append(nn.Linear(hiddens[idx - 1], hiddens[idx], bias=bias))
                modules_var.append(nn.Linear(hiddens[idx - 1], hiddens[idx], bias=bias))
            if activation == 'relu':
                modules_mean.append(nn.ReLU())
                modules_var.append(nn.ReLU())
            else:
                modules_mean.append(nn.Tanh())
                modules_var.append(nn.Tanh())
            if bn:
                modules_mean.append(nn.BatchNorm1d(hiddens[idx]))
                modules_var.append(nn.BatchNorm1d(hiddens[idx]))
        modules_mean.append(nn.Linear(hiddens[-1], d_out, bias=bias))
        modules_var.append(nn.Linear(hiddens[-1], 1, bias=bias))
        self.mean = nn.Sequential(*modules_mean)
        self.var = nn.Sequential(*modules_var)
        self.log = log

    def get_proposal_params(self, input):
        mean = self.mean(input) + input[:, :self.d_out]
        var = torch.exp(self.var(input))
        if self.log:
            var = torch.log(1 + var)
        return mean, var

    def forward(self, input):
        return self.get_proposal_params(input)


class DiagMLPProposal(Proposal):
    # 2 layer perceptron
    def __init__(self, d_in, d_hidden, d_out, log=True):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.log = log

        self.add_module("input_to_hidden",  nn.Linear(d_in, d_hidden))
        self.add_module("hidden_to_mean",  nn.Linear(d_hidden, d_out))  # proposal mean
        self.add_module("hidden_to_tau",  nn.Linear(d_hidden, d_out))  # proposal variance

    def get_proposal_params(self, input):
        hidden_layer = self.input_to_hidden(input)
        transform_layer = torch.tanh(hidden_layer)
        mean = self.hidden_to_mean(transform_layer) + input[:, :self.d_out]
        var = torch.exp(torch.min(self.hidden_to_tau(transform_layer), torch.tensor(10.)))  # to avoid constrained optimization
        if self.log:
            var = torch.log(1 + var)  # to avoid constrained optimization
        return mean, var

    def sample(self, input, no_samples):
        mean, var = self.get_proposal_params(input)  # obtain mean and variance
        return mean + torch.sqrt(var) * torch.randn(no_samples,
                                                    self.d_out)  # return samples from N(mu(theta), var(theta)I)

    def forward(self, input):
        return self.get_proposal_params(input)


class MLPOptimal(Proposal):
    # 2 layer perceptron
    def __init__(self, d_in, d_hidden, d_out, dynamics, log=True, d_stim=0, normalize=True):
        super().__init__()
        self.d_in = d_in
        self.d_stim = d_stim
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.log = log
        self.dynamics = dynamics
        self.loc = torch.zeros(1, d_in)
        self.scale = torch.ones(1, d_in)

        self.add_module("input_to_hidden", nn.Linear(d_in, d_hidden))
        self.add_module("hidden_to_mean", nn.Linear(d_hidden, d_out))  # proposal mean
        self.add_module("hidden_to_tau", nn.Linear(d_hidden, d_out))  # proposal variance

    def get_proposal_params(self, input):
        z = torch.cat((self.dynamics(input[:, :self.d_out + self.d_stim]),
                                                       input[:, self.d_out + self.d_stim:]), 1)
        z = (z - self.loc) / self.scale
        hidden_layer = self.input_to_hidden(z)
        # hidden_layer = self.input_to_hidden(torch.cat((self.dynamics(input[:, :self.d_out + self.d_stim]),
        #                                                input[:, self.d_out + self.d_stim:]), 1))
        transform_layer = torch.relu(hidden_layer)
        mean = self.hidden_to_mean(transform_layer) + input[:, :self.d_out]
        var = torch.exp(
            torch.min(self.hidden_to_tau(transform_layer), torch.tensor(10.)))  # to avoid constrained optimization
        if self.log:
            var = torch.log(1 + var)  # to avoid constrained optimization
        return mean, var

    def sample(self, input, no_samples):
        mean, var = self.get_proposal_params(input)  # obtain mean and variance
        return mean + torch.sqrt(var) * torch.randn(no_samples,
                                                    self.d_out)  # return samples from N(mu(theta), var(theta)I)

    def forward(self, input):
        return self.get_proposal_params(input)


class MLPPoissProposal(Proposal):
    # 2 layer perceptron
    def __init__(self, d_in, d_hidden, d_out, d_neuron, du=0, log=True, window=25):
        super().__init__()
        self.d_in = d_in  # input passed in is the previous latent state and a window of previous spikes
        self.d_hidden = d_hidden  # number of neurons in hidden layer
        self.d_out = d_out  # dimension of latent space
        self.d_neuron = d_neuron  # number of neurons being recorded
        self.du = du
        self.log = log
        self.window = window  # how many previous bins are being used
        self.add_module("input_to_hidden",  nn.Linear(d_out + self.d_neuron + du, d_hidden))
        self.add_module("hidden_to_mean",  nn.Linear(d_hidden, d_out))  # proposal mean
        self.add_module("hidden_to_tau",  nn.Linear(d_hidden, 1))  # proposal variance

    def get_proposal_params(self, input):
        if self.window > 1 and input[0, self.d_out:].shape[0] != self.d_neuron:
            with torch.no_grad():
                spikes = input[0, self.d_out + self.du:]  # Get the spikes
                window = int(spikes.shape[0] / self.d_neuron)
                rates = torch.mean(spikes.reshape(self.d_neuron, window), 1)  # compute estimate of firing rate

            hidden_layer = self.input_to_hidden(torch.cat((input[:, :self.d_out + self.du], rates.repeat(input.shape[0], 1)), 1))
        else:
            hidden_layer = self.input_to_hidden(input)
        transform_layer = torch.tanh(hidden_layer)
        # mean = self.hidden_to_mean(transform_layer) + input[:, :self.d_out]
        mean = self.hidden_to_mean(transform_layer)
        var = torch.exp(torch.min(self.hidden_to_tau(transform_layer), torch.tensor(10.)))  # to avoid constrained optimization
        if self.log:
            var = torch.log(1 + var)  # to avoid constrained optimization
        return mean, var

    def sample(self, input, no_samples):
        mean, var = self.get_proposal_params(input)  # obtain mean and variance
        return mean + torch.sqrt(var) * torch.randn(no_samples,
                                                    self.d_out)  # return samples from N(mu(theta), var(theta)I)

    def forward(self, input):
        return self.get_proposal_params(input)


class DNNPoissProposal(Proposal):
    # MLP with K layers
    def __init__(self, d_in, d_out, hiddens, log=False, bias=False, activation='relu', bn=False):
        super().__init__()
        self.numHiddenLayers = len(hiddens)  # number of hidden layers in the network
        self.bn = bn
        self.d_in = d_in
        self.d_out = d_out
        modules_mean = []
        modules_var = []
        for idx in range(self.numHiddenLayers):
            if idx == 0:
                modules_mean.append(nn.Linear(d_in, hiddens[idx], bias=bias))
                modules_var.append(nn.Linear(d_in, hiddens[idx], bias=bias))
            else:
                modules_mean.append(nn.Linear(hiddens[idx - 1], hiddens[idx]))
                modules_var.append(nn.Linear(hiddens[idx - 1], hiddens[idx]))
            if activation == 'relu':
                modules_mean.append(nn.ReLU())
                modules_var.append(nn.ReLU())
            else:
                modules_mean.append(nn.Tanh())
                modules_var.append(nn.Tanh())
            if bn:
                modules_mean.append(nn.BatchNorm1d(hiddens[idx]))
                modules_var.append(nn.BatchNorm1d(hiddens[idx]))
        modules_mean.append(nn.Linear(hiddens[-1], d_out))
        modules_var.append(nn.Linear(hiddens[-1], 1))
        self.mean = nn.Sequential(*modules_mean)
        self.var = nn.Sequential(*modules_var)
        self.log = log

    def get_proposal_params(self, input):
        mean = self.mean(input) + input[:, :self.d_out]
        var = torch.exp(self.var(input))
        if self.log:
            var = torch.log(1 + var)
        return mean, var

    def forward(self, input):
        return self.get_proposal_params(input)


class LinearProposal(Proposal):
    # 2 layer perceptron
    def __init__(self, d_in, d_hidden, d_out, log=True):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.log = log

        # self.add_module("input_to_hidden",  nn.Linear(d_in, d_hidden))
        self.add_module("input_to_mean",  nn.Linear(d_in, d_out))  # proposal mean
        self.add_module("input_to_tau",  nn.Linear(d_in, 1))  # proposal variance

    def get_proposal_params(self, input):
        # hidden_layer = self.input_to_hidden(input)
        # transform_layer = torch.tanh(hidden_layer)
        mean = self.input_to_mean(input)
        var = torch.exp(torch.min(self.input_to_tau(input), torch.tensor(5.)))
        if self.log:
            var = torch.log(1 + var)  # to avoid constrained optimization
        return mean, var

    def sample(self, input, no_samples):
        mean, var = self.get_proposal_params(input)  # obtain mean and variance
        return mean + torch.sqrt(var) * torch.randn(no_samples,
                                                    self.d_out)  # return samples from N(mu(theta), var(theta)I)

    def forward(self, input):
        return self.get_proposal_params(input)


class ScaledLinearProposal(Proposal):
    def __init__(self, d_in, d_out, log=True, bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.log = log

        self.add_module("dynamics", nn.Linear(d_out, d_out, bias=bias))
        self.register_parameter("tau_t", Parameter(torch.zeros(d_out)))
        self.register_parameter("beta_t", Parameter(torch.zeros(d_out)))
        self.register_parameter("mu_t", Parameter(torch.zeros(d_out)))

    def get_proposal_params(self, input):
        "x_t ~ N(mu_t + diag(beta_t)*A*x_{t-1}, diag(sigma_t))"
        mean = self.mu_t + self.beta_t * self.dynamics(input[:, :self.d_out])  # don't use observations for proposal
        var = torch.exp(self.tau_t)  # to avoid constrained optimization
        if self.log:
            var = torch.log(1 + var)  # to avoid constrained optimization
        return mean, var
        # return mean, var * torch.ones(mean.shape[0], 1)

    def sample(self, input, no_samples):
        mean, var = self.get_proposal_params(input)  # obtain mean and variance
        return mean + torch.sqrt(var) * torch.randn(no_samples,
                                                    self.d_out)  # return samples from N(mu(theta), diag(var(theta)))

    def forward(self, input):
        return self.get_proposal_params(input)


class BootstrapProposal(Proposal):
    """Proposal used to run a bootstrap particle filter"""
    def __init__(self, d_in, d_out, d_obs, var, dynamics):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_obs = d_obs  # used to remove observation appended to the input
        self.var = var * torch.ones(1)
        # self.register_parameter("var", Parameter(var * torch.ones(1).squeeze()))
        self.dynamics = dynamics  # function handle of the true dynamics

    def get_proposal_params(self, input):
        mean = self.dynamics(input[:, :-self.d_obs])
        var = self.var
        return mean, var

    def sample(self, input, no_samples):
        mean, var = self.get_proposal_params(input)  # obtain mean and variance
        return mean + torch.sqrt(var) * torch.randn(no_samples,
                                                    self.d_out)  # return samples from N(mu(theta), var(theta)I)

    def forward(self, input):
        return self.get_proposal_params(input)


class AffineBootstrapProposal(Proposal):
    """Proposal mean is an affine transformation of the dynamics"""
    def __init__(self, d_in, d_out, d_obs, dynamics, log=True, bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_obs = d_obs  # used to remove observation appended to the input
        self.dynamics = dynamics  # function handle of the true dynamics
        self.log = log

        self.add_module("A", nn.Linear(d_out, d_out, bias=bias))
        self.register_parameter("tau_t", Parameter(torch.zeros(1).squeeze()))
        self.register_parameter("beta_t", Parameter(torch.zeros(d_out)))
        self.register_parameter("mu_t", Parameter(torch.zeros(d_out)))

    def get_proposal_params(self, input):
        mean = self.mu_t + self.beta_t * self.A(self.dynamics(input[:, :-self.d_obs]))  # don't use observations for proposal
        var = torch.exp(self.tau_t)  # to avoid constrained optimization
        if self.log:
            var = torch.log(1 + var)  # to avoid constrained optimization
        return mean, var * torch.ones(mean.shape[0], 1)

    def sample(self, input, no_samples):
        mean, var = self.get_proposal_params(input)  # obtain mean and variance
        return mean + torch.sqrt(var) * torch.randn(no_samples,
                                                    self.d_out)  # return samples from N(mu(theta), var(theta)I)

    def forward(self, input):
        return self.get_proposal_params(input)


class DiffusingInducingProposal(Proposal):
    def __init__(self, sgp, diffusion):
        self.sgp = sgp
        self.diffusion = diffusion

    def forward(self, *input):
        return self.sgp.qz.mean, self.sgp.qz.cov + torch.diag(self.diffusion)
