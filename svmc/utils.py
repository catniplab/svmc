import torch
import  numpy as np
import numpy.random as npr
from torch.optim import Adam
from tqdm import tqdm
from .metric import gaussian_loss


def initalize_svmc(likelihood, dynamics, proposal, Y, dx, lr, iterations, beta=1, no_reals=10, project=False, M=1,
                      U=None, batch_size=128):
    """
    Function that will initialize the likelihood and proposal of SVMC by performing variational EM
    :param likelihood: likelihood function
    :param dynamics: dynamics of latent state
    :param proposal: proposal distribution
    :param Y: T by dy tensor that contains the data
    :param dx: dimension of latent space
    :param lr: learning rate of the optimizer
    :param iterations: number of gradient steps to take
    :param beta: from beta-vae, controls the importance of the variational posterior
    :param no_reals: number of realizations to get mean of variational posterior
    :param project: boolean variable that decides whether to do constrained optimization
    :param M: number of samples to use for computing gradient
    :param U: input
    :return: model, loss
    """
    T = Y.shape[0]  # number of points
    n_batches = int(np.ceil(T / batch_size))
    dy = Y.shape[1]  # dimension of observation
    optim = Adam([{'params': likelihood.parameters()},
                  {'params': dynamics.parameters()},
                  {'params': proposal.parameters()}], lr=lr)  # create optimizer
    loss = []
    for j in tqdm(range(iterations)):
        idx = 0
        for batch in range(n_batches):
            optim.zero_grad()  # zero out gradient
            elbo = 0
            xprev = torch.randn(M, dx)  # sample standard normal rvs
            for t in range(idx, np.min([idx + batch_size, T])):
                if U is None:
                    cat = torch.cat((xprev, Y[[t], :].repeat(M, 1)), 1)
                else:
                    cat = torch.cat((xprev, U[[t], :].repeat(M, 1), Y.repeat(M, 1)), 1)

                mu, var = proposal(cat)  # get mean and variance from proposal
                xcurr = mu + torch.sqrt(var) * torch.randn(M, dx)  # sample from proposal
                elbo += torch.mean(likelihood(Y[[t], :], xcurr)) / T  # Likelihood
                elbo += torch.mean(dynamics(xcurr, xprev)) / T  # dynamics
                elbo += beta * torch.mean(gaussian_loss(xcurr, mu, torch.log(var))) / T  # proposal
                xprev = xcurr

            idx += batch_size
            elbo = -elbo
            loss.append(elbo.item())
            elbo.backward()  # compute gradient
            optim.step()  # take gradient step

        if project:
            "Project onto your favorite manifold"
            likelihood.project_unit_norm()

    with torch.no_grad():
        proposal.eval()
        "Get MAP from variational approximation"
        X = torch.randn(T + 1, dx, no_reals)
        for real in range(no_reals):
            for t in range(T):
                mu, var = proposal(torch.cat((X[[t], :, real], Y[[t], :]), 1))  # get mean and variance from proposal
                X[t + 1, :, real] = mu + torch.sqrt(var) * torch.randn(1, dx)  # sample from proposal
        x_map = torch.mean(X, 2)
    return likelihood, dynamics, proposal, loss, x_map


def surprise(x_star, weights, sgp):
    """
    Computes entropy of a point to see if it will pass the test
    :param x_star:  test point
    :param weights: M normalized importance weights
    :param sgp: M sgp, one for each stream
    :return:  entropy
    """
    M = weights.size
    entropy = 0
    for m in range(M):
        entropy += weights[m] * sgp[m].log_weight(x_star[[m], :])

    return entropy

def update_stats(x, loc, var, N):
    if N == 0:
        loc = x
        var = torch.ones(x.shape[-1])
    else:
        if N == 1:
            var.fill_(0.)
        loc = (N * loc + x) / (N + 1)
        var = (N * var + (x - loc) ** 2) / (N + 1)
    idx = var == 0
    var[idx] = 1
    N += 1
    return loc, var, N
