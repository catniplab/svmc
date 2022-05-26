import torch

from svmc import metric


def test_gaussian_loss():
    from torch.distributions import Normal
    N = 5
    xdim = 2

    x = torch.randn(N, xdim)
    mu = torch.randn(1, xdim)
    logvar = torch.randn(1)

    loss1 = metric.gaussian_loss(x, mu, logvar)
    loss2 = -Normal(mu, torch.exp(.5 * logvar)).log_prob(x).sum(-1)

    print(loss1.shape, loss2.shape)
    assert torch.allclose(loss1, loss2)
