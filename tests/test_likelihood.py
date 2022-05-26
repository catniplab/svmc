import pytest
import torch


@pytest.mark.parametrize("d_obs, d_latent", [(2, 2)])
def test_isotropic_gaussian_likelihood(d_obs, d_latent):
    from svsmc.likelihood import ISOGaussian
    q = ISOGaussian(d_obs, d_latent)
    x = torch.rand(1, d_latent)
    y = 4 * x  # observation of latent states
    weight = q.compute_log_weight(y, x)
