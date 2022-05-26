import numpy as np
import pytest
import torch


@pytest.mark.parametrize("ra, ca, rb, cb", [(3, 5, 2, 4)])
def test_kron(ra, ca, rb, cb):
    """
    :param ra: the number of rows of left matrix
    :param ca: the number of columns of left matrix
    :param rb: the number of rows of right matrix
    :param cb: the number of columns of right matrix
    :return:
    """
    from svmc.operation import kron, yakron
    a = torch.randn(ra, ca)
    b = torch.randn(rb, cb)
    kron_out = kron(a, b)

    assert kron_out.shape == (ra * rb, ca * cb)

    b.unsqueeze_(0)
    yakron_out = yakron(a, b)
    assert yakron_out.shape == (1, ra * rb, ca * cb)

    yakron_out.squeeze_(0)

    assert torch.all(kron_out == yakron_out)
