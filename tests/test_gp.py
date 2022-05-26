import pytest


@pytest.mark.parametrize("n, m, xdim, ydim", [(10, 5, 3, 2)])
def test_sgp(n, m, xdim, ydim):
    import numpy as np
    from svmc.gp import SGP
    from svmc.covfun import SquaredExponential

    A = np.random.randn(xdim, ydim)
    x = np.random.randn(n, xdim)
    y = x @ A
    inducing = np.random.randn(m, xdim)
    covfun = SquaredExponential(1.0, 0.1)
    sgp = SGP(ydim, xdim, inducing, None, covfun, noise=0.1)
    sgp.predict(x)
    sgp.update(x, y)
    sgp.predict(x)


@pytest.mark.parametrize("n, m, xdim, ydim, max_iter, seed", [(10, 5, 3, 2, 500, 0)])
def test_hyper_gradient(n, m, xdim, ydim, max_iter, seed):
    import numpy as np
    import torch
    from torch.optim import Adam

    from svmc.gp import SGP
    from svmc.covfun import SquaredExponential

    np.random.seed(seed)
    torch.manual_seed(seed)

    lengthscale = 0.1

    A = np.random.randn(xdim, ydim)
    x = np.random.randn(n, xdim)
    y = x @ A + np.sqrt(0.1) * np.random.randn(n, ydim)
    
    x = torch.tensor(x)
    y = torch.tensor(y)
    inducing = np.random.randn(m, xdim)

    covfun = SquaredExponential(1.0, lengthscale)
    sgp = SGP(ydim, xdim, inducing, None, covfun, noise=0.1)
    sgp.update(x, y)

    optimizer = Adam(sgp.parameters(), lr=1e-3)

    for i in range(max_iter):
        optimizer.zero_grad()
        yhat = sgp.predict(x)[0]
        loss = torch.sum((y - yhat) ** 2)
        loss.backward()
        optimizer.step()

    optimized_scale = sgp.cov_func.scale.detach().numpy()

    print(lengthscale, optimized_scale)
    assert lengthscale != optimized_scale


def test_qosterior():
    import torch
    from svmc.gp import Qosterior
    mean = torch.zeros(5)
    cov = torch.eye(5)
    q = Qosterior(mean, cov)

    assert torch.allclose(q.mean, mean)
    assert torch.allclose(q.cov, cov)

    new_mean = torch.ones(5)
    new_cov = torch.eye(5) * 2
    q.mean = new_mean
    q.cov = new_cov

    assert torch.allclose(q.mean, new_mean)
    assert torch.allclose(q.cov, new_cov)


@pytest.mark.parametrize("n, m, xdim, ydim", [(10, 4, 3, 2)])
def test_batch_predict(n, m, xdim, ydim):
    """Test batch version of GP prediction
    The input x and inducing variable z are supposed to have the same batch size
    such that each sample (particle) is accompanied with m inducing variables
    """
    import numpy as np
    import torch
    from torch.linalg import solve
    from svmc.gp import SGP, predict
    from svmc.covfun import SquaredExponential
    from svmc.operation import kron

    bx = torch.zeros(n, 1, xdim).normal_()  # (batch, dim)

    covfun = SquaredExponential(1.0, 0.1)
    inducing = np.random.randn(m, xdim)
    sgp = SGP(ydim, xdim, inducing, None, covfun, noise=0.1)
    z = sgp.qz.mean
    bz = z.repeat(n, 1, 1)
    assert bz.shape == (n, m*ydim, 1)

    # Sequentially
    sf = []
    for x in bx:
        Kfz = sgp.cov_func(x, sgp.inducing)
        Kzz = sgp.cov_func(sgp.inducing)
        A = solve(Kzz, Kfz.t())
        A = A.t()
        A = kron(sgp.K, A)

        fmean = A @ z
        fmean = torch.reshape(
            fmean, (ydim, -1)
        ).t()  # (ydim * n,) => (ydim, n) => (n, ydim)
        fmean = fmean + sgp.mean_func(x)  # mean function
        sf.append(fmean)
    sf = torch.stack(sf)
    assert sf.shape == (n, 1, ydim)

    # Batch
    x = bx.squeeze(1)

    Kfz = sgp.cov_func(x, sgp.inducing)  # suppose to use the same pseudo inputs

    A = solve(Kzz, Kfz.t())
    A = A.t()
    A = kron(sgp.K, A)

    A = A.reshape(n, -1, A.shape[-1])
    assert A.shape == (n, ydim, m*ydim)
    bf = A @ bz
    bf.transpose_(2, 1)
    assert bf.shape == (n, 1, ydim)
    assert torch.allclose(sf, bf)

    bf, _ = predict(x, z, sgp.inducing, sgp.K, sgp.cov_func)
    assert bf.shape == (n, ydim)
    assert torch.allclose(sf, bf)
