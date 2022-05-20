import torch


def chol_solve(A, B):
    L = A.cholesky(upper=False)  # A = LL'
    # X = B.potrs(L, upper=False)  # gradient not implemented
    # A^{-1} B = (LL')^{-1} B = (L')^{-1} L^{-1} B = (L')^{-1} (L^{-1} B)
    # C = B.trtrs(L, upper=False, transpose=False)[0]
    # X = C.trtrs(L, upper=False, transpose=True)[0]
    X = torch.cholesky_solve(B, L)
    return X


def qr_solve(A, B):
    # QRx = b => x = R^{-1} Q'b
    Q, R = torch.qr(A)
    X = (Q.t() @ B).triangular_solve(R)[0]
    return X


def lu_solve(A, B):
    X, _ = torch.solve(B, A)
    return X


def sqrt(x, eps=1e-12):
    """Safe-gradient square root"""
    return torch.sqrt(x + eps)


def squared_scaled_dist(a, b, gamma):
    if a.shape[1] != b.shape[1]:
        raise ValueError("Inconsistent dimensions")

    a = a * gamma
    b = b * gamma
    b = b.t()  # final outcome in shape of x row * z col
    a2 = torch.sum(a ** 2, dim=1, keepdim=True)
    b2 = torch.sum(b ** 2, dim=0, keepdim=True)
    ab = a.mm(b)
    d2 = a2 - 2 * ab + b2
    return torch.clamp(d2, min=0)


def scaled_dist(a, b, gamma):
    return sqrt(squared_scaled_dist(a, b, gamma))


def kron(a, b):
    """Kronecker product"""
    ra, ca = a.shape
    rb, cb = b.shape
    return torch.reshape(
        a.reshape(ra, 1, ca, 1) * b.reshape(1, rb, 1, cb), (ra * rb, ca * cb)
    )


def yakron(a, b):
    """Batch Kronecker product"""
    assert a.dim() > 1 and b.dim() > 1
    if a.dim() == 2:
        a = a.unsqueeze(0)
    if b.dim() == 2:
        b = b.unsqueeze(0)

    ba, ra, ca = a.shape
    bb, rb, cb = b.shape

    assert ba == 1 or bb == 1 or ba == bb

    bc = max(ba, bb)

    return torch.reshape(
        a.reshape(ba, ra, 1, ca, 1) * b.reshape(bb, 1, rb, 1, cb), (bc, ra * rb, ca * cb)
    )
