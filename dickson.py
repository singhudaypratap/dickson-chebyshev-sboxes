"""Dickson polynomials D_n(x, a) over F_{2^8}.

Definition (Lidl-Mullen-Turnwald): D_0 = 2, D_1 = x, D_n = x D_{n-1} - a D_{n-2}.
In characteristic 2: D_0 = 0 and D_n = x D_{n-1} + a D_{n-2}.
Explicit form: D_n(x,a) = sum_{i=0}^{floor(n/2)} n/(n-i) * C(n-i, i) * (-a)^i * x^(n-2i).
Three independent evaluators are provided and cross-checked:
  (1) full recurrence table for a = 1, (2) 2x2 matrix powering for any a, (3) explicit formula.
"""
from math import comb
import numpy as np
import gf256 as gf

ALLX = np.arange(256, dtype=np.int64)
PERIOD = 255 * 257  # q^2 - 1 = 65535


def recurrence_table_a1(n_max: int):
    """T[n, x] = D_n(x, 1) for 0 <= n <= n_max."""
    T = np.zeros((n_max + 1, 256), dtype=np.uint8)
    T[0] = 0
    T[1] = ALLX
    prev2 = T[0].astype(np.int64)
    prev1 = T[1].astype(np.int64)
    for n in range(2, n_max + 1):
        cur = gf.mul(ALLX, prev1) ^ prev2          # a = 1
        T[n] = cur
        prev2, prev1 = prev1, cur
    return T


def _mat_mul(A, B):
    """2x2 matrices over F_256, entries are broadcastable arrays; A = (a00,a01,a10,a11)."""
    a00, a01, a10, a11 = A
    b00, b01, b10, b11 = B
    return (gf.mul(a00, b00) ^ gf.mul(a01, b10), gf.mul(a00, b01) ^ gf.mul(a01, b11),
            gf.mul(a10, b00) ^ gf.mul(a11, b10), gf.mul(a10, b01) ^ gf.mul(a11, b11))


def dickson_matpow(n: int, x, a):
    """D_n(x, a) via [D_n, D_{n-1}]^T = M^{n-1} [D_1, D_0]^T, M = [[x, a], [1, 0]] (char 2)."""
    x = np.asarray(x, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    x, a = np.broadcast_arrays(x, a)
    if n == 0:
        return np.zeros(x.shape, dtype=np.int64)
    one = np.ones(x.shape, dtype=np.int64)
    zero = np.zeros(x.shape, dtype=np.int64)
    R = (one, zero, zero, one)
    M = (x, a, one, zero)
    e = n - 1
    while e:
        if e & 1:
            R = _mat_mul(R, M)
        M = _mat_mul(M, M)
        e >>= 1
    # [D_n, D_{n-1}] = R [x, 0]
    return gf.mul(R[0], x)


def dickson_explicit(n: int, x, a: int):
    x = np.asarray(x, dtype=np.int64)
    if n == 0:
        return np.zeros(x.shape, dtype=np.int64)
    out = np.zeros(x.shape, dtype=np.int64)
    for i in range(n // 2 + 1):
        num = n * comb(n - i, i)
        assert num % (n - i) == 0
        c = (num // (n - i)) % 2            # sign (-1)^i irrelevant mod 2
        if c:
            out ^= gf.mul(gf.power(a, i), gf.power(x, n - 2 * i))
    return out
