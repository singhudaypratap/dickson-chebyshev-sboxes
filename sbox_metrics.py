"""Cryptographic invariants of an 8-bit S-box S: F_2^8 -> F_2^8.

Definitions (standard, see Nyberg 1993; Carlet 2021):
  DDT[a,b] = #{x : S(x) ^ S(x ^ a) = b},  differential uniformity DU = max_{a != 0, b} DDT[a,b]
  W(a,b)   = sum_x (-1)^{b.S(x) + a.x},    linearity L = max_{a, b != 0} |W(a,b)|
  nonlinearity NL = 2^{n-1} - L/2 = 128 - L/2
  algebraic degree = max over the 8 coordinate functions of the ANF degree
All three are invariant under affine equivalence (DU, NL) and under linear equivalence (all three).
"""
import numpy as np

N = 8
SIZE = 256
_X = np.arange(SIZE, dtype=np.int64)
_POP = np.array([bin(i).count("1") for i in range(SIZE)], dtype=np.int64)
# Sylvester-Hadamard matrix H[a,x] = (-1)^{popcount(a & x)}
_H = (1 - 2 * (_POP[_X[:, None] & _X[None, :]] & 1)).astype(np.float64)


def is_permutation(S) -> bool:
    S = np.asarray(S)
    assert S.shape == (SIZE,)
    return np.unique(S).size == SIZE


def ddt_rows(S):
    """DDT restricted to a = 1..255; shape (255, 256)."""
    S = np.asarray(S, dtype=np.int64)
    a = np.arange(1, SIZE, dtype=np.int64)[:, None]
    out = S[_X[None, :] ^ a] ^ S[_X[None, :]]
    idx = (np.arange(SIZE - 1)[:, None] * SIZE + out).ravel()
    return np.bincount(idx, minlength=(SIZE - 1) * SIZE).reshape(SIZE - 1, SIZE)


def differential_uniformity(S) -> int:
    return int(ddt_rows(S).max())


def walsh_nonzero_b(S):
    """W(a,b) for b = 1..255 (rows) and all a (cols)."""
    S = np.asarray(S, dtype=np.int64)
    b = np.arange(1, SIZE, dtype=np.int64)[:, None]
    F = (1 - 2 * (_POP[b & S[None, :]] & 1)).astype(np.float64)   # (-1)^{b.S(x)}, shape (255, 256)
    W = F @ _H                                                     # sum_x (-1)^{b.S(x)} (-1)^{a.x}
    Wi = np.rint(W).astype(np.int64)
    assert np.all(np.abs(W - Wi) < 1e-9)
    return Wi


def nonlinearity(S) -> int:
    L = int(np.abs(walsh_nonzero_b(S)).max())
    assert L % 2 == 0
    return 128 - L // 2


def algebraic_degree(S) -> int:
    S = np.asarray(S, dtype=np.int64)
    coords = ((S[None, :] >> np.arange(N)[:, None]) & 1).copy()   # (8, 256)
    for i in range(N):                                             # binary Moebius transform
        bit = 1 << i
        lo = (_X & bit) == 0
        coords[:, _X[~lo]] ^= coords[:, _X[~lo] ^ bit]
    deg = 0
    for j in range(N):
        support = _X[coords[j] == 1]
        if support.size:
            deg = max(deg, int(_POP[support].max()))
    return deg


def profile(S) -> dict:
    return {
        "DU": differential_uniformity(S),
        "NL": nonlinearity(S),
        "deg": algebraic_degree(S),
        "fixed_points": int(np.sum(np.asarray(S) == _X)),
    }
