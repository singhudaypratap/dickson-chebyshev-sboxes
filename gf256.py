"""Arithmetic in F_{2^8} = F_2[x]/(x^8+x^4+x^3+x+1), vectorised with numpy.

All elements are uint8 integers whose bits are polynomial coefficients.
Tables are built once and validated against a bit-serial reference multiply.
"""
import numpy as np

MODULUS = 0x11B
GENERATOR = 3
Q = 256


def slow_mul(a: int, b: int) -> int:
    """Bit-serial (Russian peasant) multiplication, used only to validate tables."""
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a & 0x100:
            a ^= MODULUS
    return r


def _build_tables():
    exp = np.zeros(512, dtype=np.int64)
    log = np.full(256, -1, dtype=np.int64)
    x = 1
    for i in range(255):
        exp[i] = x
        assert log[x] == -1, "generator does not have order 255"
        log[x] = i
        x = slow_mul(x, GENERATOR)
    assert x == 1, "generator order is not 255"
    exp[255:510] = exp[0:255]
    return exp, log


EXP, LOG = _build_tables()


def mul(a, b):
    """Elementwise product of uint8-valued arrays (broadcasting allowed)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    a, b = np.broadcast_arrays(a, b)
    out = np.zeros(a.shape, dtype=np.int64)
    nz = (a != 0) & (b != 0)
    out[nz] = EXP[LOG[a[nz]] + LOG[b[nz]]]
    return out


def inv(a):
    a = np.asarray(a, dtype=np.int64)
    assert np.all(a != 0), "inverse of zero requested"
    return EXP[(255 - LOG[a]) % 255]


def power(a, e: int):
    """a^e for integer e >= 0 (0^0 := 1)."""
    a = np.asarray(a, dtype=np.int64)
    out = np.ones(a.shape, dtype=np.int64)
    if e == 0:
        return out
    nz = a != 0
    out[~nz] = 0
    out[nz] = EXP[(LOG[a[nz]] * e) % 255]
    return out


def sqrt(a):
    """Unique square root in characteristic 2: a^(2^7)."""
    return power(a, 128)


def validate():
    rng = np.random.default_rng(1)
    for _ in range(4000):
        a, b = (int(v) for v in rng.integers(0, 256, 2))
        assert int(mul(a, b)) == slow_mul(a, b)
    allx = np.arange(256)
    assert np.all(mul(sqrt(allx), sqrt(allx)) == allx)
    nz = np.arange(1, 256)
    assert np.all(mul(nz, inv(nz)) == 1)
    return True


if __name__ == "__main__":
    print("F256 tables valid:", validate())
