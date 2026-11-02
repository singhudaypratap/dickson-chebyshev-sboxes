"""Experiment 11: independent validation of the GPU runs (m = 12, 13) on the CPU, with separate code.

(a) The primitive polynomials of F_{2^{2m}} reported by the GPU job are checked by a different test: Rabin's
    irreducibility test (x^(2^M) = x mod f and gcd(x^(2^(M/r)) - x, f) = 1 for every prime r | M), then
    x^((2^M-1)/p) != 1 for every prime p | 2^M - 1.
(b) The best-delta representatives reported by the GPU are re-profiled in F_{2^m} itself (not F_{q^2}), using a
    different field model (fixed primitive polynomial of degree m, log/antilog tables) and the ladder
    D_{2k} = D_k^2, D_{2k+1} = D_k D_{k+1} + x for D_n(x, 1). delta, NL (fast Walsh-Hadamard) and degree
    (Moebius transform) are computed from scratch.
(c) Theorem 5.2 is spot-checked: random weak indices give delta >= q/4.
Output: results/exp11_gpu_crosscheck.json
"""
import json, sys
import numpy as np

GPU = {12: "results/dickson_m12_summary.json", 13: "results/dickson_m13_summary.json"}


# ---------- (a) polynomial checks over F_2, polynomials as ints ----------
def pmod(a, f):
    df = f.bit_length() - 1
    while a and a.bit_length() - 1 >= df:
        a ^= f << (a.bit_length() - 1 - df)
    return a


def pmulmod(a, b, f):
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1; a <<= 1
        if a >> (f.bit_length() - 1) & 1:
            a ^= f
    return pmod(r, f)


def ppowmod(a, e, f):
    r = 1
    while e:
        if e & 1:
            r = pmulmod(r, a, f)
        a = pmulmod(a, a, f); e >>= 1
    return r


def pgcd(a, b):
    while b:
        a, b = b, pmod(a, b)
    return a


def primes(n):
    out, p = [], 2
    while p * p <= n:
        if n % p == 0:
            out.append(p)
            while n % p == 0:
                n //= p
        p += 1
    if n > 1:
        out.append(n)
    return out


def is_primitive(f):
    M = f.bit_length() - 1
    x = 2
    if ppowmod(x, 1 << M, f) != x:
        return False
    for r in primes(M):
        if pgcd(ppowmod(x, 1 << (M // r), f) ^ x, f) != 1:
            return False
    return all(ppowmod(x, ((1 << M) - 1) // p, f) != 1 for p in primes((1 << M) - 1))


# ---------- (b) profiles in F_{2^m} ----------
FIELD_POLY = {12: 0x1053, 13: 0x201B}     # x^12+x^6+x^4+x+1, x^13+x^4+x^3+x+1 (checked primitive below)


class Field:
    def __init__(self, m):
        self.m, self.q = m, 1 << m
        f = FIELD_POLY[m]; assert is_primitive(f)
        exp = np.zeros(2 * self.q, dtype=np.int64); log = np.zeros(self.q, dtype=np.int64); v = 1
        for i in range(self.q - 1):
            exp[i] = v; log[v] = i
            v <<= 1
            if v & self.q:
                v ^= f
        exp[self.q - 1:2 * self.q - 2] = exp[:self.q - 1]
        self.exp, self.log = exp, log

    def mul(self, a, b):
        r = self.exp[self.log[a] + self.log[b]]
        return np.where((a == 0) | (b == 0), 0, r)


def dickson(F, n):
    """D_n(x, 1) for all x, by the ladder on (D_k, D_{k+1})."""
    x = np.arange(F.q, dtype=np.int64)
    a, b = np.zeros_like(x), x.copy()            # D_0 = 0 (char 2), D_1 = x
    for bit in bin(n)[2:]:
        if bit == "0":
            a, b = F.mul(a, a), F.mul(a, b) ^ x
        else:
            a, b = F.mul(a, b) ^ x, F.mul(b, b)
    return a


def delta(S):
    q = S.size; x = np.arange(q); best = 0
    for a in range(1, q):
        best = max(best, int(np.bincount(S[x ^ a] ^ S, minlength=q).max()))
    return best


def nonlinearity(S):
    q = S.size; m = q.bit_length() - 1; maxw = 0
    x = np.arange(q)
    for start in range(1, q, 512):
        bs = np.arange(start, min(start + 512, q))
        par = np.bitwise_count(bs[:, None] & S[None, :]) & 1 if hasattr(np, "bitwise_count") else \
            np.vectorize(lambda v: bin(v).count("1") & 1)(bs[:, None] & S[None, :])
        W = (1 - 2 * par.astype(np.int32))      # cast first: uint8 arithmetic would wrap
        h = 1
        while h < q:
            W = W.reshape(len(bs), -1, 2, h)
            W = np.concatenate([W[:, :, 0, :] + W[:, :, 1, :], W[:, :, 0, :] - W[:, :, 1, :]], axis=2)
            W = W.reshape(len(bs), q); h *= 2
        maxw = max(maxw, int(np.abs(W).max()))
    return q // 2 - maxw // 2


def degree(S):
    q = S.size; m = q.bit_length() - 1; deg = 0
    wt = np.array([bin(i).count("1") for i in range(q)])
    for j in range(m):
        f = ((S >> j) & 1).astype(np.uint8)
        h = 1
        while h < q:
            f = f.reshape(-1, 2, h); f[:, 1, :] ^= f[:, 0, :]; f = f.reshape(q); h *= 2
        if f.any():
            deg = max(deg, int(wt[f == 1].max()))
    return deg


def is_weak(n, m):
    q = 1 << m
    A = {(s * pow(2, i, q - 1)) % (q - 1) for i in range(m) for s in (1, -1)}
    B = {(s * pow(2, i, q + 1)) % (q + 1) for i in range(2 * m) for s in (1, -1)}
    return n % (q - 1) in A or n % (q + 1) in B


if __name__ == "__main__":
    from math import gcd
    out = {"field_polys_F_2m": {str(k): hex(v) for k, v in FIELD_POLY.items()}, "polys": {}, "profiles": [], "weak_spot": []}
    for m, path in GPU.items():
        g = json.load(open(path))
        f = int(g["primitive_poly_F_q2"], 16)
        out["polys"][str(m)] = {"poly": hex(f), "degree": f.bit_length() - 1, "primitive": is_primitive(f)}
        F = Field(m); Q = (1 << (2 * m)) - 1
        assert sorted(dickson(F, 7).tolist()) != list(range(F.q)) or gcd(7, Q) == 1
        for n, d_gpu, nl_gpu, deg_gpu in g["best_DU_reps"]:
            S = dickson(F, n)
            assert np.unique(S).size == F.q, "not a permutation"
            r = {"m": m, "n": n, "gpu": [d_gpu, nl_gpu, deg_gpu],
                 "cpu": [delta(S), nonlinearity(S), degree(S)]}
            r["match"] = r["gpu"] == r["cpu"]; out["profiles"].append(r); print(r, flush=True)
        rng = np.random.default_rng(20261031 + 110 + m); found = 0
        while found < 3:
            n = int(rng.integers(1, Q))
            if gcd(n, Q) == 1 and is_weak(n, m):
                d = delta(dickson(F, n)); found += 1
                out["weak_spot"].append({"m": m, "n": n, "delta": d, "ge_q_over_4": d >= F.q // 4})
                print(out["weak_spot"][-1], flush=True)
    json.dump(out, open("results/exp11_gpu_crosscheck.json", "w"), indent=1)
    print(json.dumps(out["polys"]))
