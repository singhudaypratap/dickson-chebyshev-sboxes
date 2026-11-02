"""Experiment 5c: Dickson permutations of F_{2^11}.
DU for every Frobenius/inversion class; NL and degree for every weak class, every best-DU class and
a seeded random sample of 400 non-weak classes (NL via fast Walsh-Hadamard transform).
Checks Theorem 5.2 on all weak classes and the delta-separation conjecture on all classes.
"""
import json, time, sys
from math import gcd
from collections import Counter
from multiprocessing import Pool
import numpy as np
from exp5_general_m import setup, dickson

m = 11
CTX = {}


def init():
    CTX.update(setup(m))


def du_only(n):
    S = dickson(CTX, n); q = CTX["q"]
    X = np.arange(q, dtype=np.int64)
    best = 0
    for lo in range(1, q, 256):                       # chunk rows to bound memory
        a = np.arange(lo, min(lo + 256, q), dtype=np.int64)[:, None]
        out = S[X[None, :] ^ a] ^ S[None, :]
        idx = (np.arange(a.shape[0])[:, None] * q + out).ravel()
        best = max(best, int(np.bincount(idx, minlength=a.shape[0] * q).max()))
    return n, best


def fwht_rows(F):
    F = F.copy(); h = 1; n = F.shape[1]
    while h < n:
        F = F.reshape(F.shape[0], -1, 2, h)
        a = F[:, :, 0, :].copy(); b = F[:, :, 1, :].copy()
        F[:, :, 0, :] = a + b; F[:, :, 1, :] = a - b
        F = F.reshape(F.shape[0], n); h *= 2
    return F


def nl_deg(n):
    S = dickson(CTX, n); q = CTX["q"]
    X = np.arange(q, dtype=np.int64)
    pop = np.array([bin(i).count("1") for i in range(q)], dtype=np.int64)
    L = 0
    for lo in range(1, q, 256):
        b = np.arange(lo, min(lo + 256, q), dtype=np.int64)[:, None]
        F = (1 - 2 * (pop[b & S[None, :]] & 1)).astype(np.int64)
        L = max(L, int(np.abs(fwht_rows(F)).max()))
    coords = ((S[None, :] >> np.arange(m)[:, None]) & 1).copy()
    for i in range(m):
        bit = 1 << i; hi = X[(X & bit) != 0]
        coords[:, hi] ^= coords[:, hi ^ bit]
    deg = max(int(pop[X[coords[j] == 1]].max()) for j in range(m) if coords[j].any())
    return n, q // 2 - L // 2, deg


if __name__ == "__main__":
    t0 = time.time()
    ctx = setup(m); q, N2 = ctx["q"], ctx["N2"]
    # validation on small n: recurrence D_n = x D_{n-1} + D_{n-2} in F_{q^2}
    from exp5_general_m import gf_mul
    x_el = ctx["elem_of"]; d0 = np.zeros(q, dtype=np.int64); d1 = x_el.copy()
    for n in range(2, 41):
        d0, d1 = d1, gf_mul(x_el, d1, ctx["exp"], ctx["log"], N2) ^ d0
        assert np.array_equal(ctx["coord_of"][d1], dickson(ctx, n))
    seen = np.zeros(N2, dtype=bool); reps = []; osz = []
    pw = [pow(2, k, N2) for k in range(2 * m)]
    for n in range(1, N2):
        if seen[n] or gcd(n, N2) != 1:
            continue
        orb = {(s * p * n) % N2 for p in pw for s in (1, -1)}
        for o in orb:
            seen[o] = True
        reps.append(n); osz.append(len(orb))
    osz = np.array(osz)
    n_units = sum(1 for n in range(1, N2) if gcd(n, N2) == 1)
    assert osz.sum() == n_units
    print("classes", len(reps), "units", n_units, round(time.time() - t0), flush=True)
    pm1 = {(s * pow(2, i, q - 1)) % (q - 1) for i in range(m) for s in (1, -1)}
    pp1 = {(s * pow(2, i, q + 1)) % (q + 1) for i in range(2 * m) for s in (1, -1)}
    weak = np.array([(n % (q - 1)) in pm1 or (n % (q + 1)) in pp1 for n in reps])
    with Pool(2, initializer=init) as pool:
        du = dict(pool.imap_unordered(du_only, reps, chunksize=64))
        print("DU done", round(time.time() - t0), flush=True)
        DU = np.array([du[n] for n in reps])
        rng = np.random.default_rng(20261031 + 11)
        nonweak_idx = np.flatnonzero(~weak)
        sample = set(rng.choice(nonweak_idx, 400, replace=False).tolist())
        sample |= set(np.flatnonzero(weak).tolist()) | set(np.flatnonzero(DU == DU.min()).tolist())
        sample |= set(np.flatnonzero(~weak)[np.argsort(-DU[~weak])[:20]].tolist())   # highest-DU non-weak
        nd = dict((n, (nl, dg)) for n, nl, dg in pool.imap_unordered(nl_deg, [reps[i] for i in sorted(sample)], chunksize=8))
    NLw = np.array([nd[reps[i]][0] for i in np.flatnonzero(weak)])
    smp_nw = [i for i in sorted(sample) if not weak[i]]
    NLnw = np.array([nd[reps[i]][0] for i in smp_nw])
    best = [(reps[i], int(DU[i]), *nd[reps[i]]) for i in np.flatnonzero(DU == DU.min())]
    res = {"m": m, "q": q, "num_units": int(n_units), "num_classes": len(reps),
           "weak_units": int(osz[weak].sum()), "weak_fraction_units": float(osz[weak].sum() / n_units),
           "class_DU_dist": {int(k): int(v) for k, v in sorted(Counter(DU.tolist()).items())},
           "best_DU": int(DU.min()), "best_DU_reps_DU_NL_deg": best,
           "weak_min_DU": int(DU[weak].min()), "nonweak_max_DU": int(DU[~weak].max()),
           "separated_by_DU": bool(DU[~weak].max() < DU[weak].min()),
           "weak_max_NL_all_weak": int(NLw.max()), "weak_classes": int(weak.sum()),
           "nonweak_NL_sample_size": len(smp_nw), "nonweak_min_NL_in_sample": int(NLnw.min()),
           "bound_q_over_4": q // 4, "runtime_s": round(time.time() - t0, 1)}
    assert res["weak_min_DU"] >= q // 4 and res["weak_max_NL_all_weak"] <= q // 4
    json.dump(res, open("results/exp5c_m11_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "class_DU_dist"}), flush=True)
