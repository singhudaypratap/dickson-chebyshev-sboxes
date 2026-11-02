"""Experiment 5: Dickson permutations D_n(x,1) of F_{2^m}, m = 3..10, by the functional equation.

Evaluator: F_{q^2} = F_{2^{2m}} built from a primitive polynomial found by search; F_q is its subfield.
For x in F_q pick u with u + u^{-1} = x (u in F_q^* or u^{q+1} = 1); then D_n(x) = u^n + u^{-n}.
S-box coordinates: an F_2-basis of the subfield (any F_2-linear coordinate map is a linear equivalence,
so DU, NL and degree do not depend on that choice).
Cross-checks: (i) recurrence D_n = x D_{n-1} + D_{n-2} in F_{q^2} arithmetic for n <= 40;
(ii) for m <= 7 the permutation criterion and the coincidence rule D_a = D_b iff a in {+-b, +-q b}
exhaustively; (iii) for m = 8 the class-level (DU, NL, deg) multiset equals Experiment 1.
"""
import json, sys, time
from math import gcd
from collections import Counter
from multiprocessing import Pool
import numpy as np

M_LIST = [int(v) for v in sys.argv[1:]] or [3, 4, 5, 6, 7, 8, 9, 10]


def find_primitive(M):
    order = (1 << M) - 1
    for poly in range(1 << M | 1, 1 << (M + 1), 2):
        x, k = 1, 0
        ok = True
        while True:
            x <<= 1
            if x >> M & 1:
                x ^= poly
            k += 1
            if x == 1:
                ok = (k == order)
                break
            if k > order:
                ok = False
                break
        if ok:
            return poly
    raise RuntimeError("no primitive polynomial")


def build_field(M):
    poly = find_primitive(M)
    N = (1 << M) - 1
    exp = np.zeros(2 * N, dtype=np.int64)
    x = 1
    for k in range(N):
        exp[k] = x
        x <<= 1
        if x >> M & 1:
            x ^= poly
    assert x == 1
    exp[N:] = exp[:N]
    log = np.full(1 << M, -1, dtype=np.int64)
    log[exp[:N]] = np.arange(N)
    assert np.all(log[1:] >= 0)
    return poly, exp, log


def gf_mul(a, b, exp, log, N):
    a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
    out = np.zeros(np.broadcast(a, b).shape, dtype=np.int64)
    a, b = np.broadcast_arrays(a, b)
    nz = (a != 0) & (b != 0)
    out[nz] = exp[(log[a[nz]] + log[b[nz]]) % N]
    return out


def metrics(S, m):
    q = 1 << m
    X = np.arange(q, dtype=np.int64)
    pop = np.array([bin(i).count("1") for i in range(q)], dtype=np.int64)
    a = np.arange(1, q, dtype=np.int64)[:, None]
    out = S[X[None, :] ^ a] ^ S[X[None, :]]
    idx = (np.arange(q - 1)[:, None] * q + out).ravel()
    ddt = np.bincount(idx, minlength=(q - 1) * q)
    du = int(ddt.max())
    H = (1 - 2 * (pop[X[:, None] & X[None, :]] & 1)).astype(np.float32)
    F = (1 - 2 * (pop[a & S[None, :]] & 1)).astype(np.float32)
    W = F @ H
    L = int(np.abs(W).max())
    assert np.all(np.abs(W - np.rint(W)) == 0)
    coords = ((S[None, :] >> np.arange(m)[:, None]) & 1).copy()
    for i in range(m):
        bit = 1 << i
        hi = X[(X & bit) != 0]
        coords[:, hi] ^= coords[:, hi ^ bit]
    deg = max(int(pop[X[coords[j] == 1]].max()) for j in range(m) if coords[j].any())
    diff_spec = Counter(ddt.tolist()); walsh_spec = Counter(np.abs(W).astype(np.int64).ravel().tolist())
    return du, q // 2 - L // 2, deg, diff_spec, walsh_spec


CTX = {}


def setup(m):
    M = 2 * m; q = 1 << m; N2 = (1 << M) - 1
    poly, exp, log = build_field(M)
    sub = np.concatenate([[0], exp[np.arange(q - 1) * (q + 1)]])          # subfield F_q inside F_{q^2}
    assert np.unique(sub).size == q
    basis = []; span = {0}
    for e in sub[1:]:
        if int(e) not in span:
            basis.append(int(e)); span = span | {s ^ int(e) for s in span}
        if len(basis) == m:
            break
    assert len(basis) == m
    elem_of = np.zeros(q, dtype=np.int64)
    for c in range(q):
        v = 0
        for j in range(m):
            if c >> j & 1:
                v ^= basis[j]
        elem_of[c] = v
    coord_of = np.full(1 << M, -1, dtype=np.int64); coord_of[elem_of] = np.arange(q)
    assert set(elem_of.tolist()) == set(sub.tolist())
    logu = np.full(q, -1, dtype=np.int64)
    cand = np.concatenate([np.arange(q - 1) * (q + 1), np.arange(q + 1) * (q - 1)])   # F_q^* and mu_{q+1}
    for L in cand:
        u = exp[L % N2]; ui = exp[(-L) % N2]
        c = coord_of[u ^ ui]
        assert c >= 0
        if logu[c] < 0:
            logu[c] = L % N2
    assert np.all(logu >= 0)
    split = np.zeros(q, dtype=bool)                                          # u in F_q^*
    for L in np.arange(q - 1) * (q + 1):
        split[coord_of[exp[L] ^ exp[(-L) % N2]]] = True
    return dict(m=m, q=q, N2=N2, poly=poly, exp=exp, log=log, elem_of=elem_of, coord_of=coord_of, logu=logu, split=split)


def dickson(ctx, n):
    e = ctx["exp"]; N2 = ctx["N2"]; lu = ctx["logu"]
    vals = e[(n * lu) % N2] ^ e[(-n * lu) % N2]
    c = ctx["coord_of"][vals]
    assert np.all(c >= 0)
    return c


def init_worker(m):
    CTX.update(setup(m))


def work(n):
    S = dickson(CTX, n)
    du, nl, deg, ds, ws = metrics(S, CTX["m"])
    return n, du, nl, deg, int(np.sum(S == np.arange(CTX["q"]))), sorted(ds.items()), sorted(ws.items())


def run_m(m):
    t0 = time.time()
    ctx = setup(m); q, N2 = ctx["q"], ctx["N2"]
    # cross-check (i): recurrence in F_{q^2}
    x_el = ctx["elem_of"]; d0 = np.zeros(q, dtype=np.int64); d1 = x_el.copy()
    for n in range(2, 41):
        d0, d1 = d1, gf_mul(x_el, d1, ctx["exp"], ctx["log"], N2) ^ d0
        assert np.array_equal(ctx["coord_of"][d1], dickson(ctx, n)), (m, n)
    units = np.array([n for n in range(1, N2) if gcd(n, N2) == 1], dtype=np.int64)
    res = {"m": m, "q": q, "q2_minus_1": N2, "primitive_poly_F_q2": hex(ctx["poly"]), "num_units": int(units.size)}
    if m <= 7:   # cross-check (ii)
        tabs = {}
        for n in range(1, N2):
            S = dickson(ctx, n)
            assert (np.unique(S).size == q) == (gcd(n, N2) == 1), (m, n)
            if gcd(n, N2) == 1:
                tabs.setdefault(S.tobytes(), []).append(n)
        for g in tabs.values():
            n = g[0]
            assert set(g) == {n % N2, (-n) % N2, (q * n) % N2, (-q * n) % N2}
        res["distinct_functions"] = len(tabs)
    seen = np.zeros(N2, dtype=bool); reps = []
    for n in units:
        if seen[n]:
            continue
        reps.append(int(n))
        for k in range(2 * m):
            for s in (1, -1):
                seen[(s * pow(2, k, N2) * int(n)) % N2] = True
    res["num_classes"] = len(reps)
    pm1 = {(s * pow(2, i, q - 1)) % (q - 1) for i in range(m) for s in (1, -1)}
    pp1 = {(s * pow(2, i, q + 1)) % (q + 1) for i in range(2 * m) for s in (1, -1)}
    with Pool(2, initializer=init_worker, initargs=(m,)) as pool:
        rows = pool.map(work, reps, chunksize=16)
    weak = np.array([(n % (q - 1)) in pm1 or (n % (q + 1)) in pp1 for n, *_ in rows])
    du = np.array([r[1] for r in rows]); nl = np.array([r[2] for r in rows]); dg = np.array([r[3] for r in rows])
    # class sizes (orbit size) to weight by units
    osz = np.array([len({(s * pow(2, k, N2) * n) % N2 for k in range(2 * m) for s in (1, -1)}) for n in reps])
    res.update({
        "class_DU_dist": dict(sorted(Counter(du.tolist()).items())), "class_NL_dist": dict(sorted(Counter(nl.tolist()).items())),
        "class_deg_dist": dict(sorted(Counter(dg.tolist()).items())),
        "best_DU": int(du.min()), "best_DU_reps": [(int(r[0]), r[1], r[2], r[3]) for r in rows if r[1] == du.min()][:12],
        "max_NL": int(nl.max()), "max_NL_reps": [(int(r[0]), r[1], r[2], r[3]) for r in rows if r[2] == nl.max()][:12],
        "weak_units": int(osz[weak].sum()), "weak_fraction_units": float(osz[weak].sum() / osz.sum()),
        "weak_min_DU": int(du[weak].min()), "weak_max_NL": int(nl[weak].max()),
        "nonweak_max_DU": int(du[~weak].max()) if (~weak).any() else None,
        "nonweak_min_NL": int(nl[~weak].min()) if (~weak).any() else None,
        "bound_q_over_4": q // 4,
        "separated_by_DU": bool((~weak).any() and du[~weak].max() < du[weak].min()),
        "separated_by_NL": bool((~weak).any() and nl[~weak].min() > nl[weak].max()),
        "orbit_size_sum_equals_units": bool(osz.sum() == units.size),
    })
    assert res["orbit_size_sum_equals_units"]
    assert res["weak_min_DU"] >= q // 4 and res["weak_max_NL"] <= q // 4, "Theorem 5.2 violated"
    # class invariance spot check
    rng = np.random.default_rng(1000 + m)
    for i in rng.choice(len(reps), min(40, len(reps)), replace=False):
        n = reps[i]
        for k, s in ((int(rng.integers(0, 2 * m)), -1), (1, 1)):
            S2 = dickson(ctx, (s * pow(2, k, N2) * n) % N2)
            assert metrics(S2, m)[:3] == tuple(rows[i][1:4])
    res["runtime_s"] = round(time.time() - t0, 1)
    with open(f"results/exp5_m{m}_classes.csv", "w") as f:
        f.write("rep_n,orbit_size,weak,DU,NL,deg,fixed_points\n")
        for r, o, w in zip(rows, osz, weak):
            f.write(f"{r[0]},{o},{int(w)},{r[1]},{r[2]},{r[3]},{r[4]}\n")
    if m == 8:
        e1 = np.genfromtxt("results/exp1_classes.csv", delimiter=",", names=True, dtype=np.int64)
        a = Counter(zip(e1["DU"].tolist(), e1["NL"].tolist(), e1["deg"].tolist()))
        b = Counter(zip(du.tolist(), nl.tolist(), dg.tolist()))
        assert a == b, "m=8 mismatch with Experiment 1"
        res["m8_matches_exp1"] = True
        json.dump({str(r[0]): {"diff_spec": r[5], "walsh_spec": r[6]} for r in rows}, open("results/exp5_m8_spectra.json", "w"))
    return res


if __name__ == "__main__":
    for m in M_LIST:
        r = run_m(m)
        json.dump(r, open(f"results/exp5_m{m}_summary.json", "w"), indent=1)
        print(json.dumps({k: v for k, v in r.items() if not k.startswith("class_")}), flush=True)
