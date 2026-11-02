"""Experiment 8: invariants that settle the open CCZ pair and the EA classes of Dickson permutations of F_256.

CCZ invariant N_k(F): number of affine k-flats V of F_2^8 on which F is affine. A CCZ map is an affine
bijection of F_2^16 sending the graph of F onto the graph of G; it sends each k-flat inside graph(F) to a
k-flat inside graph(G), and k-flats inside a graph correspond one-to-one to input flats on which F is affine.
EA invariant C(F): multiset over b != 0 of (min(deg(b.F), 1 if affine), #{a != 0 : D_a(b.F) constant}).
For F' = A1 o F o A2 + A3 the component b.F' equals (A1^T b).F o A2 plus an affine function, which keeps
the degree (when >= 2) and the number of linear structures.
Sanity checks: invariance under random affine equivalence and under inversion (CCZ) for N_k, and under random
EA equivalence (with a random linear A3) for C.
"""
import json, itertools, time
from collections import Counter
import numpy as np

T = np.load("results/dickson_table_a1.npy").astype(np.int64)
P = 65535
rng = np.random.default_rng(20261031 + 8)
X = np.arange(256, dtype=np.int64)
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def subspaces(k):
    """All k-dim subspaces of F_2^8 as arrays of their 2^k elements, ordered by coefficient vector."""
    out = []
    for pivots in itertools.combinations(range(8), k):
        free = [[c for c in range(8) if c > p and c not in pivots] for p in pivots]
        nfree = sum(len(f) for f in free)
        for bits in range(1 << nfree):
            rows, t = [], 0
            for p, f in zip(pivots, free):
                r = 1 << p
                for c in f:
                    if bits >> t & 1:
                        r |= 1 << c
                    t += 1
                rows.append(r)
            el = np.zeros(1 << k, dtype=np.int64)
            for s in range(1, 1 << k):
                low = (s & -s).bit_length() - 1
                el[s] = el[s & (s - 1)] ^ rows[low]
            out.append(el)
    return np.array(out)


SUB = {k: subspaces(k) for k in (2, 3, 4, 5)}
GAUSS = {2: 10795, 3: 97155, 4: 200787, 5: 97155}
for k in SUB:
    assert len(SUB[k]) == GAUSS[k], (k, len(SUB[k]))


def flat_count(F, k):
    E = SUB[k]; K = 1 << k
    basis_idx = [1 << i for i in range(k)]
    total = 0
    for s in range(0, len(E), 4000):
        Eb = E[s:s + 4000]                                    # (B, K)
        G = F[X[None, :, None] ^ Eb[:, None, :]] ^ F[X][None, :, None]   # (B, 256, K)
        pred = np.zeros_like(G)
        for u in range(1, K):
            low = (u & -u).bit_length() - 1
            pred[:, :, u] = pred[:, :, u & (u - 1)] ^ G[:, :, 1 << low]
        ok = np.all(G == pred, axis=2)                        # F affine on x0 + U
        total += int(ok.sum())
    assert total % K == 0
    return total // K                                         # each coset counted once per element


def anf_deg(f):
    a = f.copy()
    for i in range(8):
        hi = X[(X >> i) & 1 == 1]
        a[hi] ^= a[hi ^ (1 << i)]
    s = X[a == 1]
    return int(POP[s].max()) if s.size else 0


def ea_inv(F):
    out = Counter()
    for b in range(1, 256):
        f = POP[b & F] & 1
        deg = max(1, anf_deg(f)) if anf_deg(f) >= 2 else 1
        ls = sum(1 for a in range(1, 256) if np.unique(f ^ f[X ^ a]).size == 1)
        out[(deg, ls)] += 1
    return tuple(sorted(out.items()))


def rand_lin():
    while True:
        cols = [int(v) for v in rng.integers(1, 256, 8)]
        span = {0}
        for c in cols:
            span |= {s ^ c for s in span}
        if len(span) == 256:
            return np.array([np.bitwise_xor.reduce([cols[j] for j in range(8) if x >> j & 1] or [0]) for x in range(256)])


def inverse(F):
    G = np.zeros(256, dtype=np.int64); G[F] = X; return G


t0 = time.time()
res = {"N_k": {}, "EA": {}, "checks": {}}
open_ccz = [253, 1783]
for n in [253, 259, 1783, 1801]:
    res["N_k"][n] = {k: flat_count(T[n], k) for k in (2, 3, 4, 5)}
    print(n, res["N_k"][n], round(time.time() - t0), flush=True)
# invariance checks
F = T[253]; A, B = rand_lin(), rand_lin(); c1, c2 = int(rng.integers(256)), int(rng.integers(256))
Faff = B[F[A ^ c1]] ^ c2
res["checks"]["N_k_affine_image_of_253"] = {k: flat_count(Faff, k) for k in (2, 3, 4, 5)}
res["checks"]["N_k_inverse_of_253"] = {k: flat_count(inverse(T[253]), k) for k in (2, 3, 4, 5)}
assert res["checks"]["N_k_affine_image_of_253"] == res["N_k"][253]
assert res["checks"]["N_k_inverse_of_253"] == res["N_k"][253]
res["ccz_pair_separated"] = res["N_k"][253] != res["N_k"][1783]
# EA invariants for the inverse pairs that share all affine invariants
for n in [253, 259, 763, 773, 1783, 1801]:
    res["EA"][n] = ea_inv(T[n])
A, B = rand_lin(), rand_lin(); L3 = rand_lin() if True else None
M3 = np.array([np.bitwise_xor.reduce([int(v) for v in rng.integers(0, 256, 8)][j:j + 1] or [0]) for j in range(1)])
lin3_cols = [int(v) for v in rng.integers(0, 256, 8)]                   # arbitrary (possibly singular) linear A3
A3 = np.array([np.bitwise_xor.reduce([lin3_cols[j] for j in range(8) if x >> j & 1] or [0]) for x in range(256)])
Fea = B[T[253][A]] ^ A3 ^ 0x5A
res["checks"]["EA_inv_of_EA_image_of_253_equal"] = ea_inv(Fea) == res["EA"][253]
assert res["checks"]["EA_inv_of_EA_image_of_253_equal"]
res["ea_pairs_separated"] = {f"{a}-{b}": res["EA"][a] != res["EA"][b] for a, b in [(253, 259), (763, 773), (1783, 1801), (253, 1783), (253, 1801), (259, 1783), (259, 1801)]}
res["EA"] = {k: [list(map(int, x[0])) + [int(x[1])] for x in v] for k, v in res["EA"].items()}
res["runtime_s"] = round(time.time() - t0, 1)
json.dump(res, open("results/exp8_summary.json", "w"), indent=1)
print(json.dumps(res, indent=1)[:3000])
