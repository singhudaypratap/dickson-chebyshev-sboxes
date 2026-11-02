"""Experiment 6c: numerical check that the invariants used to separate classes are EA invariants.
For F' = A1 o F o A2 + A3 (A1, A2 affine bijections, A3 affine and possibly singular):
  row alpha of DDT(F') is a column-permutation of row A2(alpha) of DDT(F);
  row b of |LAT(F')| is a column-permutation of row A1^T(b) of |LAT(F)|; degree is unchanged (deg >= 2).
So (degree, multiset of DDT rows, multiset of |LAT| rows) is EA-invariant. Tested on 40 random EA images."""
import json
import numpy as np
src = open("exp6_equivalence.py").read(); src = src[:src.index("t0 = time.time()")]
ns = {}; exec(src, ns)
T, full_ddt, abs_lat, sig_rows, sm = ns["T"], ns["full_ddt"], ns["abs_lat"], ns["sig_rows"], ns["sm"]
rng = np.random.default_rng(20261031 + 63)
X = np.arange(256)
def lin(cols): return np.array([np.bitwise_xor.reduce([cols[j] for j in range(8) if x >> j & 1] or [0]) for x in range(256)])
def rand_inv():
    while True:
        cols = [int(v) for v in rng.integers(1, 256, 8)]
        span = {0}
        for c in cols: span |= {s ^ c for s in span}
        if len(span) == 256: return lin(cols)
def inv(S):
    S = np.asarray(S, dtype=np.int64)
    return (sm.algebraic_degree(S), tuple(sorted(sig_rows(full_ddt(S)[1:]))), tuple(sorted(sig_rows(abs_lat(S)))))
reps = [int(r) for r in ns["cls"]["rep_n"]]
ok = 0; singular = 0
for t in range(40):
    n = int(rng.choice(reps)); F = T[n].astype(np.int64)
    A1, A2 = rand_inv(), rand_inv()
    cols3 = [int(v) for v in rng.integers(0, 256, 8)]; A3 = lin(cols3)
    singular += int(np.unique(A3).size < 256)
    Fp = A1[F[A2[X] ^ int(rng.integers(256))]] ^ A3 ^ int(rng.integers(256))
    ok += int(inv(Fp) == inv(F))
out = {"trials": 40, "invariant_preserved": ok, "trials_with_singular_A3": singular}
assert ok == 40
json.dump(out, open("results/exp6c_summary.json", "w"), indent=1); print(out)
