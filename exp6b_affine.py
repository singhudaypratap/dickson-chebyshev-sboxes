"""Experiment 6b: resolve the orbit pairs left open by Experiment 6.
Affine equivalence S2(x) = B(S1(Ax + a)) + b  <=>  for some a: T2 = S2 + S2(0) is linearly equivalent to
T1_a(y) = S1(y + a) + S1(a). Decided by 256 exact LE searches per pair (one per a).
CCZ pair: tested for affine equivalence with S2 and with S2^{-1} (sufficient, not necessary, for CCZ).
"""
import json, time
from collections import defaultdict, Counter
import numpy as np
src = open("exp6_equivalence.py").read()
exec(src[:src.index("t0 = time.time()")])
t0 = time.time()
lin = json.load(open("results/exp6_linear_classes.json"))
inv = {}
for r in reps:
    S = T[r].astype(np.int64); D = full_ddt(S); L = abs_lat(S)
    inv[r] = ((tuple(sorted(Counter(D[1:].ravel().tolist()).items())), tuple(sorted(Counter(L.ravel().tolist()).items()))),
              (sm.algebraic_degree(S), tuple(sorted(sig_rows(D[1:]))), tuple(sorted(sig_rows(L)))))


def affine_equiv(S1, S2):
    S1 = np.asarray(S1, dtype=np.int64); S2 = np.asarray(S2, dtype=np.int64)
    T2 = S2 ^ S2[0]
    X = np.arange(256)
    for a in range(256):
        T1 = S1[X ^ a] ^ S1[a]
        if LE(T1, T2).run():
            return True, a
    return False, None


ag = defaultdict(list)
for r in reps:
    ag[inv[r][1]].append(r)
aff_pairs = [g for g in ag.values() if len(g) > 1]
res = {"affine_open_groups": aff_pairs, "affine_results": []}
for g in aff_pairs:                      # every pair inside each open group (exhaustive)
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            ok, a = affine_equiv(T[g[i]], T[g[j]])
            res["affine_results"].append({"pair": [g[i], g[j]], "affine_equivalent": ok, "shift_a": a})
par = {r: r for r in reps}
def fnd(a):
    while par[a] != a:
        a = par[a]
    return a
for x in res["affine_results"]:
    if x["affine_equivalent"]:
        par[fnd(x["pair"][0])] = fnd(x["pair"][1])
n_aff = len({fnd(r) for r in reps})
res["affine_classes_exact"] = n_aff
# CCZ: groups with equal CCZ invariants that are not already merged by inversion
orbit_of = {}
for r in reps:
    for k in range(16):
        for s in (1, -1):
            orbit_of[(s * pow(2, k, P) * r) % P] = r
inv_rep = {r: orbit_of[pow(r, -1, P)] for r in reps}
cg = defaultdict(set)
for r in reps:
    cg[inv[r][0]].add(frozenset({r, inv_rep[r]}))
open_ccz = [[sorted(s) for s in g] for g in cg.values() if len(g) > 1]
res["ccz_open"] = open_ccz
res["ccz_tests"] = []
for g in open_ccz:
    a_rep = g[0][0]
    for other in g[1:]:
        b = other[0]
        S2 = T[b].astype(np.int64)
        S2inv = np.zeros(256, dtype=np.int64); S2inv[S2] = np.arange(256)
        e1, _ = affine_equiv(T[a_rep], S2); e2, _ = affine_equiv(T[a_rep], S2inv)
        res["ccz_tests"].append({"pair": [a_rep, b], "affine_to_S2": e1, "affine_to_S2_inverse": e2})
res["runtime_s"] = round(time.time() - t0, 1)
json.dump(res, open("results/exp6b_summary.json", "w"), indent=1)
print(json.dumps(res, indent=1))
