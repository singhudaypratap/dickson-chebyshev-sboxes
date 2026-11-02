"""Experiment 2: finite-state structure of a quantised real Chebyshev map versus exact Dickson maps.

Quantised map on a b-bit grid (M = 2^b states):
  g_k = -1 + 2k/(M-1),  f(k) = round( (T_n(g_k) + 1)/2 * (M-1) ),  T_n(x) = cos(n arccos x)  (float64)
Measures (functional graph of f): image fraction |f(S)|/M, cyclic fraction, number of cycles, longest cycle.
Dickson D_n(x,1) on F_256 (exact): the same measures; it is a permutation so image and cyclic fractions are 1.
"""
import json
import numpy as np

cfg = json.load(open("config.json"))


def functional_graph_stats(f):
    f = np.asarray(f, dtype=np.int64)
    M = f.size
    indeg = np.bincount(f, minlength=M)
    image_fraction = np.count_nonzero(indeg) / M
    # peel nodes of in-degree 0 (Kahn); survivors are exactly the cyclic nodes
    alive = np.ones(M, dtype=bool)
    stack = list(np.flatnonzero(indeg == 0))
    indeg = indeg.copy()
    while stack:
        v = stack.pop()
        alive[v] = False
        w = f[v]
        indeg[w] -= 1
        if indeg[w] == 0:
            stack.append(w)
    cyc_nodes = np.flatnonzero(alive)
    seen = np.zeros(M, dtype=bool)
    lengths = []
    for v in cyc_nodes:
        if seen[v]:
            continue
        L = 0; w = v
        while not seen[w]:
            seen[w] = True; w = f[w]; L += 1
        assert w == v
        lengths.append(L)
    lengths = np.array(lengths)
    return {"image_fraction": float(image_fraction), "cyclic_fraction": float(cyc_nodes.size / M),
            "num_cycles": int(lengths.size), "longest_cycle": int(lengths.max()),
            "fixed_points": int(np.sum(lengths == 1))}


def quantised_chebyshev(n, bits):
    M = 2 ** bits
    k = np.arange(M)
    g = -1.0 + 2.0 * k / (M - 1)
    t = np.cos(n * np.arccos(np.clip(g, -1.0, 1.0)))
    f = np.rint((t + 1.0) / 2.0 * (M - 1)).astype(np.int64)
    assert f.min() >= 0 and f.max() <= M - 1
    return f


out = {"quantised": [], "dickson": {}}
for n in cfg["quantized_chebyshev"]["degrees"]:
    for b in cfg["quantized_chebyshev"]["word_bits"]:
        s = functional_graph_stats(quantised_chebyshev(n, b))
        out["quantised"].append({"n": n, "bits": b, **s})

T = np.load("results/dickson_table_a1.npy")
classes = np.genfromtxt("results/exp1_classes.csv", delimiter=",", names=True, dtype=np.int64)
stats = [functional_graph_stats(T[int(r)]) for r in classes["rep_n"]]
lc = np.array([s["longest_cycle"] for s in stats]); nc = np.array([s["num_cycles"] for s in stats])
fp = np.array([s["fixed_points"] for s in stats])
assert all(s["image_fraction"] == 1.0 and s["cyclic_fraction"] == 1.0 for s in stats)
out["dickson"] = {
    "num_classes": int(lc.size),
    "longest_cycle_min": int(lc.min()), "longest_cycle_median": float(np.median(lc)), "longest_cycle_max": int(lc.max()),
    "num_cycles_min": int(nc.min()), "num_cycles_median": float(np.median(nc)), "num_cycles_max": int(nc.max()),
    "fixed_points_min": int(fp.min()), "fixed_points_max": int(fp.max()),
}
# Algebraic period bound: x = u + 1/u. If u lies in F_256^* (ord(u) | 255) the orbit length of x under D_n
# is the order of n in (Z/ord(u))^*/{+-1}, hence <= 16; otherwise ord(u) | 257 and the orbit length is <= 128.
import gf256 as gf
nz = np.arange(1, 256)
split = np.zeros(256, dtype=bool)
split[np.unique(nz ^ gf.inv(nz))] = True           # x = u + u^{-1} with u in F_256^*
out["dickson"]["num_x_with_u_in_F256"] = int(split.sum())


def orbit_lengths(f):
    L = np.zeros(256, dtype=np.int64)
    for x in range(256):
        y, k = int(f[x]), 1
        while y != x:
            y, k = int(f[y]), k + 1
        L[x] = k
    return L


max_split, max_nonsplit = 0, 0
for r in classes["rep_n"]:
    L = orbit_lengths(T[int(r)])
    max_split = max(max_split, int(L[split].max())); max_nonsplit = max(max_nonsplit, int(L[~split].max()))
assert max_split <= 16 and max_nonsplit <= 128
out["dickson"]["max_orbit_len_u_in_F256"] = max_split
out["dickson"]["max_orbit_len_u_order_dividing_257"] = max_nonsplit
best = json.load(open("results/exp1_summary.json"))["best_classes"][0]["rep_n"]
out["dickson"]["best_class_rep"] = int(best)
out["dickson"]["best_class_cycle_stats"] = functional_graph_stats(T[int(best)])
# random permutation reference for cycle counts
rng = np.random.default_rng(cfg["experiments"]["seed"] + 2)
rl = [functional_graph_stats(rng.permutation(256)) for _ in range(5000)]
out["random_perm_cycles"] = {"num_cycles_median": float(np.median([s["num_cycles"] for s in rl])),
                             "longest_cycle_median": float(np.median([s["longest_cycle"] for s in rl]))}
json.dump(out, open("results/exp2_summary.json", "w"), indent=1)
for r in out["quantised"]:
    print(r)
print(json.dumps(out["dickson"], indent=1)); print(out["random_perm_cycles"])
