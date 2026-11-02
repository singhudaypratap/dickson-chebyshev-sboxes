"""Experiment 4: layer collapse of cascaded Chebyshev (Dickson, a = 1) stages with secret degrees.

Cascade:   E(x) = D_{n_R}( ... D_{n_1}(x) ... ),  n_i secret units mod 65535  (nominal key space 32768^R)
Semigroup: E = D_N with N = n_1 ... n_R mod 65535, so an attacker searches only the 32768 units.
Control:   E'(x) = D_{n_2}( D_{n_1}(x) ^ k ), a key byte k between layers: test whether any single D_N fits.
"""
import json
import numpy as np
from math import gcd

cfg = json.load(open("config.json"))["collapse_demo"]
seed = json.load(open("config.json"))["experiments"]["seed"]
rng = np.random.default_rng(seed + 4)
T = np.load("results/dickson_table_a1.npy").astype(np.int64)
P = 65535
units = np.array([n for n in range(1, P) if gcd(n, P) == 1])
out = {"num_units": int(units.size), "cascade": [], "control_xor": {}}

for R in cfg["rounds"]:
    for kp in cfg["known_pairs"]:
        true_found, unique_fn, n_cand = 0, 0, []
        for _ in range(cfg["num_trials"]):
            ns = rng.choice(units, R)
            E = np.arange(256)
            for n in ns:
                E = T[n][E]
            N_true = int(np.prod([int(n) for n in ns], dtype=object) % P)
            assert np.array_equal(T[N_true], E)                              # semigroup collapse
            xs = rng.choice(256, kp, replace=False)
            cand = units[np.all(T[units][:, xs] == E[xs][None, :], axis=1)]  # exhaustive search over units
            true_found += int(N_true in set(cand.tolist()))
            unique_fn += int(len({T[N].tobytes() for N in cand}) == 1)       # all candidates = same function
            n_cand.append(len(cand))
        out["cascade"].append({"rounds": R, "known_pairs": kp, "trials": cfg["num_trials"],
                               "true_index_found": true_found, "candidates_all_same_function": unique_fn,
                               "candidates_median": float(np.median(n_cand)), "candidates_max": int(max(n_cand)),
                               "search_cost_evaluations": int(units.size),
                               "nominal_keyspace_log2": float(R * np.log2(units.size))})

fits = 0
for _ in range(cfg["num_trials"]):
    n1, n2 = rng.choice(units, 2); k = int(rng.integers(1, 256))
    E = T[n2][T[n1] ^ k]
    fits += int(np.any(np.all(T[units] == E[None, :], axis=1)))
out["control_xor"] = {"trials": cfg["num_trials"], "trials_where_a_single_D_N_fits": fits}
json.dump(out, open("results/exp4_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
