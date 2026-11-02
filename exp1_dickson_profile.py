"""Experiment 1: exhaustive profile of Dickson permutations of F_256 and structural checks.

Outputs (results/):
  exp1_summary.json        all scalar findings
  exp1_all_units.csv       n, is_perm, DU, NL, deg, fixed points, class id for every unit n mod 65535
  exp1_classes.csv         one row per class (orbit of n under n -> 2n, n -> -n)
  exp1_random_perms.csv    DU/NL/deg for random permutations (baseline)
"""
import json, time, platform
from math import gcd
import numpy as np
import gf256 as gf
import dickson as dk
import sbox_metrics as sm

cfg = json.load(open("config.json"))
assert int(cfg["field"]["modulus_poly_hex"], 16) == gf.MODULUS
rng = np.random.default_rng(cfg["experiments"]["seed"])
out = {"python": platform.python_version(), "numpy": np.__version__}
t0 = time.time()

# ---------- 0. validate field and metrics on known functions ----------
gf.validate()


def aes_sbox():
    inv = np.zeros(256, dtype=np.int64)
    inv[1:] = gf.inv(np.arange(1, 256))
    s = np.zeros(256, dtype=np.int64)
    for x in range(256):
        b = int(inv[x]); r = 0
        for i in range(8):
            bit = ((b >> i) ^ (b >> ((i + 4) % 8)) ^ (b >> ((i + 5) % 8)) ^ (b >> ((i + 6) % 8)) ^ (b >> ((i + 7) % 8)) ^ (0x63 >> i)) & 1
            r |= bit << i
        s[x] = r
    return s


AES = aes_sbox()
assert AES[0x00] == 0x63 and AES[0x01] == 0x7C and AES[0x53] == 0xED   # FIPS-197 values
checks = {
    "AES": sm.profile(AES),
    "inverse_x254": sm.profile(gf.power(dk.ALLX, 254)),
    "identity": sm.profile(dk.ALLX),
    "gold_x3_DU": sm.differential_uniformity(gf.power(dk.ALLX, 3)),
    "kasami_x57_DU": sm.differential_uniformity(gf.power(dk.ALLX, 57)),
}
assert checks["AES"]["DU"] == 4 and checks["AES"]["NL"] == 112 and checks["AES"]["deg"] == 7
assert checks["inverse_x254"]["DU"] == 4 and checks["inverse_x254"]["NL"] == 112
assert checks["identity"]["DU"] == 256 and checks["identity"]["NL"] == 0 and checks["identity"]["deg"] == 1
assert checks["gold_x3_DU"] == 2 and checks["kasami_x57_DU"] == 2          # known APN power maps
out["metric_validation"] = checks

# ---------- 1. Dickson table D_n(x,1), cross-validated ----------
NMAX = cfg["dickson"]["n_max_table"]
T = dk.recurrence_table_a1(NMAX)
P = dk.PERIOD
assert np.array_equal(T[P], T[0]) and np.array_equal(T[P + 1], T[1]), "period 65535 violated"
for n in range(0, cfg["dickson"]["explicit_formula_check_nmax"] + 1):
    for a in cfg["dickson"]["explicit_formula_check_a_values"]:
        e = dk.dickson_explicit(n, dk.ALLX, a)
        m = dk.dickson_matpow(n, dk.ALLX, a)
        assert np.array_equal(e, m), (n, a)
        if a == 1:
            assert np.array_equal(e, T[n].astype(np.int64)), n
for n in rng.integers(2, P, 300):
    assert np.array_equal(dk.dickson_matpow(int(n), dk.ALLX, 1), T[int(n)].astype(np.int64))
out["cross_validation"] = "recurrence == matrix power == explicit formula (n<=96, 4 values of a); recurrence == matrix power at 300 random n"

# ---------- 2. permutation criterion gcd(n, q^2-1) = 1, exhaustively ----------
rows = np.sort(T[1:P], axis=1)
is_perm = np.all(np.diff(rows.astype(np.int64), axis=1) > 0, axis=1)       # n = 1..65534
ns = np.arange(1, P)
crit = np.array([gcd(int(n), P) == 1 for n in ns])
assert np.array_equal(is_perm, crit), "permutation criterion mismatch"
units = ns[crit]
out["num_permutation_indices"] = int(units.size)
distinct = {T[n].tobytes() for n in units}
out["num_distinct_permutations"] = len(distinct)

# ---------- 3. classes under n -> 2n, n -> -n (mod 65535) ----------
cls = -np.ones(P, dtype=np.int64)
reps = []
for n in units:
    if cls[n] >= 0:
        continue
    orbit = set(); frontier = [int(n)]
    while frontier:
        m = frontier.pop()
        if m in orbit: continue
        orbit.add(m)
        frontier += [(2 * m) % P, (-m) % P]
    for m in orbit: cls[m] = len(reps)
    reps.append(int(n))
out["num_classes"] = len(reps)
orbit_sizes = np.bincount(cls[units])
out["orbit_size_counts"] = {int(k): int(v) for k, v in zip(*np.unique(orbit_sizes, return_counts=True))}

# Frobenius and inversion identities (as functions on F_256)
for n in units[:2000]:
    assert np.array_equal(T[(2 * n) % P].astype(np.int64), gf.mul(T[n], T[n]))   # D_{2n} = D_n^2
    assert np.array_equal(T[(-n) % P], T[n])                                     # D_{-n} = D_n

# ---------- 4. metrics for every unit n ----------
prof = {}
for i, n in enumerate(units):
    prof[int(n)] = sm.profile(T[n])
# invariance on each class (exhaustive)
for c in range(len(reps)):
    members = units[cls[units] == c]
    vals = {(prof[int(m)]["DU"], prof[int(m)]["NL"], prof[int(m)]["deg"]) for m in members}
    assert len(vals) == 1, ("class not invariant", c)
with open("results/exp1_all_units.csv", "w") as f:
    f.write("n,class,DU,NL,deg,fixed_points\n")
    for n in units:
        p = prof[int(n)]
        f.write(f"{n},{cls[n]},{p['DU']},{p['NL']},{p['deg']},{p['fixed_points']}\n")
with open("results/exp1_classes.csv", "w") as f:
    f.write("class,rep_n,orbit_size,DU,NL,deg,fixed_points_rep\n")
    for c, r in enumerate(reps):
        p = prof[r]
        f.write(f"{c},{r},{orbit_sizes[c]},{p['DU']},{p['NL']},{p['deg']},{p['fixed_points']}\n")


def dist(key, over):
    v, cnt = np.unique([prof[int(n)][key] for n in over], return_counts=True)
    return {int(a): int(b) for a, b in zip(v, cnt)}


out["DU_distribution_units"] = dist("DU", units)
out["NL_distribution_units"] = dist("NL", units)
out["deg_distribution_units"] = dist("deg", units)
out["fixed_point_distribution_units"] = dist("fixed_points", units)
out["DU_distribution_classes"] = dist("DU", reps)
out["NL_distribution_classes"] = dist("NL", reps)
best = sorted(reps, key=lambda r: (prof[r]["DU"], -prof[r]["NL"], -prof[r]["deg"], r))
out["best_classes"] = [{"rep_n": r, **prof[r]} for r in best[:10]]
out["worst_classes"] = [{"rep_n": r, **prof[r]} for r in best[-5:]]

# ---------- 5. parameter a: D_n(x,a) = b^n D_n(x/b, 1), b = sqrt(a) ----------
A = np.arange(1, 256, dtype=np.int64)
b = gf.sqrt(A)
check_ns = [int(r) for r in reps]                      # every class representative
for n in check_ns:
    lhs = dk.dickson_matpow(n, dk.ALLX[None, :], A[:, None])              # (255, 256)
    rhs = gf.mul(gf.power(b, n)[:, None], T[n][gf.mul(dk.ALLX[None, :], gf.inv(b)[:, None])].astype(np.int64))
    assert np.array_equal(lhs, rhs), n
metric_ns = [best[0], best[len(best) // 2], best[-1]] + [int(v) for v in rng.choice(reps, cfg["dickson"]["param_a_metric_check_count"] - 3, replace=False)]
a_metric = {}
for n in metric_ns:
    lhs = dk.dickson_matpow(n, dk.ALLX[None, :], A[:, None])
    vals = {tuple(sm.profile(lhs[i]).values())[:3] for i in range(255)}
    assert vals == {(prof[n]["DU"], prof[n]["NL"], prof[n]["deg"])}, n
    a_metric[n] = len(vals)
out["param_a_identity_checked_for_class_reps"] = len(check_ns)
out["param_a_metrics_checked_for_n"] = metric_ns

# ---------- 6. semigroup collapse D_m(D_n(x)) = D_{mn}(x) ----------
pairs = rng.integers(0, P, size=(cfg["dickson"]["semigroup_random_pairs"], 2))
for m, n in pairs:
    assert np.array_equal(T[m][T[n]], T[(int(m) * int(n)) % P])
out["semigroup_pairs_checked"] = int(len(pairs))

# ---------- 7. random permutation baseline ----------
R = cfg["random_baseline"]["num_random_permutations"]
rp = []
for _ in range(R):
    s = rng.permutation(256)
    rp.append((sm.differential_uniformity(s), sm.nonlinearity(s), sm.algebraic_degree(s)))
rp = np.array(rp)
np.savetxt("results/exp1_random_perms.csv", rp, fmt="%d", delimiter=",", header="DU,NL,deg", comments="")
out["random_perm"] = {
    "count": R,
    "DU_distribution": {int(a): int(c) for a, c in zip(*np.unique(rp[:, 0], return_counts=True))},
    "NL_distribution": {int(a): int(c) for a, c in zip(*np.unique(rp[:, 1], return_counts=True))},
    "deg_distribution": {int(a): int(c) for a, c in zip(*np.unique(rp[:, 2], return_counts=True))},
}
np.save("results/dickson_table_a1.npy", T[:P])
out["runtime_s"] = round(time.time() - t0, 1)
json.dump(out, open("results/exp1_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("best_classes",)}, indent=1)[:4000])
print("best", out["best_classes"][:5])
