"""Experiment 1c: characterise the weak Dickson permutations of F_256.
Claim tested exhaustively: DU(D_n) >= 72  <=>  n = +-2^i (mod 255)  or  n = +-2^j (mod 257).
Reason: then D_n coincides with a Frobenius power x -> x^(2^k) on one of the two halves
{u + 1/u : u in F_256^*} (128 points) or {u + 1/u : u^257 = 1} (128 points).
"""
import json
import numpy as np
d = np.genfromtxt("results/exp1_all_units.csv", delimiter=",", names=True, dtype=np.int64)
pm255 = {(s * pow(2, i, 255)) % 255 for i in range(8) for s in (1, -1)}
pm257 = {(s * pow(2, i, 257)) % 257 for i in range(16) for s in (1, -1)}
flag = np.array([(n % 255) in pm255 or (n % 257) in pm257 for n in d["n"]])
weak_du = d["DU"] >= 72
weak_nl = d["NL"] <= 56
out = {"size_pm2_mod255": len(pm255), "size_pm2_mod257": len(pm257),
       "flagged_units": int(flag.sum()), "DU_ge_72_units": int(weak_du.sum()), "NL_le_56_units": int(weak_nl.sum()),
       "flag_equals_DU_ge_72": bool(np.array_equal(flag, weak_du)), "flag_equals_NL_le_56": bool(np.array_equal(flag, weak_nl)),
       "max_DU_unflagged": int(d["DU"][~flag].max()), "min_NL_unflagged": int(d["NL"][~flag].min()),
       "min_DU_flagged": int(d["DU"][flag].min()), "max_NL_flagged": int(d["NL"][flag].max()),
       "both_flags_units": int(sum(((n % 255) in pm255 and (n % 257) in pm257) for n in d["n"]))}
assert out["flag_equals_DU_ge_72"] and out["flag_equals_NL_le_56"]
json.dump(out, open("results/exp1c_summary.json", "w"), indent=1)
print(out)
