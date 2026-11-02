"""Experiment 9c: controls for the exact CCZ/EA searches on F_256 (see Experiment 9d for controls that
exercise full backtracking in F_64)."""
import json, time
import numpy as np
src = open("exp9b_ea_search.py").read(); src = src[:src.index("rng = np.random.default_rng")]
ns = {}; exec(src, ns)
src9 = open("exp9_ccz_search.py").read(); src9 = src9[:src9.index("if __name__")]
n9 = {}; exec(src9, n9)
T = ns["T"]; out = []
# 253 vs 763: CCZ invariants differ, so both answers must be 'not equivalent' (rejected early by the span
# dimension: 15 vs 14). 4681 vs 7: CCZ-equivalent (D_4681 is, up to Frobenius, the inverse of D_7), so the CCZ
# answer must be 'equivalent'. The EA question for 4681 vs 7 is settled by degree (5 vs 3); the exhaustive EA
# search for this pair did not finish within 75 minutes and is not run here.
for a, b, expect in [(253, 763, False), (4681, 7, True)]:
    r = n9["ccz_equivalent"](T[a], T[b], f"CCZ {a} vs {b}"); out.append(r); print(r, flush=True)
    assert r["equivalent"] == expect
r = ns["ea_equivalent"](T[253], T[763], "EA 253 vs 763"); out.append(r); print(r, flush=True)
assert not r["ea_equivalent"]
json.dump(out, open("results/exp9c_summary.json", "w"), indent=1)
