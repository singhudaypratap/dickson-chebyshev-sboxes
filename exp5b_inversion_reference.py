"""Experiment 5b: reference S-box per field, the inversion map x -> x^{q-2} on F_{2^m}, m = 3..10,
in the same subfield coordinates as Experiment 5 (any F_2-linear coordinates give the same invariants)."""
import json
import numpy as np
from exp5_general_m import setup, metrics
out = {}
for m in range(3, 11):
    ctx = setup(m); q = ctx["q"]; e, lg, N2 = ctx["exp"], ctx["log"], ctx["N2"]
    el = ctx["elem_of"]
    vals = np.zeros(q, dtype=np.int64)
    nz = el != 0
    vals[nz] = e[(-lg[el[nz]]) % N2]
    S = ctx["coord_of"][vals]
    assert np.unique(S).size == q
    du, nl, deg, _, _ = metrics(S, m)
    out[m] = {"DU": du, "NL": nl, "deg": deg}
    print(m, out[m], flush=True)
json.dump(out, open("results/exp5b_inversion.json", "w"), indent=1)
