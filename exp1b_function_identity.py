"""Experiment 1b: exactly which unit indices n give the same function D_n(., 1) on F_256.
Claim checked exhaustively: D_m = D_n  iff  m in {n, -n, 256 n, -256 n} (mod 65535).
(256 = 2^8 is Frobenius of order 8, which is the identity on F_256; 256 = 1 mod 255 and -1 mod 257.)
"""
import json
from collections import defaultdict
from math import gcd
import numpy as np

T = np.load("results/dickson_table_a1.npy")
P = 65535
units = [n for n in range(1, P) if gcd(n, P) == 1]
groups = defaultdict(list)
for n in units:
    groups[T[n].tobytes()].append(n)
sizes = {len(g) for g in groups.values()}
for g in groups.values():
    n = g[0]
    assert set(g) == {n % P, (-n) % P, (256 * n) % P, (-256 * n) % P}, g
assert (256 % 255, 256 % 257) == (1, 256)
out = {"distinct_functions": len(groups), "group_sizes": sorted(sizes),
       "claim": "D_m = D_n iff m in {+-n, +-256n} mod 65535 (verified for all 32768 units)"}
json.dump(out, open("results/exp1b_summary.json", "w"), indent=1)
print(out)
