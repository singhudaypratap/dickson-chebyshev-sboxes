"""Experiment 9b: exact EA-equivalence decisions for the orbit pairs that share all EA invariants.
F' = A1 o F o A2 + A3 (A1, A2 affine bijections, A3 affine) iff some affine bijection L of F_2^16 maps
graph(F) onto graph(F') and maps V0 = {0} x F_2^8 onto itself. This is the CCZ search of Experiment 9 with one
extra constraint per subset sum: (p_j + P in V0) <=> (q_j + Q in V0). On span(graph F) this condition is also
sufficient for an extension to all of F_2^16 with L(V0) = V0 (dimensions of the intersections then agree).
"""
import json, time
import numpy as np
src = open("exp9_ccz_search.py").read(); src = src[:src.index("if __name__")]
ns = {}; exec(src, ns)
T, X, graph, Search, verify = ns["T"], ns["X"], ns["graph"], ns["Search"], ns["verify"]
V0 = np.zeros(1 << 16, dtype=bool); V0[np.arange(256) << 8] = True


class EASearch(Search):
    def _rec(self, j, Pspan, Qspan):
        self.nodes += 1
        if j == self.dim:
            self.solution = (list(self.basis), Qspan)
            return True
        p = self.basis[j]
        inP = self.m1[Pspan ^ p]; inV = V0[Pspan ^ p]
        for q in self.cands[j]:
            if np.any((Qspan ^ q) == 0):
                continue
            if self.m2[Qspan ^ q].tolist() != inP.tolist() or V0[Qspan ^ q].tolist() != inV.tolist():
                continue
            if self._rec(j + 1, np.concatenate([Pspan, Pspan ^ p]), np.concatenate([Qspan, Qspan ^ q])):
                return True
        return False


def ea_equivalent(F, G, label):
    S1 = graph(F); t0 = time.time(); nodes = 0
    for c in graph(G).tolist():
        s = EASearch(S1, graph(G) ^ c)
        if s.run():
            verify(S1, graph(G), c, s)
            Pspan = np.zeros(1, dtype=np.int64)
            for p in s.basis:
                Pspan = np.concatenate([Pspan, Pspan ^ p])
            Qspan = s.solution[1]
            assert np.array_equal(V0[Pspan], V0[Qspan]), "V0 not preserved"
            # first coordinate of L(x, F(x)) must be a bijection of F_2^8 (EA form: x' = A2^{-1} x)
            img = dict(zip(Pspan.tolist(), Qspan.tolist()))
            firsts = {(img[int(p)] ^ c) & 255 for p in S1.tolist()}
            assert len(firsts) == 256
            return {"pair": label, "ea_equivalent": True, "c": int(c), "nodes": nodes + s.nodes, "s": round(time.time() - t0, 1)}
        nodes += s.nodes
    return {"pair": label, "ea_equivalent": False, "nodes": nodes, "s": round(time.time() - t0, 1)}


rng = np.random.default_rng(20261031 + 91)
def rand_lin():
    while True:
        cols = [int(v) for v in rng.integers(1, 256, 8)]
        span = {0}
        for cc in cols:
            span |= {s ^ cc for s in span}
        if len(span) == 256:
            return np.array([np.bitwise_xor.reduce([cols[j] for j in range(8) if x >> j & 1] or [0]) for x in range(256)])
A, B = rand_lin(), rand_lin()
cols = [int(v) for v in rng.integers(0, 256, 8)]
A3 = np.array([np.bitwise_xor.reduce([cols[j] for j in range(8) if x >> j & 1] or [0]) for x in range(256)])
Fea = B[T[253][A]] ^ A3
out = [ea_equivalent(T[253], Fea, "253 vs random EA image (positive control)")]
print(out[-1], flush=True)
assert out[-1]["ea_equivalent"]
Finv = np.zeros(256, dtype=np.int64); Finv[T[253]] = X
for a, b in [(253, 259), (763, 773), (1783, 1801), (253, 1783), (253, 1801)]:
    out.append(ea_equivalent(T[a], T[b], f"{a} vs {b}")); print(out[-1], flush=True)
json.dump(out, open("results/exp9b_summary.json", "w"), indent=1)
