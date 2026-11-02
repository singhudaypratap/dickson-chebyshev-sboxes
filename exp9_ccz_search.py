"""Experiment 9: exact CCZ-equivalence decision for the open pair (orbits of D_253 and D_1783).

F and G are CCZ-equivalent iff an affine bijection L of F_2^16 maps graph(F) = {(x, F(x))} onto graph(G)
(Carlet, Charpin, Zinoviev 1998). Since F(0) = 0, L(0) = c lies in graph(G); the linear part of L then maps
S1 = graph(F) onto S2 = graph(G) + c. We search for that linear map exhaustively: choose 16 independent points
p_1..p_16 of S1, assign images q_j in S2 one at a time, and require for every subset sum of the assigned
points that (p_j + P) in S1  <=>  (q_j + Q) in S2. Candidates are pruned by the local invariant
t(p) = |S n (S + p)|. The search is complete: if it ends without a solution for every c, no CCZ map exists.
Positive controls: G = F^{-1} (swap of coordinates) and G = an affine image of F.
"""
import json, sys, time
import numpy as np

T = np.load("results/dickson_table_a1.npy").astype(np.int64)
X = np.arange(256, dtype=np.int64)


def graph(F):
    return (X | (np.asarray(F, dtype=np.int64) << 8))


def member(points):
    m = np.zeros(1 << 16, dtype=bool); m[points] = True; return m


def local_inv(pts, mem):
    return {int(p): int(mem[pts ^ p].sum()) for p in pts}


class Search:
    def __init__(self, S1, S2):
        self.S1, self.S2 = S1, S2
        self.m1, self.m2 = member(S1), member(S2)
        self.t1, self.t2 = local_inv(S1, self.m1), local_inv(S2, self.m2)
        # basis of S1: greedy, rarest local invariant first
        cnt = {}
        for v in self.t2.values():
            cnt[v] = cnt.get(v, 0) + 1
        order = sorted((p for p in S1.tolist() if p), key=lambda p: (cnt.get(self.t1[p], 0), p))
        basis, span = [], {0}
        for p in order:
            if p not in span:
                basis.append(p); span |= {s ^ p for s in span}
        self.dim = len(basis)                 # graph may lie in a hyperplane when F has a linear component
        span2 = {0}
        for q in S2.tolist():
            if q not in span2:
                span2 |= {s ^ q for s in span2}
        self.dim2 = len(span2).bit_length() - 1
        self.basis = basis
        self.cands = [[q for q in S2.tolist() if q and self.t2[q] == self.t1[p]] for p in basis]
        self.nodes = 0

    def run(self):
        if self.dim != self.dim2:
            return False
        return self._rec(0, np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64))

    def _rec(self, j, Pspan, Qspan):
        self.nodes += 1
        if j == self.dim:            # linear bijection span(S1) -> span(S2); extends to F_2^16
            self.solution = (list(self.basis), Qspan)
            return True
        p = self.basis[j]
        inP = self.m1[Pspan ^ p]
        for q in self.cands[j]:
            if self.m2[Qspan ^ q].tolist() != inP.tolist():
                continue
            # q must be outside span(Q): guaranteed, since otherwise q ^ Q = 0 for some Q, and 0 in S2
            # while p ^ P != 0 is the matching point; membership of 0 differs only if p ^ P not in S1 ... check:
            if np.any((Qspan ^ q) == 0):
                continue
            if self._rec(j + 1, np.concatenate([Pspan, Pspan ^ p]), np.concatenate([Qspan, Qspan ^ q])):
                return True
        return False


def verify(S1, SG, c, s):
    """Rebuild the linear map from basis images and check L(graph F) + c == graph G point by point."""
    basis = s.basis
    # Qspan is ordered like the subset sums of the basis: index bit i <-> basis[i]
    _, Qspan = s.solution
    img = {}
    Pspan = np.zeros(1, dtype=np.int64)
    for p in basis:
        Pspan = np.concatenate([Pspan, Pspan ^ p])
    for P, Q in zip(Pspan.tolist(), Qspan.tolist()):
        img[P] = Q
    mapped = sorted((img[int(p)] ^ c) for p in S1.tolist())
    assert mapped == sorted(SG.tolist()), "verification failed"
    # linearity spot check on the subspace
    for P1, P2 in [(int(a), int(b)) for a, b in np.random.default_rng(0).choice(Pspan, (2000, 2))]:
        assert img[P1 ^ P2] == img[P1] ^ img[P2]


def ccz_equivalent(F, G, label):
    S1 = graph(F); t0 = time.time(); nodes = 0
    for c in graph(G).tolist():
        s = Search(S1, graph(G) ^ c)
        if s.run():
            verify(S1, graph(G), c, s)
            return {"pair": label, "equivalent": True, "translation_c": int(c), "nodes": nodes + s.nodes, "s": round(time.time() - t0, 1)}
        nodes += s.nodes
    return {"pair": label, "equivalent": False, "nodes": nodes, "s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    rng = np.random.default_rng(20261031 + 9)
    F = T[253]
    Finv = np.zeros(256, dtype=np.int64); Finv[F] = X

    def rand_lin():
        while True:
            cols = [int(v) for v in rng.integers(1, 256, 8)]
            span = {0}
            for cc in cols:
                span |= {s ^ cc for s in span}
            if len(span) == 256:
                return np.array([np.bitwise_xor.reduce([cols[j] for j in range(8) if x >> j & 1] or [0]) for x in range(256)])
    A, B = rand_lin(), rand_lin()
    Faff = B[F[A ^ 0x3C]] ^ 0xC3
    Faff = Faff ^ Faff[0]                                    # keep G(0) = 0 for the search framing
    out = []
    for G, lab in [(Finv, "253 vs its inverse (control)"), (Faff, "253 vs affine image (control)"),
                   (T[1783], "253 vs 1783 (open pair)")]:
        r = ccz_equivalent(F, G, lab); out.append(r); print(r, flush=True)
    assert out[0]["equivalent"] and out[1]["equivalent"]
    json.dump(out, open("results/exp9_summary.json", "w"), indent=1)
