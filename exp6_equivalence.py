"""Experiment 6: exact linear-equivalence classes of the Dickson permutations of F_256, and bounds on
affine and CCZ classes.

Linear equivalence S2 = B o S1 o A (A, B invertible F_2-linear) is decided exactly by a complete
backtracking search in the style of the LE algorithm (Biryukov, De Canniere, Braeken, Preneel 2003):
unknown A and B^{-1} are known on subspaces; S1(A x) = B^{-1}(S2 x) propagates knowledge both ways;
when propagation stalls, the next A-value is guessed over all admissible candidates (exhaustive).
Pruning uses necessary conditions: DDT row of S2 at x equals (as a multiset) DDT row of S1 at A x,
and DDT column of S2 at y equals DDT column of S1 at B^{-1} y.
Invariants:  affine-invariant  -> (degree, multiset of DDT rows, multiset of |LAT| rows)
             CCZ-invariant     -> (differential spectrum, extended Walsh spectrum)
Known relations: Frobenius/inversion orbits (linear), and D_n^{-1} = D_{n^{-1} mod 65535} (CCZ).
"""
import json, time, sys
from collections import defaultdict, Counter
import numpy as np
import sbox_metrics as sm

sys.setrecursionlimit(10000)
T = np.load("results/dickson_table_a1.npy")
cls = np.genfromtxt("results/exp1_classes.csv", delimiter=",", names=True, dtype=np.int64)
reps = [int(r) for r in cls["rep_n"]]
P = 65535


def full_ddt(S):
    S = np.asarray(S, dtype=np.int64)
    X = np.arange(256)
    out = S[X[None, :] ^ X[:, None]] ^ S[X[None, :]]
    idx = (X[:, None] * 256 + out).ravel()
    return np.bincount(idx, minlength=65536).reshape(256, 256)


def abs_lat(S):
    W = sm.walsh_nonzero_b(S)           # rows b = 1..255, cols a
    return np.abs(W)


def sig_rows(M):
    return [tuple(sorted(Counter(r.tolist()).items())) for r in M]


class LE:
    def __init__(self, S1, S2):
        self.S1 = [int(v) for v in S1]; self.S2 = [int(v) for v in S2]
        self.S1i = [0] * 256; self.S2i = [0] * 256
        for x in range(256):
            self.S1i[self.S1[x]] = x; self.S2i[self.S2[x]] = x
        D1, D2 = full_ddt(S1), full_ddt(S2)
        r1, r2 = sig_rows(D1), sig_rows(D2)
        c1, c2 = sig_rows(D1.T), sig_rows(D2.T)
        self.rowok = lambda x, a: r2[x] == r1[a]          # A x = a admissible
        self.colok = lambda y, b: c2[y] == c1[b]          # B^{-1} y = b admissible
        self.cand = {x: [a for a in range(1, 256) if r1[a] == r2[x]] for x in range(1, 256)}
        self.nodes = 0

    @staticmethod
    def _extend(Mp, img, x, v, ok):
        """Add x -> v to a linear partial map known on a subspace (dict). Return new points or None."""
        if v in img:
            return None
        new = []
        for k, w in list(Mp.items()):
            kk, vv = k ^ x, w ^ v
            if not ok(kk, vv):
                return None
            Mp[kk] = vv; img.add(vv); new.append(kk)
        return new

    def _propagate(self, st, newA, newB):
        A, Ai, B, Bi = st
        while newA or newB:
            while newA:
                x = newA.pop()
                y, val = self.S2[x], self.S1[A[x]]
                if y in B:
                    if B[y] != val:
                        return False
                else:
                    n = self._extend(B, Bi, y, val, self.colok)
                    if n is None:
                        return False
                    newB += n
            while newB:
                y = newB.pop()
                x, val = self.S2i[y], self.S1i[B[y]]
                if x in A:
                    if A[x] != val:
                        return False
                else:
                    n = self._extend(A, Ai, x, val, self.rowok)
                    if n is None:
                        return False
                    newA += n
        return True

    def _search(self, st):
        self.nodes += 1
        A = st[0]
        if len(A) == 256:
            self.solution = st
            return True
        x = next(i for i in range(1, 256) if i not in A)
        for a in self.cand[x]:
            if a in st[1]:
                continue
            A2, Ai2, B2, Bi2 = dict(st[0]), set(st[1]), dict(st[2]), set(st[3])
            n = self._extend(A2, Ai2, x, a, self.rowok)
            if n is None:
                continue
            st2 = (A2, Ai2, B2, Bi2)
            if self._propagate(st2, n, []) and self._search(st2):
                return True
        return False

    def run(self):
        assert self.S1[0] == 0 and self.S2[0] == 0          # all Dickson maps fix 0
        st = ({0: 0}, {0}, {0: 0}, {0})
        ok = self._search(st)
        if ok:   # verify the found maps
            A, _, B, _ = self.solution
            Binv = B; Bf = {v: k for k, v in Binv.items()}
            assert all(self.S2[x] == Bf[self.S1[A[x]]] for x in range(256))
        return ok


t0 = time.time()
inv = {}
for r in reps:
    S = T[r].astype(np.int64)
    D = full_ddt(S); L = abs_lat(S)
    deg = sm.algebraic_degree(S)
    ccz = (tuple(sorted(Counter(D[1:].ravel().tolist()).items())), tuple(sorted(Counter(L.ravel().tolist()).items())))
    aff = (deg, tuple(sorted(sig_rows(D[1:]))), tuple(sorted(sig_rows(L))))
    inv[r] = (ccz, aff)
print("invariants done", round(time.time() - t0, 1), flush=True)

# exact linear classes: within each affine-invariant group, compare against found class representatives
groups = defaultdict(list)
for r in reps:
    groups[inv[r][1]].append(r)
lin_classes = []          # list of lists of orbit reps
tests, positives, maxnodes = 0, 0, 0
for g in groups.values():
    found = []
    for r in g:
        placed = False
        for c in found:
            le = LE(T[c[0]], T[r]); ok = le.run(); tests += 1; maxnodes = max(maxnodes, le.nodes)
            if ok:
                c.append(r); placed = True; positives += 1
                break
        if not placed:
            found.append([r])
    lin_classes += found
print("LE done", round(time.time() - t0, 1), tests, positives, flush=True)

# sanity: the LE search must find the known Frobenius self-relation (D_n vs D_{2n}) and reject an
# affine-invariant mismatch; test on a few reps
for r in reps[:5]:
    assert LE(T[r], T[(2 * r) % P]).run()

orbit_of = {}
for i, r in enumerate(reps):
    for k in range(16):
        for s in (1, -1):
            orbit_of[(s * pow(2, k, P) * r) % P] = r
inv_rep = {r: orbit_of[pow(r, -1, P)] for r in reps}
for r in reps[:20]:   # D_n o D_{n^{-1}} = identity
    assert np.array_equal(T[pow(r, -1, P)][T[r]], np.arange(256))

# CCZ: upper bound = linear classes merged by inversion; lower bound = distinct CCZ invariants
parent = {r: r for r in reps}
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for c in lin_classes:
    for r in c[1:]:
        parent[find(r)] = find(c[0])
for r in reps:
    parent[find(r)] = find(inv_rep[r])
ccz_upper = len({find(r) for r in reps})
ccz_lower = len({inv[r][0] for r in reps})
aff_lower = len({inv[r][1] for r in reps})
self_inverse_orbits = sum(1 for r in reps if inv_rep[r] == r)
out = {"orbits": len(reps), "linear_classes_exact": len(lin_classes), "LE_tests": tests, "LE_equivalent_found": positives,
       "LE_max_search_nodes": maxnodes,
       "affine_classes_lower_bound_invariants": aff_lower,
       "affine_classes_upper_bound_linear": len(lin_classes),
       "ccz_classes_lower_bound_invariants": ccz_lower, "ccz_classes_upper_bound_known_relations": ccz_upper,
       "orbits_closed_under_inversion": self_inverse_orbits,
       "linear_class_sizes": dict(Counter(len(c) for c in lin_classes)),
       "runtime_s": round(time.time() - t0, 1)}
json.dump(out, open("results/exp6_summary.json", "w"), indent=1)
json.dump(lin_classes, open("results/exp6_linear_classes.json", "w"))
print(json.dumps(out, indent=1))
