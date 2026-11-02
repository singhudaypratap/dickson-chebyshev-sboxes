"""Experiment 10: explicit, independently checkable certificates for every equivalence used as an upper bound
in the class counts of Section 4 (1 024 linear/affine -> 1 020 EA -> 513 CCZ over F_256).

For each merge the search of exp9/exp9b finds a linear map on span(graph F). Here it is extended to a full
invertible 16 x 16 matrix M over F_2 (with M(V0) = V0 for EA, V0 = {0} x F_2^8) and stored with the
translation c. The verifier below uses nothing from the search: it checks rank(M) = 16, that x -> M x + c maps
graph(F) onto graph(G) as sets, and for EA that M maps V0 onto V0. Vectors are 16-bit integers x | (y << 8).

CCZ pairing under inversion: graph(D_n^{-1}) is the coordinate swap of graph(D_n), and D_n^{-1} = D_{n'} with
n n' = 1 (mod 65535). The certificate lists, for each of the 1 024 orbits, its EA class and the EA class of the
inverse orbit; merging these pairs gives the stated CCZ count, and the swap is checked on every table.
Output: results/exp10_certificates.json
"""
import json
from math import gcd
import numpy as np

src = open("exp9_ccz_search.py").read(); src = src[:src.index("if __name__")]
ns = {}; exec(src, ns)
T, X, graph, Search = ns["T"], ns["X"], ns["graph"], ns["Search"]
src_b = open("exp9b_ea_search.py").read(); src_b = src_b[:src_b.index("rng = np.random")]
nsb = {}; exec(src_b, nsb)
EASearch = nsb["EASearch"]
N = 65535
V0_BASIS = [1 << (8 + i) for i in range(8)]


# ---------------- GF(2) linear algebra on 16-bit ints ----------------
def reduce_basis(vecs):
    """Return a dict pivot -> vector (echelon form) for span(vecs)."""
    piv = {}
    for v in vecs:
        for b in sorted(piv, reverse=True):
            if v >> b & 1:
                v ^= piv[b]
        if v:
            piv[v.bit_length() - 1] = v
            # keep fully reduced not needed; independence test only
    return piv


def in_span(piv, v):
    for b in sorted(piv, reverse=True):
        if v >> b & 1:
            v ^= piv[b]
    return v == 0


def rank(vecs):
    return len(reduce_basis(vecs))


def solve_matrix(dom, img):
    """Given 16 independent domain vectors and their images, return columns M[e_i] (i = 0..15)."""
    # express e_i in the domain basis by Gaussian elimination with tracking
    rows = [(v, 1 << k) for k, v in enumerate(dom)]
    piv = {}
    for v, tag in rows:
        for b in sorted(piv, reverse=True):
            if v >> b & 1:
                v ^= piv[b][0]; tag ^= piv[b][1]
        assert v, "domain vectors dependent"
        piv[v.bit_length() - 1] = (v, tag)
    cols = []
    for i in range(16):
        v, tag = 1 << i, 0
        for b in sorted(piv, reverse=True):
            if v >> b & 1:
                v ^= piv[b][0]; tag ^= piv[b][1]
        assert v == 0
        w = 0
        for k in range(16):
            if tag >> k & 1:
                w ^= img[k]
        cols.append(w)
    return cols


def apply(cols, v):
    w = 0
    for i in range(16):
        if v >> i & 1:
            w ^= cols[i]
    return w


def extend(basis, images, ea):
    """Extend a linear injection basis -> images (on a subspace D) to an invertible map of F_2^16;
    if ea, also with V0 -> V0 (possible when dim(D cap V0) = dim(L(D) cap V0), which is checked)."""
    dom, img = list(basis), list(images)
    if ea:
        # basis of D cap V0 and of its image: vectors of span(D) lying in V0
        span = {0: 0}
        for p, q in zip(basis, images):
            span.update({a ^ p: b ^ q for a, b in span.items()})
        DV = [a for a in span if a and a & 255 == 0]
        assert all(span[a] & 255 == 0 for a in DV), "L does not map D cap V0 into V0"
        EV = [span[a] for a in DV]
        assert rank(DV) == rank(EV)
        # complement of D cap V0 inside V0, mapped to a complement of L(D) cap V0 inside V0
        pd, pe = reduce_basis(dom), reduce_basis(img)
        free_d = [v for v in V0_BASIS]
        for v in free_d:
            if not in_span(pd, v):
                dom.append(v); pd = reduce_basis(dom)
                for w in V0_BASIS:
                    if not in_span(pe, w):
                        img.append(w); pe = reduce_basis(img); break
        assert len(dom) == len(img)
    pd, pe = reduce_basis(dom), reduce_basis(img)
    for i in range(16):
        v = 1 << i
        if len(dom) == 16:
            break
        if not in_span(pd, v):
            dom.append(v); pd = reduce_basis(dom)
            for j in range(16):
                w = 1 << j
                if not in_span(pe, w):
                    img.append(w); pe = reduce_basis(img); break
    assert len(dom) == 16 and rank(img) == 16
    return solve_matrix(dom, img)


def find_map(F, G, ea):
    S1 = graph(F)
    for c in graph(G).tolist():
        s = (EASearch if ea else Search)(S1, graph(G) ^ c)
        if s.run():
            basis, Qspan = s.solution
            images = [int(Qspan[1 << j]) for j in range(len(basis))]
            return extend([int(b) for b in basis], images, ea), int(c)
    return None, None


# ---------------- independent verifier ----------------
def verify_cert(F, G, cols, c, ea):
    if rank(cols) != 16:
        return False
    gF = [int(v) for v in graph(F)]
    gG = sorted(int(v) for v in graph(G))
    if sorted(apply(cols, v) ^ c for v in gF) != gG:
        return False
    if ea:
        if any(apply(cols, v) & 255 for v in V0_BASIS):
            return False
    return True


def orbit_rep(n):
    best, x = n, n
    for _ in range(16):
        x = 2 * x % N
        best = min(best, x, N - x)
    return best


if __name__ == "__main__":
    out = {"vector_encoding": "x | (y << 8) for the point (x, y); M given by columns M e_i, i = 0..15",
           "ea_merges": [], "ccz_extra": [], "controls": {}}
    # EA merges that reduce 1024 affine classes to 1020 EA classes (exp9b) plus the CCZ map of exp9
    for a, b in [(253, 259), (1783, 1801), (253, 1783), (763, 773)]:
        cols, c = find_map(T[a], T[b], ea=True)
        ok = cols is not None and verify_cert(T[a], T[b], cols, c, ea=True)
        out["ea_merges"].append({"F": a, "G": b, "c": c, "M_columns": cols, "verified": ok})
        print("EA", a, b, ok, flush=True)
    cols, c = find_map(T[253], T[1783], ea=False)
    out["ccz_extra"].append({"F": 253, "G": 1783, "c": c, "M_columns": cols,
                             "verified": verify_cert(T[253], T[1783], cols, c, ea=False)})
    # negative control for the verifier: a corrupted certificate must fail
    bad = list(out["ea_merges"][0]["M_columns"]); bad[3] ^= 1 << 9
    out["controls"]["corrupted_certificate_rejected"] = not verify_cert(T[253], T[259], bad, out["ea_merges"][0]["c"], True)
    out["controls"]["wrong_pair_rejected"] = not verify_cert(T[253], T[773], out["ea_merges"][0]["M_columns"],
                                                             out["ea_merges"][0]["c"], True)

    # orbits, EA classes and inversion pairing
    units = [n for n in range(1, N) if gcd(n, N) == 1]
    reps = sorted({orbit_rep(n) for n in units})
    assert len(reps) == 1024
    parent = {r: r for r in reps}
    def find(r):
        while parent[r] != r:
            r = parent[r]
        return r
    def union(a, b):
        parent[find(orbit_rep(a))] = find(orbit_rep(b))
    for m in out["ea_merges"]:
        assert m["verified"]; union(m["F"], m["G"])
    ea_classes = len({find(r) for r in reps})
    swap_ok = True
    for r in reps:
        ri = pow(r, -1, N)
        inv_table = np.zeros(256, dtype=np.int64); inv_table[T[r]] = X
        swap_ok &= bool(np.array_equal(inv_table, T[ri]))
        union(r, ri)
    ccz_upper = len({find(r) for r in reps})
    out.update({"orbits": len(reps), "ea_classes_from_certificates": ea_classes,
                "inverse_table_equals_D_n_inverse_for_all_orbits": swap_ok,
                "ccz_classes_upper_bound_from_certificates": ccz_upper,
                "self_inverse_orbits": sum(1 for r in reps if orbit_rep(pow(r, -1, N)) == r)})
    json.dump(out, open("results/exp10_certificates.json", "w"), indent=1)
    print({k: v for k, v in out.items() if k not in ("ea_merges", "ccz_extra")})
