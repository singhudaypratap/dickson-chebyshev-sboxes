"""Experiment 3: chosen-plaintext reduction (Vithayathil-Sreekumar style) against four cipher variants,
plus determinism leak and standard statistical metrics for the nonce-based design.

Attack model (chosen plaintext, as in the published reductions): the attacker queries an encryption
oracle under the unknown key, sees each returned ciphertext and nonce, then must decrypt a target
ciphertext of an unknown image. For fresh-nonce variants, every oracle call uses a fresh nonce.
Attack steps (assumes a public S and a single forward chaining pass, which is what the reduction exploits):
  1. all-zero image -> equivalent key e_i = S^{-1}(C_i) ^ C_{i-1}          (C_{-1} := 0)
  2. ceil(log_256 L) index-encoding images -> permutation pi
  3. target: x_i = S^{-1}(C*_i) ^ e_i ^ C*_{i-1};  P*[pi(i)] = x_i
"""
import json, math, os, time
import numpy as np
from scipy.stats import chi2
from skimage import data
import cipher as cp

cfg = json.load(open("config.json"))
exp = cfg["experiments"]
rng = np.random.default_rng(exp["seed"] + 3)
T = np.load("results/dickson_table_a1.npy")
s1 = json.load(open("results/exp1_summary.json"))
best_n = s1["best_classes"][0]["rep_n"]
SBOX = T[best_n].astype(np.int64)
V = cp.make_variants(SBOX, cfg["cipher"]["perm_mode"])
KEY = bytes(rng.integers(0, 256, cfg["cipher"]["key_bytes"], dtype=np.uint8))


def load_images(size):
    imgs = {}
    for name in exp["image_names"]:
        im = getattr(data, name)()
        if im.ndim == 3:   # ITU-R BT.601 luma, rounded
            im = np.rint(0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2])
        im = im.astype(np.float64)
        assert im.shape == (512, 512), (name, im.shape)
        f = 512 // size
        im = im.reshape(size, f, size, f).mean(axis=(1, 3))   # block mean, f = 1 or 2
        imgs[name] = np.rint(im).astype(np.uint8)
    return imgs


def attack(cipher, target_ct, target_pt, shape):
    S = cipher.S; Si = cipher.Sinv
    L = int(np.prod(shape))
    oracle = (lambda im: cipher.encrypt(KEY, im)[0])
    queries = 0
    C0 = oracle(np.zeros(shape, np.uint8)).ravel().tolist(); queries += 1
    e = [Si[C0[0]]] + [Si[C0[i]] ^ C0[i - 1] for i in range(1, L)]

    def undiffuse(C):
        C = C.ravel().tolist()
        return [Si[C[0]] ^ e[0]] + [Si[C[i]] ^ e[i] ^ C[i - 1] for i in range(1, L)]

    ndig = math.ceil(math.log(L, 256))
    idx = np.arange(L)
    pi = np.zeros(L, dtype=np.int64)
    for t in range(ndig):
        probe = ((idx // 256 ** t) % 256).astype(np.uint8).reshape(shape)
        x = np.array(undiffuse(oracle(probe)), dtype=np.int64); queries += 1
        pi += x * 256 ** t
    pi_valid = bool(np.unique(pi).size == L and pi.max() < L)
    x_t = np.array(undiffuse(target_ct), dtype=np.int64)
    if pi_valid:
        rec = np.zeros(L, dtype=np.int64); rec[pi] = x_t
    else:
        rec = np.full(L, -1)
    full = float(np.mean(rec == target_pt.ravel()))
    return {"chosen_images": queries, "perm_recovered_valid": pi_valid, "plaintext_recovered_fraction": full}


def entropy(im):
    p = np.bincount(im.ravel(), minlength=256) / im.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def chi_square(im):
    h = np.bincount(im.ravel(), minlength=256)
    e = im.size / 256
    return float(((h - e) ** 2 / e).sum())


def adj_corr(im):
    im = im.astype(np.float64)
    pairs = {"H": (im[:, :-1], im[:, 1:]), "V": (im[:-1, :], im[1:, :]), "D": (im[:-1, :-1], im[1:, 1:])}
    return {k: float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) for k, (a, b) in pairs.items()}


def npcr_uaci(c1, c2):
    d = c1.astype(np.int64) - c2.astype(np.int64)
    return float(np.mean(d != 0) * 100), float(np.mean(np.abs(d)) / 255 * 100)


def ideal_npcr_uaci():
    """Expected values for two independent uniform bytes X, Y:
    P(X != Y) = 255/256;  E|X - Y| = (256^2 - 1)/(3*256) = 85.33..., so E[UACI] = 85.33/255 = 257/768."""
    return 255 / 256 * 100, (256 ** 2 - 1) / (3 * 256) / 255 * 100


out = {"best_dickson_n": best_n, "sbox_profile": s1["best_classes"][0], "attacks": [], "determinism": [],
       "stats": [], "sensitivity": [], "timing_s": {}}
for size in exp["image_sizes"]:
    imgs = load_images(size)
    L = size * size
    for name, im in imgs.items():
        for vname, ci in V.items():
            t0 = time.time()
            ct, nonce = ci.encrypt(KEY, im)
            out["timing_s"].setdefault(f"{vname}_{size}", []).append(time.time() - t0)
            assert np.array_equal(ci.decrypt(KEY, ct, nonce), im), "decryption failed"
            a = attack(ci, ct, im, im.shape)
            out["attacks"].append({"image": name, "size": size, "variant": vname, **a})
            ct2, _ = ci.encrypt(KEY, im)
            out["determinism"].append({"image": name, "size": size, "variant": vname,
                                       "equal_pixel_fraction_two_encryptions": float(np.mean(ct == ct2))})
            if vname in ("B1_dickson_2pass_2round_fixedkey", "PROP_dickson_2pass_2round_nonce"):
                out["stats"].append({"image": name, "size": size, "variant": vname,
                                     "entropy_plain": entropy(im), "entropy_cipher": entropy(ct),
                                     "chi2_cipher": chi_square(ct), "corr_plain": adj_corr(im),
                                     "corr_cipher": adj_corr(ct)})
            print(size, name, vname, out["attacks"][-1]["plaintext_recovered_fraction"], flush=True)
        # sensitivity tests on the proposed cipher with a fixed nonce (so only one input changes)
        ci = V["PROP_dickson_2pass_2round_nonce"]
        N0 = bytes(rng.integers(0, 256, 12, dtype=np.uint8))
        base, _ = ci.encrypt(KEY, im, N0)
        for t in range(exp["npcr_trials_per_image"]):
            r, c = (int(v) for v in rng.integers(0, size, 2))
            im2 = im.copy(); im2[r, c] = (int(im2[r, c]) + 1) % 256
            p_n, p_u = npcr_uaci(base, ci.encrypt(KEY, im2, N0)[0])
            kb = bytearray(KEY); bit = int(rng.integers(0, 8 * len(KEY))); kb[bit // 8] ^= 1 << (bit % 8)
            k_n, k_u = npcr_uaci(base, ci.encrypt(bytes(kb), im, N0)[0])
            nb = bytearray(N0); bit = int(rng.integers(0, 96)); nb[bit // 8] ^= 1 << (bit % 8)
            n_n, n_u = npcr_uaci(base, ci.encrypt(KEY, im, bytes(nb))[0])
            out["sensitivity"].append({"image": name, "size": size, "trial": t, "pixel": [r, c],
                                       "plain_NPCR": p_n, "plain_UACI": p_u, "key_NPCR": k_n, "key_UACI": k_u,
                                       "nonce_NPCR": n_n, "nonce_UACI": n_u})
out["ideal_NPCR_UACI_percent"] = ideal_npcr_uaci()
out["chi2_critical_255_0.05"] = float(chi2.ppf(0.95, 255))
out["timing_s"] = {k: float(np.mean(v)) for k, v in out["timing_s"].items()}
json.dump(out, open("results/exp3_summary.json", "w"), indent=1)
print("done")
