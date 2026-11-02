"""Experiment 7: misuse-resistant (SIV) variant under nonce reuse and tampering, and an expanded
statistical evaluation with exact null distributions.

Part A (nonce misuse): the encryption oracle uses ONE fixed nonce for every query (worst case).
  A1 the published reduction (all-zero image + index probes) against the SIV variant
  A2 two images differing in one pixel, same nonce: NPCR/UACI
  A3 the same image twice, same nonce: identical ciphertexts (the unavoidable leak of deterministic encryption)
Part B (authenticity): random single-bit flips in d, T or N must be rejected.
Part C (statistics, SIV variant, fresh nonces): 10 images x 2 sizes x 5 keys.
  chi^2 p-values -> KS test for uniformity; adjacent correlations -> z = r sqrt(n_pairs), KS vs N(0,1);
  sensitivity (plaintext pixel, key bit, nonce bit; 100 trials per size each) -> z-scores against the
  exact null of two independent uniform byte images:
    NPCR/100 ~ Binomial(L, 255/256)/L ;  UACI uses E and Var of |X - Y|/255 computed by enumeration.
"""
import json, math, os, time
import numpy as np
from scipy.stats import chi2, kstest, norm
from skimage import data
import cipher as cp

cfg = json.load(open("config.json"))
seed = cfg["experiments"]["seed"]
rng = np.random.default_rng(seed + 7)
T = np.load("results/dickson_table_a1.npy")
best_n = json.load(open("results/exp1_summary.json"))["best_classes"][0]["rep_n"]
SBOX = T[best_n].astype(np.int64)
SIV = cp.SIVImageCipher(SBOX, rounds=cfg["cipher"]["rounds"], perm_mode=cfg["cipher"]["perm_mode"])
KEYS = [bytes(rng.integers(0, 256, 16, dtype=np.uint8)) for _ in range(5)]
N_SENS = 100


def gray512(name):
    im = getattr(data, name)()
    if im.ndim == 3:
        im = np.rint(0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2])
    im = np.asarray(im, dtype=np.float64)
    assert im.shape == (512, 512)
    return im


def image_set(size):
    base = {n: gray512(n) for n in ["camera", "moon", "brick", "grass", "gravel", "astronaut"]}
    yy, xx = np.mgrid[0:512, 0:512]
    base["zeros"] = np.zeros((512, 512)); base["ones255"] = np.full((512, 512), 255.0)
    base["gradient"] = ((xx + yy) // 4 % 256).astype(np.float64)
    base["noise"] = np.random.default_rng(seed + 70).integers(0, 256, (512, 512)).astype(np.float64)
    f = 512 // size
    return {k: np.rint(v.reshape(size, f, size, f).mean(axis=(1, 3))).astype(np.uint8) for k, v in base.items()}


# exact null moments for one pixel pair of independent uniform bytes
_d = np.abs(np.arange(256)[:, None] - np.arange(256)[None, :]) / 255.0
U_MEAN, U_VAR = float(_d.mean()), float(_d.var())
P_NE = 255 / 256


def npcr_uaci(c1, c2):
    d = np.abs(c1.astype(np.int64) - c2.astype(np.int64))
    return float(np.mean(d != 0)), float(np.mean(d) / 255.0)


def z_npcr(v, L): return (v - P_NE) / math.sqrt(P_NE * (1 - P_NE) / L)
def z_uaci(v, L): return (v - U_MEAN) / math.sqrt(U_VAR / L)


def chi_p(ct):
    h = np.bincount(ct.ravel(), minlength=256); e = ct.size / 256
    s = float(((h - e) ** 2 / e).sum())
    return s, float(chi2.sf(s, 255))


def corr_z(ct):
    im = ct.astype(np.float64); out = {}
    for k, (a, b) in {"H": (im[:, :-1], im[:, 1:]), "V": (im[:-1, :], im[1:, :]), "D": (im[:-1, :-1], im[1:, 1:])}.items():
        r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]); out[k] = (r, r * math.sqrt(a.size))
    return out


def attack_fixed_nonce(siv, key, nonce, target_ct, target_pt, shape):
    """Published reduction with every oracle query under the same nonce."""
    Si = siv.core.Sinv; L = int(np.prod(shape))
    oracle = lambda im: siv.encrypt(key, im, nonce)[0]
    C0 = oracle(np.zeros(shape, np.uint8)).ravel().tolist()
    e = [Si[C0[0]]] + [Si[C0[i]] ^ C0[i - 1] for i in range(1, L)]
    und = lambda C: [Si[C[0]] ^ e[0]] + [Si[C[i]] ^ e[i] ^ C[i - 1] for i in range(1, L)]
    idx = np.arange(L); pi = np.zeros(L, dtype=np.int64)
    for t in range(math.ceil(math.log(L, 256))):
        probe = ((idx // 256 ** t) % 256).astype(np.uint8).reshape(shape)
        pi += np.array(und(oracle(probe).ravel().tolist()), dtype=np.int64) * 256 ** t
    valid = bool(np.unique(pi).size == L and pi.max() < L)
    x_t = np.array(und(target_ct.ravel().tolist()), dtype=np.int64)
    if valid:
        rec = np.zeros(L, dtype=np.int64); rec[pi] = x_t
        return valid, float(np.mean(rec == target_pt.ravel()))
    return valid, None


out = {"best_n": best_n, "null": {"P_ne": P_NE, "UACI_mean": U_MEAN, "UACI_var_per_pixel": U_VAR},
       "A_attack": [], "A_npcr_same_nonce": [], "A_identical_same_nonce": [], "B_tamper": {}, "C_stats": [],
       "C_sens": [], "timing": {}}
t0 = time.time()
for size in (256, 512):
    imgs = image_set(size); L = size * size
    # ---- Part A
    for name in ["camera", "moon", "brick", "astronaut"]:
        key = KEYS[0]; N0 = bytes(rng.integers(0, 256, 12, dtype=np.uint8))
        im = imgs[name]
        ct, n, tag = SIV.encrypt(key, im, N0)
        assert np.array_equal(SIV.decrypt(key, ct, n, tag), im)
        valid, frac = attack_fixed_nonce(SIV, key, N0, ct, im, im.shape)
        out["A_attack"].append({"size": size, "image": name, "perm_valid": valid, "recovered": frac})
        ct2, _, _ = SIV.encrypt(key, im, N0)
        out["A_identical_same_nonce"].append(bool(np.array_equal(ct, ct2)))
        for t in range(10):
            r, c = (int(v) for v in rng.integers(0, size, 2))
            im2 = im.copy(); im2[r, c] = (int(im2[r, c]) + 1) % 256
            v_n, v_u = npcr_uaci(ct, SIV.encrypt(key, im2, N0)[0])
            out["A_npcr_same_nonce"].append({"size": size, "image": name, "NPCR": v_n, "UACI": v_u,
                                             "zN": z_npcr(v_n, L), "zU": z_uaci(v_u, L)})
    # ---- Part B (tamper), on 64 trials per field per size using camera
    im = imgs["camera"]; key = KEYS[1]
    ct, n, tag = SIV.encrypt(key, im)
    rej = {"ciphertext": 0, "tag": 0, "nonce": 0}; trials = 64
    for _ in range(trials):
        c2 = ct.copy(); i = int(rng.integers(0, L)); c2.ravel()[i] ^= 1 << int(rng.integers(0, 8))
        rej["ciphertext"] += SIV.decrypt(key, c2, n, tag) is None
        tb = bytearray(tag); b = int(rng.integers(0, 96)); tb[b // 8] ^= 1 << (b % 8)
        rej["tag"] += SIV.decrypt(key, ct, n, bytes(tb)) is None
        nb = bytearray(n); b = int(rng.integers(0, 96)); nb[b // 8] ^= 1 << (b % 8)
        rej["nonce"] += SIV.decrypt(key, ct, bytes(nb), tag) is None
    out["B_tamper"][str(size)] = {"trials_per_field": trials, "rejected": rej}
    # ---- Part C statistics
    for name, im in imgs.items():
        for ki, key in enumerate(KEYS):
            t1 = time.time()
            ct, n, tag = SIV.encrypt(key, im)
            out["timing"].setdefault(str(size), []).append(time.time() - t1)
            p = np.bincount(ct.ravel(), minlength=256) / L; p = p[p > 0]
            s, pv = chi_p(ct)
            out["C_stats"].append({"size": size, "image": name, "key": ki, "entropy": float(-(p * np.log2(p)).sum()),
                                   "chi2": s, "chi2_p": pv, "corr": corr_z(ct)})
    names = list(imgs)
    for t in range(N_SENS):
        name = names[t % len(names)]; im = imgs[name]; key = KEYS[t % len(KEYS)]
        N0 = bytes(rng.integers(0, 256, 12, dtype=np.uint8))
        base = SIV.encrypt(key, im, N0)[0]
        r, c = (int(v) for v in rng.integers(0, size, 2))
        im2 = im.copy(); im2[r, c] = (int(im2[r, c]) + 1) % 256
        kb = bytearray(key); b = int(rng.integers(0, 128)); kb[b // 8] ^= 1 << (b % 8)
        nb = bytearray(N0); b2 = int(rng.integers(0, 96)); nb[b2 // 8] ^= 1 << (b2 % 8)
        row = {"size": size, "image": name}
        for lab, ct2 in (("plain", SIV.encrypt(key, im2, N0)[0]), ("key", SIV.encrypt(bytes(kb), im, N0)[0]),
                         ("nonce", SIV.encrypt(key, im, bytes(nb))[0])):
            v_n, v_u = npcr_uaci(base, ct2)
            row[lab] = {"NPCR": v_n, "UACI": v_u, "zN": z_npcr(v_n, L), "zU": z_uaci(v_u, L)}
        out["C_sens"].append(row)
    print("size done", size, round(time.time() - t0), flush=True)

# summaries
summ = {}
for size in (256, 512):
    st = [s for s in out["C_stats"] if s["size"] == size]
    pv = np.array([s["chi2_p"] for s in st])
    zc = np.array([s["corr"][k][1] for s in st for k in "HVD"])
    summ[size] = {"n_ciphertexts": len(st), "entropy_min": min(s["entropy"] for s in st),
                  "entropy_mean": float(np.mean([s["entropy"] for s in st])),
                  "chi2_reject_5pct": int((pv < 0.05).sum()), "chi2_p_KS_uniform_p": float(kstest(pv, "uniform").pvalue),
                  "corr_abs_max": float(max(abs(s["corr"][k][0]) for s in st for k in "HVD")),
                  "corr_z_abs_gt_1.96": int((np.abs(zc) > 1.96).sum()), "corr_z_count": int(zc.size),
                  "corr_z_KS_normal_p": float(kstest(zc, "norm").pvalue)}
    se = [r for r in out["C_sens"] if r["size"] == size]
    for lab in ("plain", "key", "nonce"):
        zN = np.array([r[lab]["zN"] for r in se]); zU = np.array([r[lab]["zU"] for r in se])
        vN = np.array([r[lab]["NPCR"] for r in se]) * 100; vU = np.array([r[lab]["UACI"] for r in se]) * 100
        summ[size][lab] = {"n": len(se), "NPCR_mean": float(vN.mean()), "NPCR_sd": float(vN.std(ddof=1)),
                           "UACI_mean": float(vU.mean()), "UACI_sd": float(vU.std(ddof=1)),
                           "zN_KS_p": float(kstest(zN, "norm").pvalue), "zU_KS_p": float(kstest(zU, "norm").pvalue),
                           "zN_abs_gt_1.96": int((np.abs(zN) > 1.96).sum()), "zU_abs_gt_1.96": int((np.abs(zU) > 1.96).sum())}
    L = size * size
    summ[size]["null_NPCR_mean_pct"] = P_NE * 100
    summ[size]["null_NPCR_sd_pct"] = math.sqrt(P_NE * (1 - P_NE) / L) * 100
    summ[size]["null_UACI_mean_pct"] = U_MEAN * 100
    summ[size]["null_UACI_sd_pct"] = math.sqrt(U_VAR / L) * 100
    an = [r for r in out["A_npcr_same_nonce"] if r["size"] == size]
    summ[size]["misuse_one_pixel_same_nonce"] = {"n": len(an), "NPCR_mean": float(np.mean([r["NPCR"] for r in an]) * 100),
                                                 "UACI_mean": float(np.mean([r["UACI"] for r in an]) * 100),
                                                 "zN_KS_p": float(kstest([r["zN"] for r in an], "norm").pvalue)}
    summ[size]["enc_time_mean_s"] = float(np.mean(out["timing"][str(size)]))
out["summary"] = summ
out["runtime_s"] = round(time.time() - t0, 1)
json.dump(out, open("results/exp7_summary.json", "w"), indent=1)
print(json.dumps({"A_attack": out["A_attack"], "A_identical": out["A_identical_same_nonce"], "B": out["B_tamper"],
                  "summary": summ}, indent=1))
