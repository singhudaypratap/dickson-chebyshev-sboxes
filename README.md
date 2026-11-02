# Dickson permutations over binary fields and chaos-based image ciphers: code and data

Code and computed results accompanying

> U. P. Singh, "Dickson permutations of binary fields as S-boxes: a near-linear block, equivalence classes over F_256, and consequences for Chebyshev-based ciphers" (manuscript under review).

Every number in the paper is produced by the scripts below; the `results/` folder holds the outputs used in the manuscript.

## Environment

Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, scikit-image 0.26.0, matplotlib 3.11.2 (`pip install -r requirements.txt`);
GCC 13.3 and OpenSSL 3.0.13 for the C implementation. Random seeds are fixed in `config.json` (seed 20261031).

## Run order

Run `exp1_dickson_profile.py` first: it creates `results/dickson_table_a1.npy` (the table D_n(x,1) over F_256, about 16 MB, not stored here), which later scripts read. All scripts are run from the repository root.

```
python3 exp1_dickson_profile.py
python3 exp1b_function_identity.py
python3 exp1c_weak_classes.py
# ... then any of the scripts listed below
```

## Script-to-paper map

| Script | Paper |
| --- | --- |
| `gf256.py`, `dickson.py`, `sbox_metrics.py` | Field arithmetic, Dickson evaluators, S-box invariants (Sections 3, 5.1) |
| `exp1_dickson_profile.py` | Table 1, Figure 1; Lemmas 4.2-4.3 checks |
| `exp1b_function_identity.py` | Proposition 4.1 (coincidences) |
| `exp1c_weak_classes.py` | Proposition 5.1 (weak block of F_256) |
| `exp2_finite_state.py` | Table 3, Proposition 4.5 |
| `exp3_cipher_attacks.py` | Table 4 (published chosen-plaintext reduction) |
| `exp4_collapse.py` | Proposition 4.4 (layer collapse) |
| `exp5_general_m.py`, `exp5b_inversion_reference.py`, `exp5c_m11.py` | Table 2, Theorem 5.2, Conjecture 5.3 (m = 3..11) |
| `exp6_equivalence.py`, `exp6b_affine.py`, `exp6c_ea_invariance_check.py`, `exp8_flats_ea.py`, `exp9_ccz_search.py`, `exp9b_ea_search.py`, `exp9c_negative_controls.py` | Section 5.5 (linear, affine, EA and CCZ classes) |
| `cipher.py`, `exp7_siv_and_stats.py` | Sections 7.4-7.5 (SIV mode, misuse, tampering, statistics) |
| `ct/ct_siv.c` | Section 7.6, Table 7 (constant-time implementation, timing-leakage tests) |
| `server_jobs/gpu_dickson_profile.py` | Extension of Table 2 to m = 12, 13 (GPU, PyTorch; numpy backend for checking) |
| `server_jobs/nist_sts_siv.py` | NIST SP 800-22 battery on SIV ciphertexts with a PCG64 positive control (requires `nistrng` 1.2.3; each test gets its own copy of the bits because that version's matrix-rank test overwrites its input) |
| `server_jobs/gen_sts_input.py`, `server_jobs/run_nist_ref.sh` | NIST SP 800-22 with the NIST reference code sts-2.1.2 (downloaded from csrc.nist.gov) on the same sequences; reports in `results/nist_ref_*.txt` |
| `exp10_certificates.py` | Equivalence certificates: 16x16 matrices over F_2 for every EA/CCZ merge, an independent verifier, and replay of the 1 024 -> 1 020 -> 513 class counts |
| `exp11_gpu_crosscheck.py` | CPU re-check of the GPU runs: primitivity of the field polynomials and re-profiling of the best m = 12, 13 representatives with independent code |

## Constant-time C implementation

```
cd ct
gcc -O2 -o ct_siv ct_siv.c -lcrypto -lm
./ct_siv sboxcheck
./ct_siv dudect_sbox ct 1000000
./ct_siv dudect_cipher plain 64 20000
```

The C file implements encryption only; decryption with constant-time tag comparison is in `cipher.py`. Any image size is accepted (the sorting network is padded to a power of two); the C output equals the Python reference byte for byte.

## Reproducibility manifest

`MANIFEST.sha256` lists a SHA-256 hash for every file in this release, and `ENVIRONMENT.txt` records the versions used. Check with `sha256sum -c MANIFEST.sha256`.

## Licence

No licence file is included yet; the author will add one. Until then, all rights are reserved by the author.
