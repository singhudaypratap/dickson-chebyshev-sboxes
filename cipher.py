"""One parametrised image cipher; every variant in the paper is a configuration of this code.

Per round r = 1..R (all subkeys from PRF = HMAC-SHA256(K, label || nonce || counter)):
  1. pixel permutation pi_r: perm_mode 'sort' (used in the paper) = stable argsort of 64-bit PRF words;
     perm_mode 'fisher_yates' = Fisher-Yates with rejection sampling        x = P[pi_r]
  2. forward chaining   c_i = S(x_i ^ k1_i ^ c_{i-1}),   c_{-1} = iv1
  3. backward chaining  d_i = S(c_i ^ k2_i ^ d_{i+1}),   d_{L} = iv2      (if backward)
Variants (see VARIANTS): S in {identity, Dickson}, R, backward on/off, nonce fixed or fresh per image.
"""
import hmac, hashlib, os
import numpy as np

FIXED_NONCE = bytes(12)


def prf_stream(key: bytes, nonce: bytes, label: str):
    ctr = 0
    while True:
        block = hmac.new(key, label.encode() + nonce + ctr.to_bytes(8, "big"), hashlib.sha256).digest()
        yield from block
        ctr += 1


def prf_bytes(key, nonce, label, n):
    """First n bytes of the PRF stream (block-wise; identical output to iterating prf_stream)."""
    out, ctr = bytearray(), 0
    while len(out) < n:
        out += hmac.new(key, label.encode() + nonce + ctr.to_bytes(8, "big"), hashlib.sha256).digest()
        ctr += 1
    return list(out[:n])


def prf_sort_permutation(key, nonce, label, L):
    """Permutation = stable argsort of L big-endian 64-bit PRF words (ties broken by index).
    Its distance from uniform is at most the tie probability, below L^2 / 2^65. Unlike Fisher-Yates it can
    be computed and applied with a data-independent sorting network (constant-time implementation)."""
    r = np.frombuffer(bytes(prf_bytes(key, nonce, label, 8 * L)), dtype=">u8")
    return np.argsort(r, kind="stable").tolist()


def prf_permutation(key, nonce, label, L):
    g = prf_stream(key, nonce, label)
    perm = list(range(L))
    for i in range(L - 1, 0, -1):
        m = i + 1
        limit = (1 << 32) - ((1 << 32) % m)
        while True:
            r = (next(g) << 24) | (next(g) << 16) | (next(g) << 8) | next(g)
            if r < limit:
                break
        j = r % m
        perm[i], perm[j] = perm[j], perm[i]
    return perm


class ImageCipher:
    def __init__(self, sbox, rounds: int, backward: bool, fresh_nonce: bool, perm_mode: str):
        sbox = [int(v) for v in sbox]
        assert sorted(sbox) == list(range(256)), "S must be a permutation"
        self.S = sbox
        self.Sinv = [0] * 256
        for x, y in enumerate(sbox):
            self.Sinv[y] = x
        self.R, self.backward, self.fresh_nonce = rounds, backward, fresh_nonce
        if perm_mode not in ("fisher_yates", "sort"):
            raise ValueError(perm_mode)
        self.perm_mode = perm_mode

    def _subkeys(self, key, nonce, L):
        ks = []
        for r in range(self.R):
            iv = prf_bytes(key, nonce, f"r{r}-iv", 2)
            pf = prf_sort_permutation if self.perm_mode == "sort" else prf_permutation
            ks.append({"perm": pf(key, nonce, f"r{r}-perm", L),
                       "k1": prf_bytes(key, nonce, f"r{r}-k1", L),
                       "k2": prf_bytes(key, nonce, f"r{r}-k2", L) if self.backward else None,
                       "iv1": iv[0], "iv2": iv[1]})
        return ks

    def encrypt(self, key: bytes, img: np.ndarray, nonce: bytes = None):
        assert img.dtype == np.uint8 and img.ndim == 2
        if self.fresh_nonce:
            nonce = os.urandom(12) if nonce is None else nonce
        else:
            assert nonce is None, "deterministic variant takes no nonce"
            nonce = FIXED_NONCE
        L = img.size
        S = self.S
        x = img.ravel().tolist()
        for sk in self._subkeys(key, nonce, L):
            p, k1 = sk["perm"], sk["k1"]
            x = [x[p[i]] for i in range(L)]
            prev = sk["iv1"]
            c = [0] * L
            for i in range(L):
                prev = S[x[i] ^ k1[i] ^ prev]
                c[i] = prev
            if self.backward:
                k2 = sk["k2"]; prev = sk["iv2"]
                d = [0] * L
                for i in range(L - 1, -1, -1):
                    prev = S[c[i] ^ k2[i] ^ prev]
                    d[i] = prev
                c = d
            x = c
        return np.array(x, dtype=np.uint8).reshape(img.shape), nonce

    def decrypt(self, key: bytes, ct: np.ndarray, nonce: bytes):
        L = ct.size
        Si = self.Sinv
        y = ct.ravel().tolist()
        for sk in reversed(self._subkeys(key, nonce, L)):
            if self.backward:
                k2 = sk["k2"]
                c = [0] * L
                nxt = sk["iv2"]
                for i in range(L - 1, -1, -1):
                    c[i] = Si[y[i]] ^ k2[i] ^ nxt
                    nxt = y[i]
                y = c
            k1 = sk["k1"]
            x = [0] * L
            prev = sk["iv1"]
            for i in range(L):
                x[i] = Si[y[i]] ^ k1[i] ^ prev
                prev = y[i]
            p = sk["perm"]
            out = [0] * L
            for i in range(L):
                out[p[i]] = x[i]
            y = out
        return np.array(y, dtype=np.uint8).reshape(ct.shape)


TAG_BYTES = 12


def siv_tag(key: bytes, nonce: bytes, img: np.ndarray) -> bytes:
    """Synthetic IV (Rogaway-Shrimpton SIV pattern): T = F_{K_mac}(N || H || W || P), truncated to 96 bits.
    K_mac = HMAC(K, 'mac-key'); its 7-byte input never collides with subkey inputs (>= 25 bytes)."""
    assert img.dtype == np.uint8 and img.ndim == 2 and len(nonce) == 12
    kmac = hmac.new(key, b"mac-key", hashlib.sha256).digest()
    msg = nonce + img.shape[0].to_bytes(4, "big") + img.shape[1].to_bytes(4, "big") + img.tobytes()
    return hmac.new(kmac, msg, hashlib.sha256).digest()[:TAG_BYTES]


class SIVImageCipher:
    """Misuse-resistant variant: the per-image tweak is T = siv_tag(K, N, P) instead of N itself.
    Ciphertext = (N, T, d). Decryption recomputes T from the recovered image and rejects on mismatch."""

    def __init__(self, sbox, rounds: int, perm_mode: str):
        self.core = ImageCipher(sbox, rounds=rounds, backward=True, fresh_nonce=True, perm_mode=perm_mode)

    def encrypt(self, key, img, nonce=None):
        nonce = os.urandom(12) if nonce is None else nonce
        tag = siv_tag(key, nonce, img)
        ct, _ = self.core.encrypt(key, img, nonce=tag)
        return ct, nonce, tag

    def decrypt(self, key, ct, nonce, tag):
        """Returns the image, or None if authentication fails (malformed inputs are rejected, not raised)."""
        if not (isinstance(nonce, (bytes, bytearray)) and len(nonce) == 12 and isinstance(tag, (bytes, bytearray))
                and len(tag) == 12 and getattr(ct, "ndim", 0) == 2):
            return None
        p = self.core.decrypt(key, ct, tag)
        return p if hmac.compare_digest(siv_tag(key, nonce, p), tag) else None


def make_variants(dickson_sbox, perm_mode):
    ident = list(range(256))
    return {
        "B0_xor_fixedkey": ImageCipher(ident, rounds=1, backward=False, fresh_nonce=False, perm_mode=perm_mode),
        "B0S_dickson_fixedkey": ImageCipher(dickson_sbox, rounds=1, backward=False, fresh_nonce=False, perm_mode=perm_mode),
        "B1_dickson_2pass_2round_fixedkey": ImageCipher(dickson_sbox, rounds=2, backward=True, fresh_nonce=False, perm_mode=perm_mode),
        "PROP_dickson_2pass_2round_nonce": ImageCipher(dickson_sbox, rounds=2, backward=True, fresh_nonce=True, perm_mode=perm_mode),
    }
