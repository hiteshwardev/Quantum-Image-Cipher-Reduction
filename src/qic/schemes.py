"""Cipher constructions studied in the paper.

A plaintext-independent construction is an ordered list of layers acting on
the value and position registers of an NEQR state. The same list is executed
along three independent code paths:

* ``encrypt`` acts on the classical pixel array (the classical reduction);
* ``encrypt_state`` acts on a dense state vector through the gates of
  :class:`qic.neqr.StateVector`;
* ``pair_map`` composes the gate tables of :mod:`qic.normal_form`.

Agreement of the first two paths verifies representation independence, and
factorising the third verifies the normal form of Theorem 1.

Constructions
-------------
A  logistic XOR diffusion, then a keyed Arnold position permutation
B  bit-plane (value-qubit) permutation, Arnold permutation, logistic XOR
C  scheme A with the diffusion seed derived from a hash of the plaintext
D  logistic modular-addition diffusion, then an Arnold permutation
E  an independent key-derived substitution box at every position, then Arnold
F  an independent Arnold permutation of every bit plane, then logistic XOR

The constructions follow the operation templates of the published schemes
cited in the paper; they are not bit-for-bit reproductions of any one scheme.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import keystream as ks
from . import normal_form as nf
from .neqr import StateVector, from_bit_planes, to_bit_planes

LAYER_KINDS = ("xor", "add", "substitute", "permute_values", "permute_positions")


@dataclass(frozen=True)
class Layer:
    """One keyed gate: ``kind`` names the operation and ``data`` holds its key material."""

    kind: str
    data: np.ndarray

    def __post_init__(self):
        if self.kind not in LAYER_KINDS:
            raise ValueError(f"unknown layer kind {self.kind!r}")


def _bit_lut(bit_perm: np.ndarray, q: int) -> np.ndarray:
    """Lookup table of the value map whose output bit k is input bit ``bit_perm[k]``."""
    values = np.arange(2 ** q, dtype=np.int64)
    out = np.zeros_like(values)
    for new_bit, old_bit in enumerate(np.asarray(bit_perm, dtype=np.int64)):
        out |= ((values >> old_bit) & 1) << new_bit
    return out


class LayeredCipher:
    """A plaintext-independent cipher given by an ordered list of layers."""

    def __init__(self, name: str, layers: list[Layer], n_pix: int, q: int):
        self.name = name
        self.layers = list(layers)
        self.n_pix = int(n_pix)
        self.q = int(q)
        self.levels = 2 ** self.q

    # pixel-array path -------------------------------------------------------

    def _forward(self, img: np.ndarray, layer: Layer) -> np.ndarray:
        d = layer.data
        if layer.kind == "xor":
            return img ^ d
        if layer.kind == "add":
            return (img + d) % self.levels
        if layer.kind == "substitute":
            return d[np.arange(self.n_pix), img].astype(np.int64)
        if layer.kind == "permute_values":
            return _bit_lut(d, self.q)[img]
        out = np.zeros_like(img)
        out[d] = img
        return out

    def _backward(self, img: np.ndarray, layer: Layer) -> np.ndarray:
        d = layer.data
        if layer.kind == "xor":
            return img ^ d
        if layer.kind == "add":
            return (img - d) % self.levels
        if layer.kind == "substitute":
            inverse = np.empty(d.shape, dtype=np.int64)
            rows = np.arange(self.n_pix)[:, None]
            inverse[rows, d.astype(np.int64)] = np.arange(self.levels)[None, :]
            return inverse[np.arange(self.n_pix), img]
        if layer.kind == "permute_values":
            return _bit_lut(np.argsort(d), self.q)[img]
        return img[d]

    def encrypt(self, image: np.ndarray) -> np.ndarray:
        """Encrypt a flattened image on the classical pixel array."""
        img = np.asarray(image, dtype=np.int64).reshape(-1)
        for layer in self.layers:
            img = self._forward(img, layer)
        return img

    def decrypt(self, cipher: np.ndarray) -> np.ndarray:
        img = np.asarray(cipher, dtype=np.int64).reshape(-1)
        for layer in reversed(self.layers):
            img = self._backward(img, layer)
        return img

    def oracle(self):
        """Chosen-plaintext encryption oracle under the fixed key."""
        return self.encrypt

    # state-vector path ------------------------------------------------------

    def encrypt_state(self, state: StateVector) -> StateVector:
        """Apply the layers as unitaries to a dense state, in place."""
        for layer in self.layers:
            d = layer.data
            if layer.kind == "xor":
                state.xor_values(d)
            elif layer.kind == "add":
                state.add_values(d)
            elif layer.kind == "substitute":
                state.substitute_values(d)
            elif layer.kind == "permute_values":
                state.permute_value_qubits(d)
            else:
                state.permute_positions(d)
        return state

    # pair-map path ----------------------------------------------------------

    def pair_map(self) -> nf.PairMap:
        """Permutation of (value, position) pairs induced by the circuit."""
        pm = nf.identity(self.n_pix, self.q)
        for layer in self.layers:
            d = layer.data
            if layer.kind == "xor":
                gate = nf.xor_layer(d, self.q)
            elif layer.kind == "add":
                gate = nf.addition_layer(d, self.q)
            elif layer.kind == "substitute":
                gate = nf.substitution_layer(d, self.q)
            elif layer.kind == "permute_values":
                gate = nf.bitplane_layer(d, self.n_pix, self.q)
            else:
                gate = nf.position_permutation(d, self.q)
            pm = nf.compose(pm, gate)
        return pm

    def normal_form(self) -> tuple[np.ndarray, np.ndarray]:
        """Ground-truth ``(pi, phi)`` obtained by factorising the pair map."""
        return nf.factorize(self.pair_map())


class PlaintextDependentCipher:
    """Scheme C: scheme A with a diffusion seed hashed from the plaintext.

    The receiver needs the digest to rebuild the keystream, so the digest
    travels with the cipher image. The classical encryption oracle returns
    only the cipher image.
    """

    def __init__(self, key: int, n_pix: int, q: int = 8):
        self.name = "C"
        self.key = int(key)
        self.n_pix = int(n_pix)
        self.q = int(q)
        self.pi = ks.keyed_arnold(key, n_pix, "C/arnold")

    def schedule(self, digest: bytes) -> LayeredCipher:
        stream = ks.logistic_stream(ks.digest_unit(digest), self.n_pix, self.q)
        layers = [Layer("xor", stream), Layer("permute_positions", self.pi)]
        return LayeredCipher("C", layers, self.n_pix, self.q)

    def digest(self, image: np.ndarray) -> bytes:
        return ks.plaintext_digest(self.key, image, "C/stream")

    def encrypt_with_digest(self, image: np.ndarray) -> tuple[np.ndarray, bytes]:
        d = self.digest(image)
        return self.schedule(d).encrypt(image), d

    def encrypt(self, image: np.ndarray) -> np.ndarray:
        return self.encrypt_with_digest(image)[0]

    def decrypt(self, cipher: np.ndarray, digest: bytes) -> np.ndarray:
        return self.schedule(digest).decrypt(cipher)

    def oracle(self):
        return self.encrypt


class BitPlanePermutationCipher:
    """Scheme F: an independent position permutation of every bit plane, then XOR.

    Changing one pixel can change up to ``q`` cipher pixels, so no unitary
    realises this cipher on NEQR. On the bit-plane representation (BRQI), where
    every bit occupies its own position, the cipher is a layered circuit with
    ``L = 2`` and ``q N`` positions; :meth:`bit_level_cipher` returns it.
    """

    def __init__(self, key: int, n_pix: int, q: int = 8):
        self.name = "F"
        self.n_pix = int(n_pix)
        self.q = int(q)
        self.plane_perms = np.stack(
            [ks.keyed_arnold(key, n_pix, f"F/arnold/{k}") for k in range(q)])
        self.stream = ks.logistic_stream(ks.key_unit(key, "F/stream"), n_pix, q)

    def encrypt(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.int64).reshape(-1)
        out = np.zeros_like(img)
        for k in range(self.q):
            moved = np.zeros_like(img)
            moved[self.plane_perms[k]] = (img >> k) & 1
            out |= moved << k
        return out ^ self.stream

    def decrypt(self, cipher: np.ndarray) -> np.ndarray:
        x = np.asarray(cipher, dtype=np.int64).reshape(-1) ^ self.stream
        out = np.zeros_like(x)
        for k in range(self.q):
            out |= ((x >> k) & 1)[self.plane_perms[k]] << k
        return out

    def oracle(self):
        return self.encrypt

    def bit_level_cipher(self) -> LayeredCipher:
        """The same cipher on the bit-plane representation, position ``k N + y``."""
        offsets = (np.arange(self.q, dtype=np.int64) * self.n_pix)[:, None]
        pi_bits = (self.plane_perms + offsets).reshape(-1)
        flips = to_bit_planes(self.stream, self.q)
        layers = [Layer("permute_positions", pi_bits), Layer("xor", flips)]
        return LayeredCipher("F (bit level)", layers, self.q * self.n_pix, 1)

    def encrypt_via_bits(self, image: np.ndarray) -> np.ndarray:
        """Encrypt through the bit-level circuit; equals :meth:`encrypt`."""
        bits = self.bit_level_cipher().encrypt(to_bit_planes(image, self.q))
        return from_bit_planes(bits, self.q)


# factories -----------------------------------------------------------------

def scheme_a(key: int, n_pix: int, q: int = 8) -> LayeredCipher:
    stream = ks.logistic_stream(ks.key_unit(key, "A/stream"), n_pix, q)
    pi = ks.keyed_arnold(key, n_pix, "A/arnold")
    return LayeredCipher("A", [Layer("xor", stream), Layer("permute_positions", pi)], n_pix, q)


def scheme_b(key: int, n_pix: int, q: int = 8) -> LayeredCipher:
    bit_perm = ks.keyed_bit_permutation(key, q, "B/planes")
    pi = ks.keyed_arnold(key, n_pix, "B/arnold")
    stream = ks.logistic_stream(ks.key_unit(key, "B/stream"), n_pix, q)
    layers = [Layer("permute_values", bit_perm), Layer("permute_positions", pi),
              Layer("xor", stream)]
    return LayeredCipher("B", layers, n_pix, q)


def scheme_c(key: int, n_pix: int, q: int = 8) -> PlaintextDependentCipher:
    return PlaintextDependentCipher(key, n_pix, q)


def scheme_d(key: int, n_pix: int, q: int = 8) -> LayeredCipher:
    stream = ks.logistic_stream(ks.key_unit(key, "D/stream"), n_pix, q)
    pi = ks.keyed_arnold(key, n_pix, "D/arnold")
    return LayeredCipher("D", [Layer("add", stream), Layer("permute_positions", pi)], n_pix, q)


def scheme_e(key: int, n_pix: int, q: int = 8) -> LayeredCipher:
    tables = ks.keyed_sboxes(key, n_pix, q, "E/sboxes")
    pi = ks.keyed_arnold(key, n_pix, "E/arnold")
    layers = [Layer("substitute", tables), Layer("permute_positions", pi)]
    return LayeredCipher("E", layers, n_pix, q)


def scheme_f(key: int, n_pix: int, q: int = 8) -> BitPlanePermutationCipher:
    return BitPlanePermutationCipher(key, n_pix, q)


SCHEMES = {"A": scheme_a, "B": scheme_b, "C": scheme_c,
           "D": scheme_d, "E": scheme_e, "F": scheme_f}

DESCRIPTIONS = {
    "A": "Arnold permutation, logistic XOR diffusion",
    "B": "bit-plane permutation, Arnold permutation, logistic XOR",
    "C": "scheme A with a plaintext-hashed diffusion seed",
    "D": "Arnold permutation, logistic modular-addition diffusion",
    "E": "per-position substitution boxes, Arnold permutation",
    "F": "independent per-plane Arnold permutations, logistic XOR",
}
