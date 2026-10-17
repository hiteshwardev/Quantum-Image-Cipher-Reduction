"""Key schedules: chaotic keystreams, keyed Arnold permutations, key derivation.

Every keyed quantity is derived from an integer key and a text label with
SHA-256, so two quantities with different labels are unrelated even under the
same key. A plaintext-dependent schedule additionally mixes a SHA-256 digest of
the plaintext into the derivation.

Value diffusion uses the logistic map ``x <- r x (1 - x)`` with ``r = 3.99``.
A keystream value is extracted as ``k_i = floor(1e14 x_i) mod 2**q``. The
extraction keeps low-order digits of the orbit, which are close to uniform,
instead of ``floor(2**q x_i)``, which inherits the strongly non-uniform
invariant density of the map. The choice is heuristic and is ours; no result
of the study depends on the statistical quality of the stream, because the
attacks recover an equivalent key whatever the stream is.

Position scrambling uses a keyed generalised Arnold cat map on the square pixel
grid,

    (r, c) -> (r + a c, b r + (a b + 1) c)  mod side,

applied ``rounds`` times. The matrix has determinant one, so the map is a
bijection of the grid for every side length. Images whose pixel count is not a
perfect square are rejected rather than silently given another permutation.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .seeding import GLOBAL_SEED


def _digest(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
        h.update(b"|")
    return h.digest()


def key_int(key: int, label: str) -> int:
    """Return a 63-bit integer derived from ``key`` and ``label``."""
    d = _digest(str(int(key)).encode(), label.encode())
    return int.from_bytes(d[:8], "big") >> 1


def key_unit(key: int, label: str) -> float:
    """Return a logistic-map seed in ``[1e-6, 1 - 1e-6]`` derived from the key."""
    d = _digest(str(int(key)).encode(), label.encode())
    x = int.from_bytes(d[:8], "big") / 2.0 ** 64
    return min(max(x, 1e-6), 1.0 - 1e-6)


def plaintext_digest(key: int, image: np.ndarray, label: str) -> bytes:
    """SHA-256 digest of the key, a label and the plaintext pixels."""
    img = np.ascontiguousarray(np.asarray(image, dtype=np.int64).reshape(-1))
    return _digest(str(int(key)).encode(), label.encode(), img.tobytes())


def digest_unit(digest: bytes) -> float:
    """Map a digest to a logistic-map seed in ``[1e-6, 1 - 1e-6]``."""
    x = int.from_bytes(digest[:8], "big") / 2.0 ** 64
    return min(max(x, 1e-6), 1.0 - 1e-6)


def logistic_stream(x0: float, length: int, q: int, r: float = 3.99,
                    burn: int = 1000) -> np.ndarray:
    """Keystream of ``length`` values in ``{0, ..., 2**q - 1}``.

    The orbit starts at ``x0``, discards ``burn`` transient iterates and then
    emits ``floor(1e14 x_i) mod 2**q`` for each further iterate.
    """
    x = float(x0)
    if not 0.0 < x < 1.0:
        raise ValueError("the logistic seed must lie in (0, 1)")
    for _ in range(burn):
        x = r * x * (1.0 - x)
    modulus = 2 ** q
    out = [0] * length
    for i in range(length):
        x = r * x * (1.0 - x)
        out[i] = int(x * 1e14) % modulus
    return np.asarray(out, dtype=np.int64)


def grid_side(n_pix: int) -> int:
    """Side length of a square image with ``n_pix`` pixels."""
    side = int(round(n_pix ** 0.5))
    if side * side != n_pix:
        raise ValueError(f"{n_pix} pixels do not form a square image")
    return side


def arnold_permutation(n_pix: int, a: int, b: int, rounds: int) -> np.ndarray:
    """Position permutation of the generalised Arnold cat map.

    Returns ``pi`` with ``pi[y]`` the destination of pixel ``y`` (row-major
    index) after ``rounds`` applications of the map with parameters ``a, b``.
    """
    side = grid_side(n_pix)
    rows, cols = np.divmod(np.arange(n_pix, dtype=np.int64), side)
    for _ in range(rounds):
        rows, cols = (rows + a * cols) % side, (b * rows + (a * b + 1) * cols) % side
    return rows * side + cols


def keyed_arnold(key: int, n_pix: int, label: str) -> np.ndarray:
    """Arnold permutation whose parameters ``a, b`` and round count derive from the key."""
    side = grid_side(n_pix)
    k = key_int(key, label)
    span = max(side - 1, 1)
    a = 1 + k % span
    b = 1 + (k // span) % span
    rounds = 1 + (k // (span * span)) % 8
    return arnold_permutation(n_pix, a, b, rounds)


def keyed_generator(key: int, label: str) -> np.random.Generator:
    """Random generator for key-derived tables (permutations, substitution boxes)."""
    return np.random.default_rng(np.random.SeedSequence([GLOBAL_SEED, key_int(key, label)]))


def keyed_bit_permutation(key: int, q: int, label: str) -> np.ndarray:
    """Key-derived permutation of the ``q`` bit planes."""
    return keyed_generator(key, label).permutation(q).astype(np.int64)


def keyed_sboxes(key: int, n_pix: int, q: int, label: str) -> np.ndarray:
    """Independent key-derived substitution box at every position.

    Returns ``tables`` of shape ``(n_pix, 2**q)`` with ``tables[y]`` a
    permutation of the values, stored in the smallest unsigned integer type.
    """
    levels = 2 ** q
    dtype = np.uint8 if levels <= 256 else np.uint16
    base = np.broadcast_to(np.arange(levels, dtype=dtype), (n_pix, levels))
    return keyed_generator(key, label).permuted(base, axis=1)
