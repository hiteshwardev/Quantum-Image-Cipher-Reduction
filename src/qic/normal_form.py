"""Permutation maps on (value, position) pairs and the cipher normal form.

A basis-preserving gate permutes the computational basis states ``|v>|y>``.
Such a gate is represented by two integer tables of shape ``(L, N)``,

    vout[v, y]   value component of the image of (v, y)
    yout[v, y]   position component of the image of (v, y)

and a circuit by the composition of its gates. Theorem 1 of the paper states
that a cipher which maps every NEQR image to an NEQR image and whose cipher
image determines the plaintext acts as

    C[pi(y)] = phi_y(I[y]),

with a position permutation ``pi`` and value bijections ``phi_y``. For a
basis-preserving circuit this holds exactly when ``yout`` does not depend on
``v``; ``check_well_defined`` tests the condition and returns a witness when it
fails.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .neqr import all_images, neqr_complement_basis

PairMap = tuple[np.ndarray, np.ndarray]


# --------------------------------------------------------------- gate tables

def identity(n_pix: int, q: int) -> PairMap:
    levels = 2 ** q
    vout = np.repeat(np.arange(levels)[:, None], n_pix, axis=1)
    yout = np.repeat(np.arange(n_pix)[None, :], levels, axis=0)
    return vout, yout


def compose(first: PairMap, second: PairMap) -> PairMap:
    """Return the map that applies ``first`` and then ``second``."""
    v1, y1 = first
    v2, y2 = second
    flat = v1 * v1.shape[1] + y1
    return np.take(v2, flat), np.take(y2, flat)


def position_permutation(pi: np.ndarray, q: int) -> PairMap:
    """|v>|y> -> |v>|pi(y)>."""
    pi = np.asarray(pi, dtype=np.int64)
    vout, _ = identity(pi.size, q)
    yout = np.repeat(pi[None, :], 2 ** q, axis=0)
    return vout, yout


def xor_layer(stream: np.ndarray, q: int) -> PairMap:
    """|v>|y> -> |v XOR s(y)>|y>."""
    s = np.asarray(stream, dtype=np.int64)
    _, yout = identity(s.size, q)
    return np.arange(2 ** q)[:, None] ^ s[None, :], yout


def addition_layer(stream: np.ndarray, q: int) -> PairMap:
    """|v>|y> -> |v + s(y) mod L>|y>."""
    s = np.asarray(stream, dtype=np.int64)
    _, yout = identity(s.size, q)
    return (np.arange(2 ** q)[:, None] + s[None, :]) % 2 ** q, yout


def bitplane_layer(bit_perm: np.ndarray, n_pix: int, q: int) -> PairMap:
    """Permute the bit planes of every pixel: output bit k is input bit bit_perm[k]."""
    values = np.arange(2 ** q)
    table = np.zeros_like(values)
    for new_bit, old_bit in enumerate(bit_perm):
        table |= ((values >> int(old_bit)) & 1) << new_bit
    _, yout = identity(n_pix, q)
    return np.repeat(table[:, None], n_pix, axis=1), yout


def substitution_layer(tables: np.ndarray, q: int) -> PairMap:
    """Apply an independent value bijection at each position, ``tables[y, v]``."""
    tables = np.asarray(tables, dtype=np.int64)
    _, yout = identity(tables.shape[0], q)
    return tables.T.copy(), yout


def value_controlled_position_flip(control_bit: int, target_bit: int,
                                   n_pix: int, q: int) -> PairMap:
    """Flip a position qubit when a value qubit is set.

    This multiply controlled NOT is unitary and basis preserving, but its
    position component depends on the value, which Theorem 1 excludes.
    """
    vout, yout = identity(n_pix, q)
    control = (np.arange(2 ** q) >> control_bit) & 1
    yout = np.where(control[:, None] == 1, yout ^ (1 << target_bit), yout)
    return vout, yout


# --------------------------------------------------------------- structure

def is_basis_permutation(pm: PairMap) -> bool:
    """True if the tables describe a bijection of the pair set."""
    vout, yout = pm
    n_pix = vout.shape[1]
    flat = vout.reshape(-1) * n_pix + yout.reshape(-1)
    return np.unique(flat).size == flat.size


def check_well_defined(pm: PairMap) -> tuple[bool, dict]:
    """Test whether the circuit maps every NEQR image to an NEQR image.

    Returns ``(True, info)`` when the position component is independent of the
    value and the induced maps are bijections. Otherwise returns ``(False,
    info)``. When the position depends on the value, ``info`` holds two images
    that differ in one pixel; their landing positions differ in exactly one
    element, so at least one of the two images has a collision.
    """
    vout, yout = pm
    levels, n_pix = vout.shape
    varies = np.flatnonzero(np.any(yout != yout[:1], axis=0))
    if varies.size:
        y = int(varies[0])
        v2 = int(np.flatnonzero(yout[:, y] != yout[0, y])[0])
        image_a = np.zeros(n_pix, dtype=np.int64)
        image_b = image_a.copy()
        image_b[y] = v2
        return False, {"reason": "position depends on value", "position": y,
                       "values": (0, v2), "image_a": image_a, "image_b": image_b}
    pi = yout[0]
    if np.unique(pi).size != n_pix:
        return False, {"reason": "position map is not a permutation"}
    if not np.array_equal(np.sort(vout, axis=0),
                          np.broadcast_to(np.arange(levels)[:, None], vout.shape)):
        return False, {"reason": "a value map is not a bijection"}
    return True, {"reason": "normal form"}


def factorize(pm: PairMap) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pi, phi)`` with ``phi[y, v] = phi_y(v)``."""
    ok, info = check_well_defined(pm)
    if not ok:
        raise ValueError(info["reason"])
    vout, yout = pm
    return yout[0].copy(), vout.T.copy()


def apply_normal_form(image: np.ndarray, pi: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Classical cipher C[pi(y)] = phi_y(I[y])."""
    img = np.asarray(image, dtype=np.int64).reshape(-1)
    out = np.empty_like(img)
    out[pi] = phi[np.arange(img.size), img]
    return out


def invert_normal_form(cipher: np.ndarray, pi: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Inverse of :func:`apply_normal_form`."""
    c = np.asarray(cipher, dtype=np.int64).reshape(-1)
    inverse = np.empty_like(phi)
    rows = np.arange(phi.shape[0])[:, None]
    inverse[rows, phi] = np.arange(phi.shape[1])[None, :]
    return inverse[np.arange(c.size), c[pi]]


def apply_pair_map(image: np.ndarray, pm: PairMap) -> np.ndarray:
    """Apply a basis-preserving circuit to an image, assuming it is well defined."""
    vout, yout = pm
    img = np.asarray(image, dtype=np.int64).reshape(-1)
    y = np.arange(img.size)
    out = np.empty_like(img)
    out[yout[img, y]] = vout[img, y]
    return out


def count_undefined_images(pm: PairMap) -> tuple[int, int]:
    """Exhaustively count images whose cipher image is undefined."""
    vout, yout = pm
    levels, n_pix = vout.shape
    digits = all_images(n_pix, int(round(np.log2(levels))))
    total = digits.shape[0]
    landing = yout[digits, np.arange(n_pix)[None, :]]
    landing.sort(axis=1)
    collide = np.any(landing[:, 1:] == landing[:, :-1], axis=1)
    return int(collide.sum()), int(total)


def bit_level_form(pi: np.ndarray, phi: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Normal form of the same cipher on the bit-plane representation, if it exists.

    The bit-level normal form exists when every ``phi_y`` permutes the bits of
    a value and then flips some of them. It is returned as ``(pi_bits,
    phi_bits)`` with ``L = 2`` and ``q N`` positions, bit ``k`` of pixel ``y``
    at position ``k N + y``, so ``phi_bits`` has shape ``(q N, 2)``. Otherwise
    the function returns ``None``.
    """
    pi = np.asarray(pi, dtype=np.int64)
    phi = np.asarray(phi, dtype=np.int64)
    n_pix, levels = phi.shape
    flips = phi[:, 0]
    targets = np.empty((q, n_pix), dtype=np.int64)
    rebuilt = np.repeat(flips[:, None], levels, axis=1)
    values = np.arange(levels)
    for k in range(q):
        moved = phi[:, 1 << k] ^ flips
        if np.any(moved == 0) or np.any(moved & (moved - 1)):
            return None
        targets[k] = np.log2(moved).round().astype(np.int64)
        rebuilt ^= ((values[None, :] >> k) & 1) * moved[:, None]
    if not np.array_equal(rebuilt, phi):
        return None
    pi_bits = (targets * n_pix + pi[None, :]).reshape(-1)
    flip_bits = ((flips[None, :] >> targets) & 1).reshape(-1)
    phi_bits = np.stack([flip_bits, 1 - flip_bits], axis=1)
    return pi_bits, phi_bits


# --------------------------------------------------------------- unitaries

def unitary_from_pair_map(pm: PairMap, phases: np.ndarray | None = None) -> np.ndarray:
    """Dense unitary of a basis-preserving circuit, with optional phases.

    ``phases[v, y]`` multiplies the image of ``|v>|y>``; the result is a
    monomial (generalised permutation) matrix.
    """
    vout, yout = pm
    levels, n_pix = vout.shape
    dim = levels * n_pix
    u = np.zeros((dim, dim), dtype=np.complex128)
    src = (np.arange(levels)[:, None] * n_pix + np.arange(n_pix)[None, :]).reshape(-1)
    dst = (vout * n_pix + yout).reshape(-1)
    weight = np.ones(dim) if phases is None else np.exp(1j * np.asarray(phases)).reshape(-1)
    u[dst, src] = weight
    return u


def dense_realisation(pm: PairMap, generator: np.random.Generator) -> np.ndarray:
    """A dense, non-monomial unitary that implements the same cipher as ``pm``.

    The unitary is ``P (Pi_S + Q V Q^dagger)``, where ``P`` is the permutation
    matrix of ``pm``, ``Pi_S`` projects onto the span of NEQR states, the
    columns of ``Q`` span its complement and ``V`` is a Haar-random unitary on
    that complement. It agrees with ``P`` on every NEQR state.
    """
    vout, _ = pm
    levels, n_pix = vout.shape
    p = unitary_from_pair_map(pm)
    q_basis = neqr_complement_basis(n_pix, int(round(np.log2(levels))))
    k = q_basis.shape[1]
    z = (generator.standard_normal((k, k)) + 1j * generator.standard_normal((k, k))) / np.sqrt(2)
    qmat, rmat = np.linalg.qr(z)
    v = qmat * (np.diag(rmat) / np.abs(np.diag(rmat)))
    projector = np.eye(levels * n_pix) - q_basis @ q_basis.T
    return p @ (projector + q_basis @ v @ q_basis.T)


# --------------------------------------------------------------- spread test

def single_pixel_spread(encrypt: Callable[[np.ndarray], np.ndarray], image: np.ndarray,
                        generator: np.random.Generator, trials: int, q: int) -> np.ndarray:
    """Hamming distances between the encryptions of single-pixel neighbours.

    By Lemma 1 the NEQR overlap of two images is ``1 - d_H / N``. A unitary
    preserves overlaps, so a cipher realised by a unitary on NEQR must map
    single-pixel neighbours to single-pixel neighbours. The function returns
    the output distances for ``trials`` random neighbours of ``image``.
    """
    img = np.asarray(image, dtype=np.int64).reshape(-1)
    base = np.asarray(encrypt(img), dtype=np.int64)
    out = np.empty(trials, dtype=np.int64)
    for t in range(trials):
        y = int(generator.integers(img.size))
        neighbour = img.copy()
        neighbour[y] = (img[y] + int(generator.integers(1, 2 ** q))) % 2 ** q
        out[t] = np.count_nonzero(np.asarray(encrypt(neighbour)) != base)
    return out
