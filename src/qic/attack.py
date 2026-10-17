"""Chosen-plaintext equivalent-key attacks and query-count bounds.

A plaintext-independent cipher in normal form acts as
``C[pi(y)] = phi_y(I[y])``. Writing ``T_z = phi_{pi^{-1}(z)}`` for the value
bijection observed at output position ``z``, the pair ``(pi^{-1}, {T_z^{-1}})``
decrypts every cipher image produced under the key. It is an equivalent key.

Generic attack (any value bijections)
    Stage 1 queries the constant images ``I = v`` for ``v = 0, ..., L - 2``.
    A constant image hides the permutation, because every source pixel carries
    the same value, so the reply gives ``C_v[z] = T_z(v)`` at every output
    position. The column ``v = L - 1`` is the one value missing from each row
    and needs no query. Stage 2 queries ``ceil(log_L N)`` tag images whose
    pixel ``y`` holds digit ``r`` of ``y`` in base ``L``. Inverting ``T_z`` on
    the replies yields the digits of ``pi^{-1}(z)``. Total
    ``(L - 1) + ceil(log_L N)`` queries.

XOR fast path
    When every ``phi_y`` is an XOR mask, the all-zero image reveals the whole
    mask and the cost drops to ``1 + ceil(log_L N)``.

Bit-level attack
    On the bit-plane representation every bit is a one-bit pixel, and the
    generic attack with ``L = 2`` and ``q N`` positions costs
    ``1 + ceil(log_2(q N))`` queries. It breaks every cipher that acts on bits
    by a position permutation and bit flips, including per-plane permutations.

Lower bounds
    A query returns ``N q`` bits. An attack that recovers the equivalent key of
    every cipher in a class ``K`` must separate ``|K|`` functions, so it needs
    at least ``log2 |K| / (N q)`` queries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .neqr import from_bit_planes, to_bit_planes

Oracle = Callable[[np.ndarray], np.ndarray]


def _table_dtype(levels: int) -> type:
    return np.uint8 if levels <= 256 else np.uint16


def tag_queries(n_pix: int, levels: int) -> int:
    """Smallest ``m`` with ``levels**m >= n_pix``, the number of tag images."""
    m, reach = 0, 1
    while reach < n_pix:
        reach *= levels
        m += 1
    return m


def tag_image(n_pix: int, levels: int, r: int) -> np.ndarray:
    """Tag image ``r``: pixel ``y`` holds digit ``r`` of ``y`` in base ``levels``."""
    return (np.arange(n_pix, dtype=np.int64) // levels ** r) % levels


@dataclass
class EquivalentKey:
    """``pi_inv[z]`` is the source of output ``z``; ``inv_tables[z, c]`` inverts ``T_z``."""

    pi_inv: np.ndarray
    inv_tables: np.ndarray
    queries: int

    def decrypt(self, cipher: np.ndarray) -> np.ndarray:
        c = np.asarray(cipher, dtype=np.int64).reshape(-1)
        values = self.inv_tables[np.arange(c.size), c].astype(np.int64)
        # zeros rather than empty: a failed attack can return a pi_inv with
        # repeated entries, and unwritten pixels must stay deterministic
        out = np.zeros(c.size, dtype=np.int64)
        out[self.pi_inv] = values
        return out


def generic_attack(oracle: Oracle, n_pix: int, q: int) -> EquivalentKey:
    """Recover an equivalent key of any plaintext-independent normal-form cipher."""
    levels = 2 ** q
    dtype = _table_dtype(levels)
    tables = np.zeros((n_pix, levels), dtype=np.int64)
    for v in range(levels - 1):
        tables[:, v] = oracle(np.full(n_pix, v, dtype=np.int64))
    queries = levels - 1
    tables[:, levels - 1] = levels * (levels - 1) // 2 - tables[:, : levels - 1].sum(axis=1)
    tables %= levels
    inv_tables = np.zeros((n_pix, levels), dtype=dtype)
    inv_tables[np.arange(n_pix)[:, None], tables] = np.arange(levels, dtype=dtype)[None, :]
    pi_inv = np.zeros(n_pix, dtype=np.int64)
    rows = np.arange(n_pix)
    for r in range(tag_queries(n_pix, levels)):
        reply = np.asarray(oracle(tag_image(n_pix, levels, r)), dtype=np.int64)
        pi_inv += inv_tables[rows, reply].astype(np.int64) * levels ** r
        queries += 1
    return EquivalentKey(pi_inv % n_pix, inv_tables, queries)


def xor_attack(oracle: Oracle, n_pix: int, q: int, budget: int | None = None) -> EquivalentKey:
    """Fast path for XOR diffusion with at most ``budget`` queries.

    With a budget smaller than ``1 + ceil(log_L N)`` the missing digits of the
    source index default to zero, which gives the partial recovery reported
    in the recovery-versus-queries experiment.
    """
    levels = 2 ** q
    total = 1 + tag_queries(n_pix, levels)
    budget = total if budget is None else min(int(budget), total)
    mask = np.zeros(n_pix, dtype=np.int64)
    queries = 0
    if budget >= 1:
        mask = np.asarray(oracle(np.zeros(n_pix, dtype=np.int64)), dtype=np.int64)
        queries = 1
    pi_inv = np.zeros(n_pix, dtype=np.int64)
    for r in range(max(budget - 1, 0)):
        reply = np.asarray(oracle(tag_image(n_pix, levels, r)), dtype=np.int64)
        pi_inv += (reply ^ mask) * levels ** r
        queries += 1
    values = np.arange(levels, dtype=np.int64)
    inv_tables = (values[None, :] ^ mask[:, None]).astype(_table_dtype(levels))
    return EquivalentKey(pi_inv % n_pix, inv_tables, queries)


@dataclass
class BitLevelKey:
    """Equivalent key on the bit-plane representation."""

    key: EquivalentKey
    q: int

    @property
    def queries(self) -> int:
        return self.key.queries

    def decrypt(self, cipher: np.ndarray) -> np.ndarray:
        return from_bit_planes(self.key.decrypt(to_bit_planes(cipher, self.q)), self.q)


def bit_level_attack(oracle: Oracle, n_pix: int, q: int) -> BitLevelKey:
    """Generic attack with ``L = 2`` on the ``q N`` bit positions, through an image oracle."""
    def bit_oracle(bits: np.ndarray) -> np.ndarray:
        return to_bit_planes(oracle(from_bit_planes(bits, q)), q)
    return BitLevelKey(generic_attack(bit_oracle, q * n_pix, 1), q)


def key_matches(key: EquivalentKey, pi: np.ndarray, phi: np.ndarray) -> bool:
    """True if a recovered equivalent key equals the ground truth ``(pi, phi)``.

    The source map must invert ``pi`` and every inverse table must invert the
    value bijection ``T_z = phi_{pi^{-1}(z)}``. A key that passes decrypts every
    cipher image produced under the true key, so recovery is exact by
    construction rather than by sampling.
    """
    pi = np.asarray(pi, dtype=np.int64)
    n_pix, levels = phi.shape
    if not np.array_equal(key.pi_inv[pi], np.arange(n_pix)):
        return False
    source = np.empty(n_pix, dtype=np.int64)
    source[pi] = np.arange(n_pix)
    tables = np.asarray(phi, dtype=np.int64)[source]
    recovered = key.inv_tables[np.arange(n_pix)[:, None], tables].astype(np.int64)
    return bool(np.array_equal(recovered, np.broadcast_to(np.arange(levels), (n_pix, levels))))


def pixel_accuracy(recovered: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of pixels recovered exactly."""
    return float(np.mean(np.asarray(recovered).reshape(-1) == np.asarray(truth).reshape(-1)))


# query costs and counting bounds ------------------------------------------

def generic_cost(n_pix: int, q: int) -> int:
    return 2 ** q - 1 + tag_queries(n_pix, 2 ** q)


def xor_cost(n_pix: int, q: int) -> int:
    return 1 + tag_queries(n_pix, 2 ** q)


def bit_level_cost(n_pix: int, q: int) -> int:
    return 1 + tag_queries(q * n_pix, 2)


def log2_factorial(n: int) -> float:
    return math.lgamma(n + 1) / math.log(2)


def _bound(log2_classes: float, n_pix: int, q: int) -> int:
    return math.ceil(log2_classes / (n_pix * q) - 1e-12)


def generic_lower_bound(n_pix: int, q: int) -> int:
    """Queries needed for all normal-form ciphers, ``|K| = N! (L!)^N``."""
    return _bound(n_pix * log2_factorial(2 ** q) + log2_factorial(n_pix), n_pix, q)


def xor_lower_bound(n_pix: int, q: int) -> int:
    """Queries needed for permutation plus XOR ciphers, ``|K| = N! L^N``."""
    return _bound(n_pix * q + log2_factorial(n_pix), n_pix, q)


def bit_level_lower_bound(n_pix: int, q: int) -> int:
    """Queries needed for bit permutation plus bit flips, ``|K| = (qN)! 2^(qN)``."""
    return _bound(q * n_pix + log2_factorial(q * n_pix), n_pix, q)
