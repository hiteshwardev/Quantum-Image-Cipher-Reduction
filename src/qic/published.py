"""A published NEQR/GQIR scheme, reimplemented from its paper.

Liu et al. (2024)
    X.-D. Liu, Q.-H. Chen, R.-S. Zhao, G.-Z. Liu, S. Guan, L.-L. Wu and
    X.-K. Fan, "Quantum image encryption algorithm based on four-dimensional
    chaos", Front. Phys. 12, 1230294 (2024), doi:10.3389/fphy.2024.1230294.

What the paper specifies, and what this module follows:

* the four-dimensional system of its Eq. (1),
  ``x' = alpha (y - x) + w``, ``y' = gamma x - y - x z``, ``z' = x y - beta z``,
  ``w' = -y z + r w``, with ``alpha = 10``, ``beta = 8/3``, ``gamma = 28``,
  ``r = -1``, integrated by a Runge-Kutta method;
* the initial values ``x(t0) = 1.1``, ``y(t0) = 2.2``, ``z(t0) = 3.3``;
* one discrete time ``t_n`` per image point, with ``K1 = x(t_n)``,
  ``K2 = y(t_n)``, ``K3 = z(t_n)``;
* the grey-value operator of its Eq. (10),
  ``P_n = floor((K2 + 1) / (K2 + K3 + 2) * 1e14) mod 256``;
* a grey-value layer that combines ``P_n`` with the pixel value, followed by
  the Arnold map of its Eq. (8), ``A = [[1, 1], [1, 2]]``, applied a
  key-controlled number of times to the position register.

What the paper leaves open, and the choice made here (every choice is a
parameter of :func:`liu2024`):

* the initial value ``w(t0)`` (default 1.0), the step size ``h`` (default
  0.001), the number of discarded transient steps (default 0) and the number
  of Arnold iterations (default 10);
* the grey-value combination: the text of the paper says XOR, while its
  Eq. (10) adds ``P_n``. Both readings are provided (``"xor"`` and ``"add"``).

The equivalent-key attack of :mod:`qic.attack` does not use any of these
quantities, so the outcome of the attack is the same for every choice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .keystream import arnold_permutation, grid_side
from .schemes import Layer, LayeredCipher


@dataclass(frozen=True)
class Liu2024Key:
    """Key and open parameters of the scheme of Liu et al. (2024)."""

    x0: float = 1.1
    y0: float = 2.2
    z0: float = 3.3
    w0: float = 1.0
    h: float = 0.001
    transient: int = 0
    rounds: int = 10
    alpha: float = 10.0
    beta: float = 8.0 / 3.0
    gamma: float = 28.0
    r: float = -1.0


def _field(s: np.ndarray, k: Liu2024Key) -> np.ndarray:
    x, y, z, w = s
    return np.array([k.alpha * (y - x) + w,
                     k.gamma * x - y - x * z,
                     x * y - k.beta * z,
                     -y * z + k.r * w])


def hyperchaotic_orbit(key: Liu2024Key, length: int) -> np.ndarray:
    """``(length, 4)`` array of ``(x, y, z, w)`` at ``t_1, ..., t_length``.

    Classical fourth-order Runge-Kutta with step ``key.h``; ``key.transient``
    steps are discarded first. Row ``n`` belongs to image point ``n`` in
    row-major order.
    """
    s = np.array([key.x0, key.y0, key.z0, key.w0], dtype=np.float64)
    h = float(key.h)
    out = np.empty((int(length), 4), dtype=np.float64)
    for i in range(int(key.transient) + int(length)):
        k1 = _field(s, key)
        k2 = _field(s + 0.5 * h * k1, key)
        k3 = _field(s + 0.5 * h * k2, key)
        k4 = _field(s + h * k3, key)
        s = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(s)):
            raise FloatingPointError("the orbit left the range of floating-point numbers")
        j = i - int(key.transient)
        if j >= 0:
            out[j] = s
    return out


def grey_operator(orbit: np.ndarray) -> np.ndarray:
    """``P_n = floor((K2 + 1) / (K2 + K3 + 2) * 1e14) mod 256`` of Eq. (10)."""
    k2, k3 = orbit[:, 1], orbit[:, 2]
    value = np.floor((k2 + 1.0) / (k2 + k3 + 2.0) * 1e14)
    return np.mod(value, 256.0).astype(np.int64)


def liu2024(n_pix: int, key: Liu2024Key = Liu2024Key(), combine: str = "xor",
            q: int = 8) -> LayeredCipher:
    """The scheme of Liu et al. (2024) as a layered circuit on NEQR.

    ``combine`` selects the grey-value layer: ``"xor"`` (text of the paper) or
    ``"add"`` (Eq. (10) of the paper, read modulo ``2**q``). The position layer
    is the Arnold map ``(r, c) -> (r + c, r + 2 c)``, iterated ``key.rounds``
    times.
    """
    if combine not in ("xor", "add"):
        raise ValueError("combine must be 'xor' or 'add'")
    if q != 8:
        raise ValueError("the operator P_n of Eq. (10) is defined modulo 256, so q = 8")
    grid_side(n_pix)
    p = grey_operator(hyperchaotic_orbit(key, n_pix))
    pi = arnold_permutation(n_pix, 1, 1, int(key.rounds))
    layers = [Layer(combine, p), Layer("permute_positions", pi)]
    return LayeredCipher(f"Liu2024-{combine}", layers, n_pix, q)


def value_tables(cipher: LayeredCipher) -> tuple[np.ndarray, np.ndarray]:
    """Ground truth ``(pi, T)`` of a value layer followed by a position permutation.

    ``T[z, v]`` is the cipher value at output position ``z`` when the source
    pixel of ``z`` holds ``v``; tables are stored as ``uint8`` so that the check
    stays affordable at ``512 x 512`` pixels.
    """
    value, position = cipher.layers
    if position.kind != "permute_positions" or value.kind not in ("xor", "add"):
        raise ValueError("expected a value layer followed by a position permutation")
    pi = np.asarray(position.data, dtype=np.int64)
    source = np.empty_like(pi)
    source[pi] = np.arange(pi.size)
    stream = np.asarray(value.data, dtype=np.int64)[source].astype(np.uint8)[:, None]
    levels = np.arange(cipher.levels, dtype=np.uint8)[None, :]
    if value.kind == "xor":
        return pi, levels ^ stream
    # uint8 arithmetic wraps modulo 256, which is the addition of Eq. (10)
    return pi, levels + stream


def key_is_exact(recovered, pi: np.ndarray, tables: np.ndarray) -> bool:
    """True if a recovered equivalent key inverts ``pi`` and every table ``T_z``."""
    n_pix, levels = tables.shape
    if not np.array_equal(recovered.pi_inv[pi], np.arange(n_pix)):
        return False
    rows = np.arange(n_pix)[:, None]
    back = recovered.inv_tables[rows, tables.astype(np.int64)]
    return bool(np.all(back == np.arange(levels, dtype=back.dtype)[None, :]))
