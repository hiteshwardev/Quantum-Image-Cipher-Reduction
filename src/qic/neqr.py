"""NEQR states, overlaps, readout and a full-size state-vector simulator.

The bit-plane representation (BRQI) is handled as an NEQR image with one-bit
pixels: :func:`to_bit_planes` places bit ``k`` of pixel ``y`` at position
``k N + y``.

Conventions
-----------
An image is a vector ``I`` of ``N`` pixels with values in ``{0, ..., L - 1}``,
``L = 2**q``. Two-dimensional images are flattened row by row. The NEQR state is

    |psi_I> = N**-0.5 * sum_y |I[y]> (x) |y>,

with a ``q``-qubit value register and an ``n``-qubit position register,
``N = 2**n``. Amplitudes are stored in an array ``A`` of shape ``(L, N)`` with
``A[v, y]`` the amplitude of ``|v>|y>``; flattening ``A`` row by row gives the
basis index ``v * N + y``.

Readout
-------
On the NEQR manifold every occupied basis state has probability exactly
``1 / N`` and every other basis state has probability zero. Readout therefore
declares ``|v>|y>`` occupied when its probability exceeds ``1 / (2 N)``. The
threshold sits midway between the two values the exact theory allows, so it is
insensitive to rounding error, which is many orders of magnitude smaller. A
state is accepted as a cipher image only if every position carries exactly one
occupied value.
"""
from __future__ import annotations

import numpy as np


def neqr_state(image: np.ndarray, q: int) -> np.ndarray:
    """Return the NEQR state of a flattened image as a dense vector."""
    img = np.asarray(image, dtype=np.int64).reshape(-1)
    n_pix = img.size
    levels = 2 ** q
    if img.min() < 0 or img.max() >= levels:
        raise ValueError("pixel value outside the q-bit range")
    amp = np.zeros((levels, n_pix), dtype=np.complex128)
    amp[img, np.arange(n_pix)] = 1.0 / np.sqrt(n_pix)
    return amp.reshape(-1)


def neqr_states(images: np.ndarray, q: int) -> np.ndarray:
    """NEQR states of a batch of flattened images, shape ``(B, L N)``."""
    imgs = np.asarray(images, dtype=np.int64)
    batch, n_pix = imgs.shape
    amp = np.zeros((batch, 2 ** q, n_pix), dtype=np.complex128)
    amp[np.arange(batch)[:, None], imgs, np.arange(n_pix)[None, :]] = 1.0 / np.sqrt(n_pix)
    return amp.reshape(batch, -1)


def all_images(n_pix: int, q: int) -> np.ndarray:
    """Every image with ``n_pix`` pixels of ``q`` bits, one per row (``L**N`` rows)."""
    levels = 2 ** q
    codes = np.arange(levels ** n_pix, dtype=np.int64)[:, None]
    return (codes // levels ** np.arange(n_pix, dtype=np.int64)[None, :]) % levels


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Number of positions at which two images differ."""
    return int(np.count_nonzero(np.asarray(a).reshape(-1) != np.asarray(b).reshape(-1)))


def neqr_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Inner product of two NEQR states, equal to 1 - d_H(a, b) / N."""
    n_pix = np.asarray(a).size
    return 1.0 - hamming(a, b) / n_pix


def readout_from_amplitudes(amp: np.ndarray) -> np.ndarray:
    """Read the cipher image from amplitudes of shape ``(..., L, N)``.

    Leading axes index a batch of states. Raises ``ValueError`` if some
    position carries zero or several occupied values, that is, if a state is
    not an NEQR image.
    """
    n_pix = amp.shape[-1]
    occupied = np.abs(amp) ** 2 > 0.5 / n_pix
    per_position = occupied.sum(axis=-2)
    if np.any(per_position != 1):
        bad = int(np.count_nonzero(per_position != 1))
        raise ValueError(f"not an NEQR image: {bad} positions without a unique value")
    return np.argmax(occupied, axis=-2).astype(np.int64)


def readout(state: np.ndarray, n_pix: int, q: int) -> np.ndarray:
    """Read the cipher image from a dense state vector."""
    return readout_from_amplitudes(np.asarray(state).reshape(2 ** q, n_pix))


def to_bit_planes(image: np.ndarray, q: int) -> np.ndarray:
    """Binary image of the bit-plane representation (BRQI).

    Bit ``k`` of pixel ``y`` occupies position ``k N + y``, so the result is
    an image with ``q N`` one-bit pixels.
    """
    img = np.asarray(image, dtype=np.int64).reshape(-1)
    return ((img[None, :] >> np.arange(q, dtype=np.int64)[:, None]) & 1).reshape(-1)


def from_bit_planes(bits: np.ndarray, q: int) -> np.ndarray:
    """Inverse of :func:`to_bit_planes`."""
    planes = np.asarray(bits, dtype=np.int64).reshape(q, -1)
    return np.sum(planes << np.arange(q, dtype=np.int64)[:, None], axis=0)


def neqr_complement_basis(n_pix: int, q: int) -> np.ndarray:
    """Orthonormal basis of the orthogonal complement of span{|psi_I>}.

    The span of all NEQR states has dimension ``N (L - 1) + 1``. Its complement
    consists of vectors ``sum_y c_y sum_v |v>|y>`` with ``sum_y c_y = 0`` and
    therefore has dimension ``N - 1``. The returned matrix has shape
    ``(L N, N - 1)`` and orthonormal columns.
    """
    levels = 2 ** q
    # Orthonormal coefficient vectors c orthogonal to (1, ..., 1).
    ones = np.ones((n_pix, 1)) / np.sqrt(n_pix)
    basis, _ = np.linalg.qr(np.hstack([ones, np.eye(n_pix)[:, : n_pix - 1]]))
    coeffs = basis[:, 1:n_pix]
    columns = np.zeros((levels * n_pix, n_pix - 1))
    for k in range(n_pix - 1):
        block = np.tile(coeffs[:, k], (levels, 1)) / np.sqrt(levels)
        columns[:, k] = block.reshape(-1)
    return columns


class StateVector:
    """Dense state of the value and position registers.

    Every method applies a unitary to the whole Hilbert space of dimension
    ``L * N``, zero amplitudes included, so the simulation does not assume the
    NEQR structure that it is used to test.
    """

    def __init__(self, amplitudes: np.ndarray):
        self.amp = np.asarray(amplitudes)
        if not np.iscomplexobj(self.amp):
            self.amp = self.amp.astype(np.complex128)
        self.levels, self.n_pix = self.amp.shape
        self.q = int(round(np.log2(self.levels)))

    @classmethod
    def from_image(cls, image: np.ndarray, q: int,
                   dtype: type = np.complex128) -> "StateVector":
        """NEQR state of an image.

        ``complex64`` halves the memory of full-size states. It is exact for
        the permutation gates used here, because they only move amplitudes,
        and ``N**-0.5`` is a power of two whenever ``N`` is a power of four.
        """
        img = np.asarray(image, dtype=np.int64).reshape(-1)
        amp = np.zeros((2 ** q, img.size), dtype=dtype)
        amp[img, np.arange(img.size)] = 1.0 / np.sqrt(img.size)
        return cls(amp)

    # Each permutation gate P is applied as a gather, (P a)[w] = a[P^-1(w)],
    # which touches every amplitude once and is much faster than a scatter.

    def permute_positions(self, pi: np.ndarray) -> None:
        """Apply |v>|y> -> |v>|pi(y)>."""
        pi = np.asarray(pi, dtype=np.int64)
        source = np.empty_like(pi)
        source[pi] = np.arange(pi.size)
        self.amp = np.take(self.amp, source, axis=1)

    def _gather_values(self, source: np.ndarray) -> None:
        """Set amplitude (w, y) to the old amplitude (source[w, y], y)."""
        self.amp = np.take_along_axis(self.amp, source, axis=0)

    def xor_values(self, stream: np.ndarray) -> None:
        """Apply |v>|y> -> |v XOR s(y)>|y>; the gate is its own inverse."""
        values = np.arange(self.levels, dtype=np.int64)[:, None]
        self._gather_values(values ^ np.asarray(stream, dtype=np.int64)[None, :])

    def add_values(self, stream: np.ndarray) -> None:
        """Apply |v>|y> -> |v + s(y) mod L>|y>."""
        values = np.arange(self.levels, dtype=np.int64)[:, None]
        self._gather_values((values - np.asarray(stream, dtype=np.int64)[None, :]) % self.levels)

    def substitute_values(self, tables: np.ndarray) -> None:
        """Apply |v>|y> -> |T_y(v)>|y> with ``tables[y, v] = T_y(v)``."""
        tables = np.asarray(tables, dtype=np.int64)
        inverse = np.argsort(tables, axis=1)
        self._gather_values(np.ascontiguousarray(inverse.T))

    def permute_value_qubits(self, bit_perm: np.ndarray) -> None:
        """Permute the value qubits: output bit k is input bit ``bit_perm[k]``.

        The value axis is split into one axis per qubit and the axes are
        transposed, which is how a SWAP network acts on a state vector.
        """
        q = self.q
        # Axis i of the reshaped tensor holds bit (q - 1 - i).
        tensor = self.amp.reshape((2,) * q + (self.n_pix,))
        order = [q - 1 - int(bit_perm[q - 1 - i]) for i in range(q)] + [q]
        self.amp = np.transpose(tensor, order).reshape(self.levels, self.n_pix).copy()

    def apply_phases(self, theta: np.ndarray) -> None:
        """Multiply every amplitude by exp(i theta[v, y])."""
        self.amp = self.amp * np.exp(1j * np.asarray(theta))

    def hadamard_value_qubit(self, bit: int) -> None:
        """Apply a Hadamard gate to value qubit ``bit``."""
        q = self.q
        tensor = self.amp.reshape((2,) * q + (self.n_pix,))
        axis = q - 1 - bit
        a0 = np.take(tensor, 0, axis=axis)
        a1 = np.take(tensor, 1, axis=axis)
        tensor = np.stack([(a0 + a1) / np.sqrt(2), (a0 - a1) / np.sqrt(2)], axis=axis)
        self.amp = tensor.reshape(self.levels, self.n_pix)

    def readout(self) -> np.ndarray:
        """Deterministic readout; raises ``ValueError`` off the NEQR manifold."""
        return readout_from_amplitudes(self.amp)

    def sample_image(self, generator: np.random.Generator) -> np.ndarray:
        """Draw one value at every position from its conditional distribution.

        This models a receiver who reads every position from repeated
        preparation and measurement of the state. When the value at a position
        is not deterministic, two such readouts disagree there.
        """
        prob = np.abs(self.amp) ** 2
        cond = prob / prob.sum(axis=0, keepdims=True)
        cdf = np.cumsum(cond, axis=0)
        u = generator.random(self.n_pix)
        return np.argmax(cdf > u[None, :], axis=0).astype(np.int64)
