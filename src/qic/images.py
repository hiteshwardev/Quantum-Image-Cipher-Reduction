"""Synthetic image ensemble and its spectral characterisation.

The ensemble has two families of 8-bit square images.

Structured images (12): gradients, checkerboards, radial and sinusoidal
patterns, smoothed noise and a simple scene, all with the strong local
correlation that image ciphers are designed to remove.

Power-law fields: stationary Gaussian random fields whose power spectrum
falls as ``1/f**beta``, with ``beta`` spread evenly over an interval that
brackets the value ``beta = 2`` typical of natural scenes. The fields reproduce
the second-order statistics of natural images, on which every metric of the
battery depends, without licensed test images.

The ensemble is characterised, not assumed: :func:`spectral_slope` fits the
exponent of the radially averaged power spectrum of each image.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .metrics import adjacent_correlation, entropy
from .seeding import rng


def to_8bit(field: np.ndarray) -> np.ndarray:
    """Rescale a real array linearly onto the integers 0 to 255."""
    a = np.asarray(field, dtype=np.float64)
    lo, hi = a.min(), a.max()
    if hi - lo == 0:
        return np.zeros(a.shape, dtype=np.int64)
    return np.rint(255.0 * (a - lo) / (hi - lo)).astype(np.int64)


def structured_images(side: int) -> dict[str, np.ndarray]:
    y, x = np.mgrid[0:side, 0:side].astype(np.float64)
    r = np.hypot(x - side / 2, y - side / 2)
    noise = rng("ensemble/smooth-noise").standard_normal((side, side))
    smooth = noise
    for _ in range(3):
        smooth = ndimage.uniform_filter(smooth, size=9, mode="wrap")
    disk = (x - 0.35 * side) ** 2 + (y - 0.6 * side) ** 2 < (0.15 * side) ** 2
    chebyshev = np.maximum(np.abs(x - side / 2), np.abs(y - side / 2))
    return {
        "gradient-horizontal": to_8bit(x),
        "gradient-vertical": to_8bit(y),
        "gradient-diagonal": to_8bit(x + y),
        "radial": to_8bit(r),
        "checkerboard-16": to_8bit((x // 16 + y // 16) % 2),
        "checkerboard-8": to_8bit((x // 8 + y // 8) % 2),
        "sinusoid": to_8bit(np.sin(2 * np.pi * x / 32) + np.cos(2 * np.pi * y / 48)),
        "rings": to_8bit(np.sin(r / 3)),
        "smoothed-noise": to_8bit(smooth),
        "scene": to_8bit(0.5 * x / side + disk),
        "squares": to_8bit((chebyshev // 12) % 2),
        "modular-ramp": to_8bit((3 * x + 7 * y) % 97),
    }


def power_law_field(side: int, beta: float, generator: np.random.Generator) -> np.ndarray:
    """8-bit Gaussian random field with power spectrum proportional to ``1/f**beta``."""
    white = np.fft.fft2(generator.standard_normal((side, side)))
    f = np.hypot(*np.meshgrid(np.fft.fftfreq(side), np.fft.fftfreq(side), indexing="ij"))
    f[0, 0] = np.inf
    return to_8bit(np.real(np.fft.ifft2(white * f ** (-beta / 2.0))))


def make_ensemble(side: int = 256, n_fields: int = 108,
                  beta_range: tuple[float, float] = (1.2, 3.0)) -> list[tuple[str, np.ndarray]]:
    """The 12 structured images followed by ``n_fields`` power-law fields."""
    images = list(structured_images(side).items())
    for i, beta in enumerate(np.linspace(*beta_range, n_fields)):
        field = power_law_field(side, float(beta), rng(f"ensemble/field/{i}"))
        images.append((f"field-{i:03d}-beta-{beta:.2f}", field))
    return images


def nominal_beta(name: str) -> float:
    """Spectral exponent used to generate a power-law field, NaN for structured images."""
    return float(name.rsplit("-", 1)[1]) if name.startswith("field-") else float("nan")


def spectral_slope(image: np.ndarray, bins: int = 30) -> float:
    """Exponent ``beta`` of a power law fitted to the radially averaged spectrum."""
    a = np.asarray(image, dtype=np.float64)
    a = a - a.mean()
    power = (np.abs(np.fft.fft2(a)) ** 2).reshape(-1)
    f = np.hypot(*np.meshgrid(np.fft.fftfreq(a.shape[0]), np.fft.fftfreq(a.shape[1]),
                              indexing="ij")).reshape(-1)
    keep = (f > 0) & (power > 0)
    f, power = f[keep], power[keep]
    edges = np.logspace(np.log10(f.min()), np.log10(f.max()), bins + 1)
    centres, levels = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (f >= lo) & (f < hi)
        if np.count_nonzero(sel) > 4:
            centres.append(np.log10(np.sqrt(lo * hi)))
            levels.append(np.log10(power[sel].mean()))
    if len(centres) < 5:
        return float("nan")
    return float(-np.polyfit(centres, levels, 1)[0])


def characterise(image: np.ndarray, q: int = 8) -> dict[str, float]:
    """Spectral exponent, adjacent correlations and entropy of a plain image."""
    return {"spectral_slope": spectral_slope(image),
            "corr_h": adjacent_correlation(image, "h"),
            "corr_v": adjacent_correlation(image, "v"),
            "corr_d": adjacent_correlation(image, "d"),
            "entropy": entropy(image, q)}


def demo_image(side: int = 128) -> np.ndarray:
    """Test image with edges, a gradient, stripes and a disk, for the worked attack."""
    y, x = np.mgrid[0:side, 0:side].astype(np.float64) / side
    img = 0.35 * x + 0.15 * y
    img = np.where((x - 0.68) ** 2 + (y - 0.32) ** 2 < 0.04, 0.95, img)
    img = np.where((x > 0.12) & (x < 0.45) & (y > 0.55) & (y < 0.88), 0.08, img)
    stripes = (x > 0.55) & (x < 0.9) & (y > 0.62) & (y < 0.9)
    img = np.where(stripes, 0.5 + 0.4 * np.sin(2 * np.pi * 12 * x), img)
    return to_8bit(img)


# standard test images --------------------------------------------------------

#: Grey-level test images bundled with scikit-image whose documentation states
#: a CC0 licence or no copyright restriction; all are 512 x 512, 8-bit.
STANDARD_IMAGES = ("camera", "brick", "grass", "gravel")


def standard_images() -> dict[str, np.ndarray]:
    """The CC0 test images of :data:`STANDARD_IMAGES` as 8-bit integer arrays.

    scikit-image ships these files with the package, so no download is needed.
    The import is local so that the rest of the package does not depend on it.
    """
    from skimage import data

    out = {}
    for name in STANDARD_IMAGES:
        img = np.asarray(getattr(data, name)(), dtype=np.int64)
        if img.shape != (512, 512) or img.min() < 0 or img.max() > 255:
            raise ValueError(f"unexpected test image {name!r}: {img.shape}")
        out[name] = img
    return out
