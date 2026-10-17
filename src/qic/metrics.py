"""Statistical security metrics, NIST SP 800-22 tests and equivalence testing.

The metric battery is the one the quantum image encryption literature reports:
global Shannon entropy, adjacent-pixel correlation, NPCR and UACI, a histogram
chi-square test, MSE and PSNR against the plain image, and a subset of the
NIST SP 800-22 statistical tests applied to the cipher-image bit stream.

Reference values for an ideal cipher with ``q``-bit pixels and ``N`` pixels:

* entropy of ``N`` independent uniform pixels, plug-in estimator
  (Miller-Madow bias): ``q - (2**q - 1) / (2 N ln 2)``;
* NPCR between two independent uniform images: ``1 - 2**-q``;
* UACI between two independent uniform images: ``(2**q + 1) / (3 * 2**q)``.

The NIST tests follow SP 800-22 Rev. 1a and its reference implementation,
including the integer truncation in the summation limits of the cumulative
sums test. :func:`nist_known_answers` reproduces the worked examples of the
standard.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special, stats


# ------------------------------------------------------------ image metrics

def entropy(image: np.ndarray, q: int) -> float:
    """Plug-in Shannon entropy of the grey-level histogram, in bits."""
    counts = np.bincount(np.asarray(image, dtype=np.int64).reshape(-1), minlength=2 ** q)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def expected_entropy_uniform(n_pix: int, q: int) -> float:
    """Expected plug-in entropy of ``n_pix`` independent uniform ``q``-bit pixels."""
    return q - (2 ** q - 1) / (2 * n_pix * math.log(2))


def adjacent_correlation(image: np.ndarray, direction: str) -> float:
    """Pearson correlation over all adjacent pixel pairs of a 2-D image.

    ``direction`` is ``"h"`` (horizontal), ``"v"`` (vertical) or ``"d"``
    (diagonal). Returns NaN for a constant image, where the correlation is
    undefined.
    """
    a = np.asarray(image, dtype=np.float64)
    if direction == "h":
        x, y = a[:, :-1], a[:, 1:]
    elif direction == "v":
        x, y = a[:-1, :], a[1:, :]
    elif direction == "d":
        x, y = a[:-1, :-1], a[1:, 1:]
    else:
        raise ValueError("direction must be 'h', 'v' or 'd'")
    x = x.reshape(-1) - x.mean()
    y = y.reshape(-1) - y.mean()
    den = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    return float(np.dot(x, y) / den) if den > 0 else float("nan")


def npcr(c1: np.ndarray, c2: np.ndarray) -> float:
    """Number of pixels change rate, in percent."""
    return float(100.0 * np.mean(np.asarray(c1).reshape(-1) != np.asarray(c2).reshape(-1)))


def uaci(c1: np.ndarray, c2: np.ndarray, q: int) -> float:
    """Unified average changing intensity, in percent."""
    a = np.asarray(c1, dtype=np.float64).reshape(-1)
    b = np.asarray(c2, dtype=np.float64).reshape(-1)
    return float(100.0 * np.mean(np.abs(a - b)) / (2 ** q - 1))


def npcr_ideal(q: int) -> float:
    return 100.0 * (1.0 - 2.0 ** -q)


def uaci_ideal(q: int) -> float:
    return 100.0 * (2 ** q + 1) / (3 * 2 ** q)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.mean((x - y) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, q: int) -> float:
    """Peak signal-to-noise ratio in dB."""
    m = mse(a, b)
    return float("inf") if m == 0 else float(10.0 * np.log10((2 ** q - 1) ** 2 / m))


def histogram_chi_square(image: np.ndarray, q: int) -> tuple[float, float]:
    """Chi-square statistic and p-value for a uniform grey-level histogram."""
    counts = np.bincount(np.asarray(image, dtype=np.int64).reshape(-1), minlength=2 ** q)
    expected = counts.sum() / 2 ** q
    stat = float(np.sum((counts - expected) ** 2) / expected)
    return stat, float(stats.chi2.sf(stat, 2 ** q - 1))


# ------------------------------------------------------- NIST SP 800-22

def image_bits(image: np.ndarray, q: int) -> np.ndarray:
    """Bit stream of an image: pixels in row-major order, most significant bit first."""
    img = np.asarray(image, dtype=np.int64).reshape(-1)
    return ((img[:, None] >> np.arange(q - 1, -1, -1)[None, :]) & 1).reshape(-1).astype(np.int8)


def bits_from_string(text: str) -> np.ndarray:
    return np.array([int(c) for c in text if c in "01"], dtype=np.int8)


def nist_frequency(bits: np.ndarray) -> float:
    """Frequency (monobit) test."""
    n = bits.size
    s = int(np.sum(2 * bits.astype(np.int64) - 1))
    return float(special.erfc(abs(s) / math.sqrt(n) / math.sqrt(2.0)))


def nist_block_frequency(bits: np.ndarray, block: int) -> float:
    """Frequency test within blocks of ``block`` bits."""
    n_blocks = bits.size // block
    props = bits[: n_blocks * block].reshape(n_blocks, block).mean(axis=1)
    chi2 = 4.0 * block * float(np.sum((props - 0.5) ** 2))
    return float(special.gammaincc(n_blocks / 2.0, chi2 / 2.0))


def block_frequency_size(n: int) -> int:
    """Smallest power of two meeting SP 800-22: ``M >= 20``, ``M > 0.01 n``, fewer than 100 blocks."""
    block = 32
    while not (block > 0.01 * n and n // block < 100):
        block *= 2
    return block


def nist_runs(bits: np.ndarray) -> float:
    """Runs test; returns 0.0 when the frequency prerequisite fails, as SP 800-22 prescribes."""
    n = bits.size
    pi = float(np.mean(bits))
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return 0.0
    v_obs = 1 + int(np.count_nonzero(bits[1:] != bits[:-1]))
    num = abs(v_obs - 2.0 * n * pi * (1.0 - pi))
    den = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)
    return float(special.erfc(num / den))


def _longest_run_cdf(block: int, max_run: int) -> float:
    """Probability that ``block`` fair bits contain no run of ones longer than ``max_run``."""
    # ways[j] counts strings ending in exactly j ones
    ways = np.zeros(max_run + 1, dtype=object)
    ways[0] = 1
    for _ in range(block):
        new = np.zeros_like(ways)
        new[0] = sum(ways)
        new[1:] = ways[:-1]
        ways = new
    return sum(ways) / (1 << block)


def longest_run_parameters(n: int) -> tuple[int, list[int], np.ndarray]:
    """Block length, category boundaries and category probabilities of SP 800-22.

    The probabilities are computed exactly. For ``M = 8`` and ``M = 128`` they
    agree with the tables of the standard to the printed precision; the
    ``M = 10**4`` table of the standard deviates in the third decimal, a case
    that does not arise for the stream lengths of this study.
    """
    if n < 128:
        raise ValueError("the longest-run test needs at least 128 bits")
    if n < 6272:
        block, cats = 8, [1, 2, 3, 4]
    elif n < 750000:
        block, cats = 128, [4, 5, 6, 7, 8, 9]
    else:
        block, cats = 10000, [10, 11, 12, 13, 14, 15, 16]
    cdf = [_longest_run_cdf(block, c) for c in cats[:-1]]
    probs = np.diff([0.0] + cdf + [1.0])
    return block, cats, probs


def nist_longest_run(bits: np.ndarray) -> float:
    """Test for the longest run of ones in a block."""
    block, cats, probs = longest_run_parameters(bits.size)
    n_blocks = bits.size // block
    blocks = bits[: n_blocks * block].reshape(n_blocks, block).astype(np.int64)
    run = np.zeros(n_blocks, dtype=np.int64)
    best = np.zeros(n_blocks, dtype=np.int64)
    for j in range(block):
        run = (run + 1) * blocks[:, j]
        np.maximum(best, run, out=best)
    category = np.clip(best, cats[0], cats[-1]) - cats[0]
    counts = np.bincount(category, minlength=len(cats))
    expected = n_blocks * probs
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    return float(special.gammaincc((len(cats) - 1) / 2.0, chi2 / 2.0))


def _tdiv(a: int, b: int) -> int:
    """Integer division truncating toward zero, as in C."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def nist_cumulative_sums(bits: np.ndarray, reverse: bool = False) -> float:
    """Cumulative sums test, forward or reverse."""
    n = bits.size
    x = 2 * bits.astype(np.int64) - 1
    if reverse:
        x = x[::-1]
    z = int(np.max(np.abs(np.cumsum(x))))
    sqrt_n = math.sqrt(n)
    k1 = np.arange(_tdiv(-_tdiv(n, z) + 1, 4), _tdiv(_tdiv(n, z) - 1, 4) + 1)
    k2 = np.arange(_tdiv(-_tdiv(n, z) - 3, 4), _tdiv(_tdiv(n, z) - 1, 4) + 1)
    phi = stats.norm.cdf
    sum1 = np.sum(phi((4 * k1 + 1) * z / sqrt_n) - phi((4 * k1 - 1) * z / sqrt_n))
    sum2 = np.sum(phi((4 * k2 + 3) * z / sqrt_n) - phi((4 * k2 + 1) * z / sqrt_n))
    return float(1.0 - sum1 + sum2)


NIST_TESTS = ("frequency", "block_frequency", "runs", "longest_run",
              "cusum_forward", "cusum_reverse")


def nist_battery(image: np.ndarray, q: int) -> dict[str, float]:
    """P-values of the six tests on the bit stream of an image."""
    bits = image_bits(image, q)
    return {
        "frequency": nist_frequency(bits),
        "block_frequency": nist_block_frequency(bits, block_frequency_size(bits.size)),
        "runs": nist_runs(bits),
        "longest_run": nist_longest_run(bits),
        "cusum_forward": nist_cumulative_sums(bits),
        "cusum_reverse": nist_cumulative_sums(bits, reverse=True),
    }


def nist_second_level(pvalues: np.ndarray, alpha: float = 0.01) -> dict:
    """Second-level assessment of SP 800-22 Section 4.2 over many sequences.

    Proportion: the fraction of sequences with ``p >= alpha`` must lie in
    ``p_hat +/- 3 sqrt(p_hat (1 - p_hat) / m)``, ``p_hat = 1 - alpha``.
    Uniformity: the p-values are binned into ten intervals and a chi-square
    test with nine degrees of freedom gives ``P_T``; uniformity holds when
    ``P_T >= 1e-4``.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    passed = int(np.count_nonzero(p >= alpha))
    p_hat = 1.0 - alpha
    half = 3.0 * math.sqrt(p_hat * (1.0 - p_hat) / m)
    bins = np.minimum((p * 10).astype(np.int64), 9)
    counts = np.bincount(bins, minlength=10)
    chi2 = float(np.sum((counts - m / 10) ** 2 / (m / 10)))
    p_t = float(special.gammaincc(9 / 2.0, chi2 / 2.0))
    return {"sequences": m, "passed": passed, "proportion": passed / m,
            "interval_low": p_hat - half, "interval_high": min(p_hat + half, 1.0),
            "proportion_ok": bool(p_hat - half <= passed / m),
            "bin_counts": counts.tolist(), "chi2": chi2, "p_uniformity": p_t,
            "uniformity_ok": bool(p_t >= 1e-4)}


def pi_bits(n: int) -> np.ndarray:
    """First ``n`` bits of the binary expansion of pi (``11.0010...``), exact integer arithmetic."""
    guard = 64
    scale = 1 << (n + guard)

    def arctan_inv(x: int) -> int:
        total, term, k, sign = 0, scale // x, 1, 1
        while term:
            total += sign * (term // k)
            term //= x * x
            k += 2
            sign = -sign
        return total

    value = (16 * arctan_inv(5) - 4 * arctan_inv(239)) >> (guard + 2)
    return np.array([int(c) for c in format(value, "b")[:n]], dtype=np.int8)


# Worked examples of SP 800-22 Rev. 1a, Section 2: (test, input, expected p-value).
_NIST_EXAMPLE_128 = ("11001100000101010110110001001100111000000000001001001101010100010001"
                     "001111010110100000001101011111001100111001101101100010110010")
NIST_EXAMPLES = (
    ("frequency", "1011010101", 0.527089),
    ("frequency", "pi100", 0.109599),
    ("block_frequency", ("0110011010", 3), 0.801252),
    ("block_frequency", ("pi100", 10), 0.706438),
    ("runs", "1001101011", 0.147232),
    ("runs", "pi100", 0.500798),
    ("longest_run", _NIST_EXAMPLE_128, 0.180609),
    ("cusum_forward", "1011010111", 0.4116588),
    ("cusum_forward", "pi100", 0.219194),
    ("cusum_reverse", "pi100", 0.114866),
)


def nist_known_answers() -> list[dict]:
    """Recompute the worked examples of the standard, compared to six decimal places."""
    def bits_of(spec: str) -> np.ndarray:
        return pi_bits(100) if spec == "pi100" else bits_from_string(spec)

    records = []
    for test, spec, expected in NIST_EXAMPLES:
        if test == "frequency":
            got = nist_frequency(bits_of(spec))
        elif test == "block_frequency":
            got = nist_block_frequency(bits_of(spec[0]), spec[1])
        elif test == "runs":
            got = nist_runs(bits_of(spec))
        elif test == "longest_run":
            got = nist_longest_run(bits_of(spec))
        else:
            got = nist_cumulative_sums(bits_of(spec), reverse=test == "cusum_reverse")
        records.append({"test": test, "input": spec if isinstance(spec, str) else spec[0],
                        "expected": expected, "computed": got,
                        "agrees": bool(round(got, 6) == round(expected, 6))})
    return records


# ----------------------------------------------------------- statistics

def bootstrap_mean_ci(values: np.ndarray, generator: np.random.Generator,
                      n_boot: int = 10000, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval of the mean."""
    v = np.asarray(values, dtype=np.float64)
    means = v[generator.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    tail = 100 * (1 - level) / 2
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return float(v.mean()), float(lo), float(hi)


def tost(sample_a: np.ndarray, sample_b: np.ndarray, margin: float,
         alpha: float = 0.05, paired: bool = False) -> dict:
    """Two one-sided t tests for equivalence of two means within ``+/- margin``.

    Equivalence is declared when both one-sided nulls (difference at most
    ``-margin``, difference at least ``+margin``) are rejected at ``alpha``,
    which is the same as the ``1 - 2 alpha`` confidence interval of the
    difference lying inside the margin. With ``paired=True`` the samples are
    matched (the same images under two ciphers) and the test uses the
    differences; otherwise it is Welch's test. Hedges' g with the pooled
    standard deviation is reported as effect size.
    """
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    na, nb = a.size, b.size
    va, vb = a.var(ddof=1), b.var(ddof=1)
    diff = float(a.mean() - b.mean())
    if paired:
        if na != nb:
            raise ValueError("paired samples must have equal size")
        se = float(np.std(a - b, ddof=1)) / math.sqrt(na)
        df = float(na - 1)
    else:
        se = math.sqrt(va / na + vb / nb)
        df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    if se == 0:
        raise ValueError("zero standard error: the difference has no sampling variation")
    p_lower = float(stats.t.sf((diff + margin) / se, df))
    p_upper = float(stats.t.cdf((diff - margin) / se, df))
    t_crit = float(stats.t.ppf(1 - alpha, df))
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    correction = 1 - 3 / (4 * (na + nb) - 9)
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "difference": diff,
            "ci_low": diff - t_crit * se, "ci_high": diff + t_crit * se,
            "ci_level": 1 - 2 * alpha, "margin": float(margin), "df": float(df),
            "p_lower": p_lower, "p_upper": p_upper, "p_tost": max(p_lower, p_upper),
            "equivalent": bool(max(p_lower, p_upper) < alpha),
            "hedges_g": float(diff / pooled * correction) if pooled > 0 else float("nan"),
            "n_a": int(na), "n_b": int(nb)}
