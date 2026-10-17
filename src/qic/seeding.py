"""Reproducible random streams.

Every random quantity in the study derives from one global seed. A stream is
identified by a text tag; the tag is hashed with SHA-256 and the digest is mixed
with the global seed through ``numpy.random.SeedSequence``. Streams with
different tags are statistically independent, and the same tag always yields
the same stream on every platform. Python's built-in ``hash`` is avoided because
it is randomised per process.
"""
from __future__ import annotations

import hashlib

import numpy as np

GLOBAL_SEED = 20260712


def tag_to_int(tag: str) -> int:
    """Return a 64-bit integer derived from a text tag."""
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def rng(tag: str, seed: int = GLOBAL_SEED) -> np.random.Generator:
    """Return an independent generator for the stream named ``tag``."""
    return np.random.default_rng(np.random.SeedSequence([seed, tag_to_int(tag)]))
