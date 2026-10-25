"""Tests for the reimplemented published scheme. Run with ``pytest`` from the repository root."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qic import attack, images, neqr, normal_form, published  # noqa: E402
from qic.seeding import rng  # noqa: E402


def test_orbit_matches_field_at_small_step():
    key = published.Liu2024Key(h=1e-4)
    orbit = published.hyperchaotic_orbit(key, 2)
    s0 = np.array([key.x0, key.y0, key.z0, key.w0])
    derivative = (orbit[1] - orbit[0]) / key.h
    assert np.allclose(derivative, published._field(orbit[0], key), rtol=1e-3, atol=1e-3)
    assert not np.allclose(orbit[0], s0)


def test_transient_shifts_orbit():
    base = published.hyperchaotic_orbit(published.Liu2024Key(), 30)
    shifted = published.hyperchaotic_orbit(published.Liu2024Key(transient=10), 20)
    assert np.allclose(shifted, base[10:30])


def test_grey_operator_range_and_formula():
    orbit = published.hyperchaotic_orbit(published.Liu2024Key(), 1000)
    p = published.grey_operator(orbit)
    assert p.min() >= 0 and p.max() <= 255
    k2, k3 = orbit[5, 1], orbit[5, 2]
    assert p[5] == int(np.floor((k2 + 1) / (k2 + k3 + 2) * 1e14) % 256)


@pytest.mark.parametrize("combine", ["xor", "add"])
def test_state_path_and_normal_form(combine):
    cipher = published.liu2024(64, combine=combine)
    gen = rng(f"test/liu/{combine}")
    pi, phi = cipher.normal_form()
    for _ in range(5):
        img = gen.integers(0, 256, 64)
        enc = cipher.encrypt(img)
        state = cipher.encrypt_state(neqr.StateVector.from_image(img, 8))
        assert np.array_equal(state.readout(), enc)
        assert np.array_equal(normal_form.apply_normal_form(img, pi, phi), enc)
        assert np.array_equal(cipher.decrypt(enc), img)


@pytest.mark.parametrize("combine", ["xor", "add"])
def test_value_tables_match_pair_map(combine):
    cipher = published.liu2024(64, combine=combine)
    pi, tables = published.value_tables(cipher)
    pi_nf, phi = cipher.normal_form()
    assert np.array_equal(pi, pi_nf)
    source = np.empty_like(pi)
    source[pi] = np.arange(pi.size)
    assert np.array_equal(tables.astype(np.int64), np.asarray(phi, dtype=np.int64)[source])


@pytest.mark.parametrize("combine", ["xor", "add"])
def test_generic_attack_is_exact(combine):
    cipher = published.liu2024(1024, published.Liu2024Key(h=0.005, rounds=3), combine=combine)
    key = attack.generic_attack(cipher.oracle(), 1024, 8)
    assert key.queries == attack.generic_cost(1024, 8)
    assert published.key_is_exact(key, *published.value_tables(cipher))


def test_xor_fast_path_needs_xor_layer():
    xor = published.liu2024(1024, combine="xor")
    add = published.liu2024(1024, combine="add")
    assert published.key_is_exact(attack.xor_attack(xor.oracle(), 1024, 8), *published.value_tables(xor))
    assert not published.key_is_exact(attack.xor_attack(add.oracle(), 1024, 8), *published.value_tables(add))


def test_key_check_rejects_wrong_key():
    a = published.liu2024(256, published.Liu2024Key(x0=1.1))
    b = published.liu2024(256, published.Liu2024Key(x0=1.2))
    key_b = attack.generic_attack(b.oracle(), 256, 8)
    assert not published.key_is_exact(key_b, *published.value_tables(a))


def test_rejects_unsupported_arguments():
    with pytest.raises(ValueError):
        published.liu2024(64, combine="multiply")
    with pytest.raises(ValueError):
        published.liu2024(64, q=4)
    with pytest.raises(ValueError):
        published.liu2024(60)


def test_standard_images():
    imgs = images.standard_images()
    assert tuple(imgs) == images.STANDARD_IMAGES
    for img in imgs.values():
        assert img.shape == (512, 512) and img.dtype == np.int64
        assert 0 <= img.min() and img.max() <= 255
