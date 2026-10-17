"""Unit tests for the qic package. Run with ``pytest`` from the repository root."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qic import attack, images, keystream, metrics, neqr, normal_form, schemes  # noqa: E402
from qic.seeding import rng  # noqa: E402


@pytest.mark.parametrize("name", ["A", "B", "D", "E"])
@pytest.mark.parametrize("side,q", [(2, 2), (4, 3), (8, 4)])
def test_three_paths_agree(name, side, q):
    n_pix = side * side
    cipher = schemes.SCHEMES[name](17, n_pix, q)
    gen = rng(f"test/paths/{name}/{side}/{q}")
    for _ in range(5):
        img = gen.integers(0, 2 ** q, n_pix)
        enc = cipher.encrypt(img)
        state = cipher.encrypt_state(neqr.StateVector.from_image(img, q))
        assert np.array_equal(state.readout(), enc)
        pi, phi = cipher.normal_form()
        assert np.array_equal(normal_form.apply_normal_form(img, pi, phi), enc)
        assert np.array_equal(normal_form.invert_normal_form(enc, pi, phi), img)
        assert np.array_equal(cipher.decrypt(enc), img)


def test_lemma_overlap():
    gen = rng("test/lemma")
    for _ in range(20):
        a, b = gen.integers(0, 8, (2, 16))
        overlap = np.vdot(neqr.neqr_state(a, 3), neqr.neqr_state(b, 3)).real
        assert overlap == pytest.approx(neqr.neqr_overlap(a, b), abs=1e-15)


def test_excluded_gate_counts():
    gate = normal_form.value_controlled_position_flip(0, 0, 4, 2)
    assert normal_form.is_basis_permutation(gate)
    ok, info = normal_form.check_well_defined(gate)
    assert not ok and info["reason"] == "position depends on value"
    assert normal_form.count_undefined_images(gate) == (192, 256)


def test_dense_realisation_acts_like_permutation():
    cipher = schemes.scheme_e(3, 4, 2)
    pm = cipher.pair_map()
    u = normal_form.dense_realisation(pm, rng("test/dense"))
    assert np.allclose(u.conj().T @ u, np.eye(16), atol=1e-12)
    states = neqr.neqr_states(neqr.all_images(4, 2), 2)
    out = (u @ states.T).T.reshape(-1, 4, 4)
    assert np.array_equal(neqr.readout_from_amplitudes(out),
                          np.array([cipher.encrypt(i) for i in neqr.all_images(4, 2)]))


@pytest.mark.parametrize("name", ["A", "B", "D", "E"])
def test_generic_attack_recovers_equivalent_key(name):
    n_pix, q = 256, 8
    cipher = schemes.SCHEMES[name](99, n_pix, q)
    key = attack.generic_attack(cipher.oracle(), n_pix, q)
    assert key.queries == attack.generic_cost(n_pix, q) == 256
    pi, phi = cipher.normal_form()
    assert attack.key_matches(key, pi, phi)
    secret = rng(f"test/secret/{name}").integers(0, 2 ** q, n_pix)
    assert np.array_equal(key.decrypt(cipher.encrypt(secret)), secret)


def test_xor_and_bit_level_attacks():
    n_pix, q = 1024, 8
    a = schemes.scheme_a(5, n_pix, q)
    f = schemes.scheme_f(5, n_pix, q)
    secret = rng("test/secret/af").integers(0, 2 ** q, n_pix)
    key = attack.xor_attack(a.oracle(), n_pix, q)
    assert key.queries == attack.xor_cost(n_pix, q) == 3
    assert np.array_equal(key.decrypt(a.encrypt(secret)), secret)
    bits = attack.bit_level_attack(f.oracle(), n_pix, q)
    assert bits.queries == attack.bit_level_cost(n_pix, q) == 14
    assert np.array_equal(bits.decrypt(f.encrypt(secret)), secret)
    assert np.array_equal(f.encrypt_via_bits(secret), f.encrypt(secret))
    assert attack.key_matches(bits.key, *f.bit_level_cipher().normal_form())
    pi_bits, phi_bits = normal_form.bit_level_form(*a.normal_form(), q)
    assert attack.key_matches(attack.bit_level_attack(a.oracle(), n_pix, q).key, pi_bits, phi_bits)
    assert normal_form.bit_level_form(*schemes.scheme_d(5, n_pix, q).normal_form(), q) is None


def test_plaintext_dependent_scheme_resists():
    n_pix, q = 256, 8
    c = schemes.scheme_c(11, n_pix, q)
    key = attack.generic_attack(c.oracle(), n_pix, q)
    secret = rng("test/secret/c").integers(0, 2 ** q, n_pix)
    cipher, digest = c.encrypt_with_digest(secret)
    assert np.array_equal(c.decrypt(cipher, digest), secret)
    assert attack.pixel_accuracy(key.decrypt(cipher), secret) < 0.05


def test_counting_bounds():
    assert attack.xor_lower_bound(65536, 8) == attack.xor_cost(65536, 8) == 3
    assert attack.generic_lower_bound(65536, 8) == 213
    assert attack.bit_level_lower_bound(65536, 8) == 19


def test_arnold_rejects_non_square():
    with pytest.raises(ValueError):
        keystream.arnold_permutation(8, 1, 1, 1)
    pi = keystream.keyed_arnold(1, 64, "t")
    assert np.array_equal(np.sort(pi), np.arange(64))


def test_readout_rejects_non_neqr_state():
    state = neqr.StateVector.from_image(np.arange(4) % 4, 2)
    state.hadamard_value_qubit(1)
    with pytest.raises(ValueError):
        state.readout()


def test_nist_known_answers():
    records = metrics.nist_known_answers()
    assert len(records) == 10 and all(r["agrees"] for r in records)


def test_reference_values():
    assert metrics.expected_entropy_uniform(65536, 8) == pytest.approx(7.997193, abs=5e-7)
    assert metrics.npcr_ideal(8) == pytest.approx(99.609375)
    assert metrics.uaci_ideal(8) == pytest.approx(33.463542, abs=5e-7)


def test_tost_detects_equivalence_and_difference():
    gen = rng("test/tost")
    a, b = gen.normal(0, 1, 200), gen.normal(0, 1, 200)
    assert metrics.tost(a, b, margin=0.5)["equivalent"]
    assert not metrics.tost(a, b + 3, margin=0.5)["equivalent"]


def test_ensemble_is_deterministic():
    first = images.make_ensemble(side=32, n_fields=3)
    second = images.make_ensemble(side=32, n_fields=3)
    assert [n for n, _ in first] == [n for n, _ in second]
    assert all(np.array_equal(x, y) for (_, x), (_, y) in zip(first, second))
