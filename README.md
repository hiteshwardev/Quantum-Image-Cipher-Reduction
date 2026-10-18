# A Structure Theorem for NEQR Quantum Image Ciphers: Reduction to Classical Permutation-Diffusion Ciphers, a Near-Optimal Chosen-Plaintext Attack and the Limits of Statistical Security Metrics

**Author:** Hitesh Kumar Singh · [ORCID: 0009-0008-1583-4848](https://orcid.org/0009-0008-1583-4848)
*Department of Physics, Kurukshetra University, Kurukshetra, Haryana, India*

---

## Abstract

Quantum image ciphers on NEQR are credited with security from their quantum encoding. This work proves that any unitary that maps every NEQR image to an NEQR image, with an invertible cipher image, acts on the pixel array as a position permutation followed by position-dependent value bijections. Such ciphers are classical. One chosen-plaintext attack recovers an equivalent key of every plaintext-independent member with 257 queries for 256 × 256 eight-bit images, against an information-theoretic lower bound of 213. On 120 images, a broken cipher is statistically equivalent to an attack-resistant one on entropy, correlation, key-sensitivity NPCR and UACI, and PSNR, and both pass the NIST SP 800-22 tests that were run.

## Core Objectives

- Characterize exactly which unitary operations can implement an image cipher on NEQR with a deterministic classical cipher image.
- Determine what this implies for security, and whether any security property can come from the quantum representation.
- Test whether the statistical metrics used in the literature can distinguish a completely broken cipher from an attack-resistant one.

## Scientific Significance

Reviews of quantum image encryption attribute security to superposition, entanglement and no-cloning, and the evidence offered is almost always a fixed set of statistics on cipher images. This project shows that for the studied class the representation adds nothing, so the security question reduces to classical cryptanalysis. It also shows that such statistics are necessary sanity checks but not evidence of security. The results do not claim that no quantum image cipher can be secure; they identify the hypotheses under which a security argument must be made.

## Methodological Approach

- **Structure theorem:** the cipher is identified with an automorphism of the Hamming graph of images, whose automorphism group is the wreath product S<sub>L</sub> ≀ S<sub>N</sub>.
- **Attack and bounds:** a generic equivalent-key recovery algorithm with constant images and base-L tag images, with matching information-theoretic lower bounds and specializations (XOR, bit level).
- **Three independent code paths** (classical pixel array, dense state-vector simulation, composed gate tables) used to cross-check results exactly, including dense non-basis-preserving unitaries.
- **Statistical battery:** entropy, adjacent-pixel correlation, NPCR/UACI, PSNR and NIST SP 800-22 tests on 120 images, with paired two one-sided tests (TOST) against margins fixed in advance.

## Contributions

1. A **structure theorem** showing that NEQR-preserving, invertible unitaries have the permutation-diffusion form, including dense unitaries that do not preserve the computational basis.
2. **Representation independence:** the quantum and classical ciphers compute the same function, so no security property can come from the representation.
3. A unified chosen-plaintext attack for arbitrary position-dependent substitutions with (L − 1) + ⌈log<sub>L</sub> N⌉ queries, within a factor 1.21 of a matching lower bound (257 vs 213), and an XOR specialization that meets its bound (3 queries). The constituent techniques are classical; the contribution is the unification and the lower bounds.
4. A proof that value-controlled position gates leave most images without a cipher image, and that independent bit-plane permutations cannot be realized on NEQR.
5. A tested demonstration that the metric battery does not separate a broken cipher from an attack-resistant one.

6. **Positioning relative to prior work.** Equivalent-key recovery, tag-image attacks on permutation-only ciphers and zero-image attacks on diffusion layers are established in classical image cryptanalysis (e.g. Li et al.; Zhang et al., "On the security of a class of diffusion mechanisms for image encryption"). The novelty claimed here is the structure theorem for NEQR-preserving unitaries, representation independence, and the matching query lower bounds. 

## Key Findings

- Pixel-array and state-vector computations agree on all 541,696 enumerated cases and on 120 full-size images (state dimension 2²⁴).
- The attack recovers the exact equivalent key of every plaintext-independent construction implemented (257 queries; 3 for XOR; 20 at bit level).
- Value-controlled position gates leave 75.0%, 75.0% and 93.8% of images without a cipher image at the three tested sizes.
- Broken and attack-resistant ciphers are equivalent on seven battery metrics; plaintext-sensitivity NPCR (0.0015% vs 99.6%) and the attack itself are the only measures that separate them.

## Scope and Limitations

The constructions follow published operation templates and are not exact reimplementations of any single scheme. The theorem covers NEQR (and the bit-plane representation) with classical readout; protocols that transmit quantum states are outside it. FRQI is not covered. The image ensemble is synthetic.

## Roles

Hitesh Kumar Singh is the sole author. Hitesh Kumar Singh conceived the study, developed the theory and proofs, wrote the software, performed the analysis and wrote the manuscript. This included designing the exhaustive verification and the statistical equivalence testing.

## Reproducibility and Availability

The repository contains the notebook, code and generated data with SHA-256 checksums that reproduce every result, table and figure. The code is released under the MIT licence.
