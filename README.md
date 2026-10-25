# NEQR-Preserving Quantum Image Ciphers Are Classical Permutation–Substitution Ciphers: A Structure Theorem and Its Consequences

**Author:** Hitesh Kumar Singh · [ORCID: 0009-0008-1583-4848](https://orcid.org/0009-0008-1583-4848)
*Department of Physics, Kurukshetra University, Kurukshetra, Haryana, India*

---

## Summary

Quantum image ciphers built on the novel enhanced quantum representation (NEQR) are usually credited with security from superposition, entanglement or the no-cloning theorem. This repository contains the proofs, code and data for the following results.

- **Structure theorem.** A unitary that maps every NEQR image to an NEQR image, with a cipher image that determines the plain image, is an automorphism of the Hamming graph of images. It therefore acts on the pixel array as a position permutation followed by a value bijection at each position. This holds for dense unitaries that do not preserve the computational basis.
- **Excluded constructions.** Value-controlled position gates leave the cipher image undefined for most images (75.0 %, 75.0 % and 93.8 % at the three register sizes enumerated), and independent permutations of the bit planes cannot be realised by any unitary on NEQR.
- **Representation independence.** For an adversary who submits classical images and receives classical cipher images, the quantum cipher and the classical cipher on the pixel array compute the same function, so every classical attack on permutation–substitution image ciphers applies unchanged. This does not cover protocols that transmit quantum states or expose a coherent quantum interface.
- **Case study.** The scheme of Liu et al. (*Front. Phys.* 12, 1230294, 2024) is reimplemented from its equations. On four CC0 test images of 512 × 512 pixels, an equivalent decryption key (not the chaotic initial values) is recovered exactly from 4 chosen images (XOR reading of the paper) or 258 (addition reading). The result is the same for all 72 tested settings of the quantities the paper does not state.

The chosen-plaintext techniques used here (constant images, base-L tag images, equivalent keys) and the observation that statistical metrics do not certify security are established in classical image cryptanalysis (see, for example, Hazzazi et al., *Array* 29, 100642, 2026). The automorphism group of the Hamming graph is classical (Brouwer, Cohen and Neumaier, 1989). The repository applies these results; it does not claim them.

## Repository layout

| Path | Content |
|---|---|
| `notebook.ipynb` | Runs the complete study top to bottom and writes every table, number and figure |
| `src/qic/` | Package: NEQR states and simulator (`neqr`), normal form (`normal_form`), key schedules (`keystream`), cipher constructions (`schemes`), the published scheme of Liu et al. (`published`), attacks and counting bounds (`attack`), statistical metrics and NIST SP 800-22 tests (`metrics`), images (`images`), figure style (`plotting`) |
| `tests/` | Unit tests (`pytest`) |
| `results/` | Generated data with SHA-256 checksums (`SHA256SUMS`); `run_info.json` records versions and run times |
| `figures/` | Generated figures (PNG and PDF) |

## Main numbers

| Quantity | Value |
|---|---|
| Pixel-array and state-vector paths, exhaustive small registers | 541,696 cases, 0 mismatches |
| 256 × 256 images, 8 bit (state dimension 2²⁴) | 120 images, 0 mismatches |
| Chosen images for an equivalent key, 256 × 256, any value bijections / XOR | 257 / 3 (counting bounds 213 / 3) |
| Liu et al. (2024), 512 × 512, XOR reading / addition reading | 4 / 258 chosen images, exact key on all four test images |
| Liu et al. (2024), settings of the unstated quantities | 72 of 72 with exact key |

## Reproducing the results

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate qic
pytest -q tests
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
cd results && sha256sum -c SHA256SUMS
```

A full run takes about ten minutes on one CPU core and needs about 3.5 GB of memory. `results/SHA256SUMS` lists the checksum of every data file, so a rerun in the pinned environment can be compared with the archived results using `sha256sum -c SHA256SUMS` in `results/`.

## Scope

The theorem covers ciphers whose output is a classical image read from an NEQR state, which is how the cited schemes report their cipher images, and the bit-plane representation. Protocols that transmit quantum states, and FRQI, whose grey values sit in amplitudes, are outside it. The six template constructions follow the operations of published schemes; the case study follows one published scheme equation by equation and states the choices made where the paper is silent.

## Licence

Code: MIT licence. The four test images (`camera`, `brick`, `grass`, `gravel`) are distributed with scikit-image under CC0 or without copyright restriction.
