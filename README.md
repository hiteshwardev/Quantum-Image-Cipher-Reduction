# Quantum image ciphers on NEQR reduce to classical permutation-diffusion ciphers

Code and data for the article *A structure theorem for NEQR quantum image ciphers: reduction to
classical permutation-diffusion ciphers, a near-optimal chosen-plaintext attack and the limits of
statistical security metrics* by Hitesh Kumar Singh.

The study proves that a unitary which maps every NEQR image to an NEQR image, with a cipher image
that determines the plain image, acts on the pixel array as a position permutation followed by
position-dependent value bijections. It then gives one chosen-plaintext attack that recovers an
equivalent key of every plaintext-independent cipher of this form, compares the query count with
information-theoretic lower bounds, and shows that the standard statistical battery (entropy,
correlation, NPCR, UACI, PSNR and NIST SP 800-22) cannot tell a broken cipher from a resistant one.

## Repository layout

```
notebook.ipynb     the complete study: theory, validation, experiments E1 to E4, figures, final results
src/qic/           reusable modules imported by the notebook
tests/test_qic.py  unit tests of the modules
figures/           the 16 figures of the article, PNG at 600 dpi and PDF
results/           result files written by the notebook, with SHA256SUMS
requirements.txt   pinned Python dependencies
environment.yml    conda environment built from requirements.txt
```

## Running the study

Python 3.13 and the packages in `requirements.txt` are required.

```
python -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook.ipynb
```

or, with conda, `conda env create -f environment.yml` followed by the same `jupyter` command. The
notebook can also be opened in Jupyter and run from top to bottom. A full run takes about nine
minutes on one CPU core and needs about 3.5 GB of memory, mostly for the full-size state vectors
of dimension 2^24 and the gate tables of the full-size ciphers.

The notebook writes every figure to `figures/` and every result file to `results/`:

| File | Content |
|---|---|
| `results.json` | every number reported in the article |
| `ensemble_statistics.csv` | spectral exponent, adjacent-pixel correlations and entropy of each plain image |
| `attack_coverage.csv` | the three attacks against the six constructions, five keys each |
| `query_scaling.csv` | measured and theoretical query counts and lower bounds for 16 to 1048576 pixels |
| `per_image_metrics.csv` | the metric battery for every image and both ciphers |
| `nist_pvalues.csv` | p-values of the six NIST SP 800-22 tests for every cipher image |
| `run_info.json` | software versions and wall-clock times (machine dependent) |
| `SHA256SUMS` | checksums of the deterministic result files |

## Reproducibility

Every random quantity derives from the global seed 20260712 through `qic.seeding.rng(tag)`, which
hashes a text tag with SHA-256 and combines it with the seed. Re-running the notebook in the same
environment reproduces every file listed in `results/SHA256SUMS` byte for byte; check with

```
cd results && sha256sum -c SHA256SUMS
```

`run_info.json` is excluded because it records timings.

## Modules

| Module | Purpose |
|---|---|
| `seeding` | reproducible random streams from the global seed |
| `neqr` | NEQR states, overlaps, readout, bit-plane conversion and a dense state-vector simulator |
| `normal_form` | gate tables on value-position pairs, factorisation into the normal form, dense realisations |
| `keystream` | logistic keystreams, keyed Arnold permutations and key derivation |
| `schemes` | the six cipher constructions, each runnable on pixel arrays and on state vectors |
| `attack` | the generic, XOR and bit-level equivalent-key attacks, query costs and lower bounds |
| `metrics` | the metric battery, the NIST SP 800-22 tests with their worked examples, TOST and bootstrap |
| `images` | the synthetic image ensemble and its spectral characterisation |
| `plotting` | figure style, layout checks and export |

Run the unit tests with `python -m pytest tests`.

## License

MIT, see `LICENSE`.
