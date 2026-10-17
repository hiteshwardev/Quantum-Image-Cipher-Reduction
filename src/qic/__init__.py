"""Reduction of basis-preserving quantum image ciphers to classical ciphers.

Modules
-------
seeding     reproducible random streams derived from one global seed
neqr        NEQR states, overlaps, readout and a full-size state-vector simulator
normal_form permutation maps on (value, position) pairs and the normal form
keystream   chaotic keystreams, keyed Arnold permutations, key derivation
schemes     the cipher constructions studied in the paper
attack      chosen-plaintext equivalent-key attacks and query-count bounds
metrics     statistical security metrics, NIST SP 800-22 tests, TOST
images      the synthetic image ensemble and its spectral characterisation
plotting    figure style, layout verification and export
"""

__version__ = "1.0.0"
