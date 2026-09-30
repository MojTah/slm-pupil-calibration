# Release v0.1.0 validation

Checked on 2026-09-30 in the existing Windows/Python 3.11.15 scientific environment.

- 80 focused tests passed across held-pixel propagation, gain extrema, mean-response envelopes, phase pooling and finite calibration.
- All 185 frozen input files matched `DATA_MANIFEST.json`; every member of the release data ZIP also matched after compression.
- The largest-domain opaque-target replay reproduced all 603 scientific CSV rows exactly.
- The finite-calibration replay reproduced all 1,512 scientific CSV rows exactly, including 567 cases outside the local reciprocal reporting cutoff and 945 cases reported only as local expansions.
- `plot_publication.py` passed its complex two-exposure identity, limiting-case and saved-domain-ratio checks. The opaque-target plot verified its source CSV hash before rendering.

The replay used existing saved optical arrays and a previously validated interpreter. It does not demonstrate a fresh dependency installation, another operating system, independent end-to-end propagation, laboratory performance or convergence outside the tested domains. Original producer metadata and source snapshots are retained in the release data for provenance.
