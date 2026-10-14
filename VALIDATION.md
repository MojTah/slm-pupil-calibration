# Release v0.1.0 validation

Checked on 2026-09-30 in the existing Windows/Python 3.11.15 scientific environment.

- 80 focused tests passed across held-pixel propagation, gain extrema, mean-response envelopes, phase pooling and finite calibration.
- All 185 local frozen input files matched `DATA_MANIFEST.json`; every member of the locally prepared data ZIP also matched after compression. Those numerical inputs and their metadata are not part of this public code release.
- The largest-domain opaque-target replay reproduced all 603 scientific CSV rows exactly.
- The finite-calibration replay reproduced all 1,512 scientific CSV rows exactly, including 567 cases outside the local reciprocal reporting cutoff and 945 cases reported only as local expansions.
- `plot_publication.py` passed its complex two-exposure identity, limiting-case and saved-domain-ratio checks. The opaque-target plot verified its source CSV hash before rendering.

The replay used existing saved optical arrays and a previously validated interpreter. It does not demonstrate a fresh dependency installation, another operating system, independent end-to-end propagation, laboratory performance or convergence outside the tested domains. Original producer metadata and source snapshots are retained in the separately maintained private input package for provenance.

## Additional diagnostic validation, 2026-10-14

Checked in the same Windows/Python 3.11.15 environment:

- The existing 80 tests passed again.
- `analyze_readout_sensitivity.py` ran beside copies of its public helper modules in an isolated staging directory. Only the seven required input runs, comprising seven manifests and seven arrays, were copied there. All input hashes remained unchanged.
- All ten readout/domain rows, the 201 piston/flux rows, signed D transfer endpoints, flux extrema, and the quadrant sensitivity exactly matched the previously checked calculation. Both CSV outputs were byte-identical.
- `check_source_hold.py` ran directly from the public code directory without saved inputs. A separate isolated replay exactly matched the checked mode counts, source-hold ratios, coefficient errors and unchanged detector integration factor.
- Eight negative path cases covered an existing output directory, a path outside `local-results`, a traversal outside it, and the `local-results` root itself for each entrypoint. Every case was rejected before numerical work or output creation.
- Neither diagnostic added numerical input arrays, manuscript files, or other project material to the public code tree. Outputs remain under ignored `local-results/` directories.

The entrypoints retain input/source identities, numerical assertions, a 60-second Python-boundary deadline, and a 2 GB observed peak-memory check. These checks are not OS-enforced resource limits. No new optical propagation, fresh dependency installation, or additional operating-system validation was performed.
