# Release v0.1.0 validation

## Orthogonal-carrier addition, 2026-10-16

The existing 80 tests and six new independent command tests passed in the recorded Windows environment. The public-layout test run needs no numerical inputs. The adapter retains the production operators and changes only the strip carrier direction. A source-bound local runner reproduced all original 6D/128 Grams bit for bit, then validated the alternative at 128 and 256 angles. Every bright-control Gram block stayed bit-identical to its matched original. Timeout preservation, refusal to overwrite existing output and fresh recovery were exercised before the runs.

Independent contractions checked both B/D transfer intervals, ideal pairing, integrated strip power and raw zero-point offsets. They agree with the producer's summaries within 3e-12 percentage points. This is an independent analysis of the same arrays, not an independent optical simulator. A 128-to-256 angular check leaves an absolute-gain change of 1.822e-4; it does not establish absolute-gain convergence. No alternative-domain or hardware result is claimed.

The published analysis and plot entrypoints were executed from this repository against the private inputs. The scientific summaries and checks matched exactly; both CSV files and the rendered PNG were byte-identical to the reviewed results. Existing output directories are refused. No clean installation or additional operating system was tested.

New numerical arrays, configurations, launch manifests and source snapshots remain local. Only the adapter, independent tests, saved-array analysis and figure renderer are added to this public repository. The original release verification below remains historical.

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
