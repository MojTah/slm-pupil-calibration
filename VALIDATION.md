# Validation

## Code checkpoint v0.2.0 — fresh code-only driver, 2026-10-16

The complete public driver ran from a source-only staging directory, with no archived input files, in **694.299 seconds** and peaked at **1,276,231,680 bytes** on Windows/Python 3.11.15. All four stages completed and all source/artifact checks passed. The generated 512-by-512 fixed readout was bit-identical to the archived reference; its SHA256 is `81fc8a8066a6af2613e8b05ca4657368bf9562ea95bb746f178fb7e6ca3f4972`. All twelve coherent-endpoint and paired-error values in the four rows of “Matched 6D/256 calibration-transfer comparison” matched exactly: maximum difference zero percentage points. The private archive was read only by a separate acceptance checker after generation. This result is environment-specific; byte-identical results are not promised across platforms or libraries.

The final public-code suite passed **97 tests** in 19.23 seconds: 91 numerical, command and entrypoint tests plus six resource-helper tests. The latter include native Windows peak-memory retention and simulated Linux/macOS API responses. The public driver reuses the production optical operators. A separate privately maintained optical implementation described in the paper also passed its matched 6D comparison, sharing the mathematical model and prescribed readout; it is not executed by this driver.

The driver now uses `runtime_resources.py` for the true historical process memory peak. Windows uses the native high-water mark; Linux/macOS branches use `getrusage` with platform-specific units. Native Windows allocation/release and simulated Linux/macOS responses were checked separately; simulated branches do not establish optical-workflow portability. All five entrypoint tests and six memory-helper tests passed together. A newly frozen source-only candidate completed its small canary in0.372 seconds, peak79,511,552 bytes. The earlier candidate below remains historical evidence.

Five focused tests passed for `reproduce_6d.py`: a real small propagation through the actual CLI, four transfer rows checked against direct raw-exposure pooling, unchanged original/orthogonal bright blocks, refusal to overwrite, sampled-reference strip exclusion, pre-stage and in-execution deadlines, durable partial records and recovery in a fresh directory. The existing 86 numerical/command tests also passed.

A source-only staging directory containing the driver and its public Python dependencies completed the same small canary without any archived inputs. It took 0.487 seconds inside the driver and peaked at 78,888,960 bytes on the existing Windows/Python environment. This eight-pixel/eight-angle canary verifies the entrypoint and lifecycle; it does not reproduce the paper's values.

No fresh dependency installation or optical workflow on another operating system has been tested. The fresh public command does not regenerate every historical domain, figure or finite-noise table. Existing saved-array validation below remains separate evidence. Historical replay commands check source identities and require matching-version test provenance; old archives do not automatically validate scripts changed in this checkpoint. Use the corresponding recorded code revision for a historical replay.

## Historical orthogonal-carrier addition, 2026-10-16

The existing 80 tests and six new independent command tests passed in the recorded Windows environment. The public-layout test run needs no numerical inputs. The adapter retains the production operators and changes only the strip carrier direction. A source-bound local runner reproduced all original 6D/128 Grams bit for bit, then validated the alternative at 128 and 256 angles. Every bright-control Gram block stayed bit-identical to its matched original. Timeout preservation, refusal to overwrite existing output and fresh recovery were exercised before the runs.

Independent contractions checked both B/D transfer intervals, ideal pairing, integrated strip power and raw zero-point offsets. They agree with the producer's summaries within 3e-12 percentage points. This is an independent analysis of the same arrays, not an independent optical simulator. A 128-to-256 angular check leaves an absolute-gain change of 1.822e-4; it does not establish absolute-gain convergence. No alternative-domain or hardware result is claimed.

The published analysis and plot entrypoints were executed from this repository against the private inputs. The scientific summaries and checks matched exactly; both CSV files and the rendered PNG were byte-identical to the reviewed results. The saved-array analyses refuse existing output directories; plot renderers can replace their chosen output files. No clean installation or additional operating system was tested.

New numerical arrays, configurations, launch manifests and source snapshots remain local. Only the adapter, independent tests, saved-array analysis and figure renderer are added to this public repository. The original release verification below remains historical.

## Historical release v0.1.0, 2026-09-30

Checked in the existing Windows/Python 3.11.15 scientific environment.

- 80 focused tests passed across held-pixel propagation, gain extrema, mean-response envelopes, phase pooling and finite calibration.
- All 185 local frozen input files matched `DATA_MANIFEST.json`; every member of the locally prepared data ZIP also matched after compression. Those numerical inputs and their metadata are not part of this public code release.
- The largest-domain opaque-target replay reproduced all 603 scientific CSV rows exactly.
- The finite-calibration replay reproduced all 1,512 scientific CSV rows exactly, including 567 cases outside the local reciprocal reporting cutoff and 945 cases reported only as local expansions.
- `plot_publication.py` passed its complex two-exposure identity, limiting-case and saved-domain-ratio checks. The opaque-target plot verified its source CSV hash before rendering.

The replay used existing saved optical arrays and a previously validated interpreter. It does not demonstrate a fresh dependency installation, another operating system, independent end-to-end propagation, laboratory performance or convergence outside the tested domains. Original producer metadata and source snapshots are retained in the separately maintained private input package for provenance.

## Historical additional diagnostic validation, 2026-10-14

Checked in the same Windows/Python 3.11.15 environment:

- The existing 80 tests passed again.
- `analyze_readout_sensitivity.py` ran beside copies of its public helper modules in an isolated staging directory. Only the seven required input runs, comprising seven manifests and seven arrays, were copied there. All input hashes remained unchanged.
- All ten readout/domain rows, the 201 piston/flux rows, signed D transfer endpoints, flux extrema, and the quadrant sensitivity exactly matched the previously checked calculation. Both CSV outputs were byte-identical.
- `check_source_hold.py` ran directly from the public code directory without saved inputs. A separate isolated replay exactly matched the checked mode counts, source-hold ratios, coefficient errors and unchanged detector integration factor.
- Eight negative path cases covered an existing output directory, a path outside `local-results`, a traversal outside it, and the `local-results` root itself for each entrypoint. Every case was rejected before numerical work or output creation.
- Neither diagnostic added numerical input arrays, manuscript files, or other project material to the public code tree. Outputs remain under ignored `local-results/` directories.

The entrypoints retain input/source identities, numerical assertions, a 60-second Python-boundary deadline, and a 2 GB observed peak-memory check. These checks are not OS-enforced resource limits. No new optical propagation, fresh dependency installation, or additional operating-system validation was performed.
