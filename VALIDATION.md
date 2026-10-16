# Validation

## Fresh manuscript route, v0.3.0, 2026-10-16

The new fixed-case builder, numerical postprocessor, common-mask calculation and renderer run from public source and scalar settings. Generated arrays and provenance remain local. The complete full-size 14-case workflow has not been executed. A tiny workflow tests interfaces and coverage, not the manuscript's numerical values.

Before the final reference optimization, the public suite passed 109 tests in 86.77 seconds. On the frozen revised builder,10 focused tests passed in 54.63 seconds. Independent tests confirmed bitwise equality of the reference bright blocks and readout at both one and three samples per physical pixel. The revised reference omits only the strip field and Gram products that the readout does not use. Source records distinguish the sampled D reference from the centered B target. The numerical analysis formulas are unchanged; its only final change removes obsolete coverage metadata.

The common-mask CLI passed 10 tests in 97.17 seconds and an independent three-test check. Its optical core matches the separately verified local 15D common-mask calculation. The full-size public wrapper itself has not been run. It requires a complete fresh bundle, checks source/input identities, and writes a separate local-gain diagnostic; it does not substitute that target into the finite-piston/noise arrays.

The exact 14-case tiny build, analysis, common-mask and rendering chain passed 11 renderer tests in 188.17 seconds. All 21 staged Python identities and all 31 output hashes were verified. The output contains ten PDF/PNG figures and eleven TeX fragments, with 21 coverage entries. Four B-transfer intervals are present, including 15D/128; removing one produces explicit partial coverage. Independent review checked that repaired panel and downstream values. All eleven fragments compiled offline and were visually inspected. The new rendering took 75.793 seconds and peaked at 225,746,944 bytes. These eight-pixel/eight-angle figures visibly identify their canary scale.

The independent saved-array regression reproduces504 finite-poke cases,24 minima and 6030 sensitivity points, with numerical differences at roundoff level. This tests the postprocessor against the private archive; it is not fresh optical propagation. Fresh postprocessing records its own Python, NumPy, SciPy and platform separately from the build runtime, along with input/source/output checksums. Cross-version bit identity is not promised.

The first full-size representative attempt exceeded its 600-second reference cap, leaving no completed case. Its failure record was preserved. After the bright-only reference repair, an execution-control error caused a separately attempted launch to be interrupted before any case completed; it supplies no numerical validation. The full readiness matrix was reassessed before a distinct bounded run. That run regenerated the full sampled reference in 651.231 seconds, with readout-weight SHA-256 `81fc8a8066a6af2613e8b05ca4657368bf9562ea95bb746f178fb7e6ca3f4972`, but reached its whole-run deadline during the original 15D/256 calculation. It exited with PARTIAL status after 1500.197 seconds, at a 1,566,773,248-byte peak working set. That attempt produced no completed 15D artifact or full-size analysis. A later recovery was interrupted at the user's request, also without a completed 15D artifact. Both interrupted records and the valid reference are preserved.

On resumption, the unchanged source-only representative profile completed using the verified fresh reference. Its new original 15D/256 calculation took 755.652 seconds; the resumed build took 755.779 seconds overall and peaked at 1,566,543,872 bytes. Source, scalar case, runtime and artifact identities were checked before reuse. The build used a 2400-second total allowance, a 1200-second 15D cap and a 2 GB observed-memory limit. These are Python-boundary/high-water checks rather than operating-system enforcement.

Fresh analysis completed in 1.079 seconds. A separately coded acceptance check then verified all source/input/output hashes, recomputed the reference weights, independently contracted the Grams and searched 4096 phases with bounded polishing. The weights were bit-identical to the historical reference; the reference bright block and original 15D Gram had zero maximum numerical difference from their archived counterparts. Independently computed gains, paired residual and selected transfer values also agreed exactly between fresh and archived arrays. Agreement with the analysis CSVs was within 2.23e-14 in the reported units. For target D, the coherent interval was [-1.246108185%, +2.852117044%] and the paired value was +0.148683375%. This representative profile does not generate the opaque B or orthogonal-command cases.

The lead ran the builder and analysis. The independent acceptance script was written by a separate read-only numerical auditor and executed by the lead after generation; its optical inputs came from the same production operators. This is code reproduction with an independently implemented numerical check, not an independent optical model or an external human replication. Private archived arrays were used only afterward as comparison oracles. The acceptance check passed in 1.581 seconds at a 289,460,224-byte peak working set. Raw numerical/provenance files remain local.

The earlier successful full 6D run below remains bound to its original source revision. Clean installation, native Linux/macOS full-size execution, full 14-case numerical agreement, device behavior and absolute-gain convergence remain unverified. The command and coverage map is in `reproduction-plan.md`.

## Documentation checkpoint v0.2.1, 2026-10-16

The regeneration module's opening docstring now names the matched command comparison in Supplement S8 instead of an obsolete table number. This is the only Python change from v0.2.0; executable statements and numerical operators are unchanged. The full run below used the exact v0.2.0 source identities, which remain preserved. Version v0.2.1 does not claim a second full optical run.

## Code checkpoint v0.2.0 — fresh code-only driver, 2026-10-16

The complete public driver ran from a source-only staging directory, with no archived input files, in **694.299 seconds** and peaked at **1,276,231,680 bytes** on Windows/Python 3.11.15. All four stages completed and all source/artifact checks passed. The generated 512-by-512 fixed readout was bit-identical to the archived reference; its SHA256 is `81fc8a8066a6af2613e8b05ca4657368bf9562ea95bb746f178fb7e6ca3f4972`. All twelve coherent-endpoint and paired-error values in the four rows of “Matched 6D/256 calibration-transfer comparison” matched exactly: maximum difference zero percentage points. The private archive was read only by a separate acceptance checker after generation. This result is environment-specific; byte-identical results are not promised across platforms or libraries.

The final public-code suite passed **97 tests** in 19.23 seconds: 91 numerical, command and entrypoint tests plus six resource-helper tests. The latter include native Windows peak-memory retention and simulated Linux/macOS API responses. The public driver reuses the production optical operators. A separate privately maintained optical implementation described in the paper also passed its matched 6D comparison, sharing the mathematical model and prescribed readout; it is not executed by this driver.

The driver now uses `runtime_resources.py` for the true historical process memory peak. Windows uses the native high-water mark; Linux/macOS branches use `getrusage` with platform-specific units. Native Windows allocation/release and simulated Linux/macOS responses were checked separately; simulated branches do not establish optical-workflow portability. All five entrypoint tests and six memory-helper tests passed together. A newly frozen source-only candidate completed its small canary in 0.372 seconds, peak 79,511,552 bytes. The earlier candidate below remains historical evidence.

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
