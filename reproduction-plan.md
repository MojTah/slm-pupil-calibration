# Fresh manuscript reproduction route

This route generates new local numerical data from public source and stated scalar settings. It does not load the historical private archive. Separate commands build validated Grams, produce numerical CSVs, and render figures and numeric TeX tables. A completed tiny rendering check is not evidence that the full-size manuscript calculations have been reproduced.

The main calculation retains the published separate centered/translated floating-point iris masks. The common-mask local-gain diagnostic is a separate command; it does not replace the finite-piston or noise results. Fill-factor and device-leakage experiments remain outside this route.

## Small executable check

Run from the public code directory in an environment containing the pinned dependencies:

```powershell
python -m pytest -q -p no:cacheprovider test_reproduce_paper.py test_fresh_paper_analysis.py
python reproduce_paper.py build --profile all --canary --output local-results/paper-canary --wall-budget 60
python reproduce_paper.py analyze --input local-results/paper-canary --output local-results/paper-canary-analysis
python reproduce_common_mask.py --input local-results/paper-canary --output local-results/paper-canary-common --wall-budget 60
python render_paper.py --input local-results/paper-canary-analysis --build-input local-results/paper-canary --common-mask-input local-results/paper-canary-common --output local-results/paper-canary-render
python -m pytest -q -p no:cacheprovider test_render_paper.py
```

The canary uses eight pupil pixels and eight modulation angles. Its case names retain the nominal manuscript case labels for graph testing; manifests and CSVs explicitly mark canary status and actual angle count. Canary results are not paper numbers.

## Representative full-size 15D calculation

```powershell
python reproduce_paper.py build --profile representative-15d --output local-results/paper-15d --wall-budget 2400
python reproduce_paper.py analyze --input local-results/paper-15d --output local-results/paper-15d-analysis
```

This profile freshly constructs the sampled 6D reference (three samples per physical pixel,128 angles), its fixed readout, and the original-command 15D/256 Gram. The reference propagation omits the strip field: only its unchanged bright block is used to define the D readout. This prunes work without changing the sampled-reference convention. `transfer-summary.csv` contains the 15D coherent gain endpoints and D-target transfer/pairing values. No opaque B result is produced because the profile does not propagate B. Its execution still needs host-specific admission; source availability is not evidence that a full run passed.

Runtime varies between runs. An initial reference exceeded its 600-second cap. The revised bright-only reference completed in 651.231 seconds, but that attempt hit its 1500-second overall deadline during 15D propagation. After verified reference reuse, the unchanged original 15D/256 calculation completed in 755.652 seconds under its 1200-second stage cap; the resumed build, analysis and separate acceptance check all completed. See VALIDATION.md for the exact scope and failures. The command above allows 2400 seconds overall, with a 720-second reference cap and 1200-second 15D cap. These are not OS-enforced limits: a native operation can delay an interruption. Timings are measurements, not performance guarantees.

## Full fixed case graph

```powershell
python reproduce_paper.py build --profile all --output local-results/paper-all --wall-budget 10800
python reproduce_paper.py analyze --input local-results/paper-all --output local-results/paper-all-analysis
python reproduce_common_mask.py --input local-results/paper-all --output local-results/paper-all-common --wall-budget 900
python render_paper.py --input local-results/paper-all-analysis --build-input local-results/paper-all --common-mask-input local-results/paper-all-common --output local-results/paper-all-render
```

The profile has14 basis calls: one sampled 6D/s3/128 bright-only reference; original 6D128,6D256,12D256,15D128 and 15D256; opaque 6D256,12D256,15D128 and 15D256; orthogonal 6D128,6D256,12D256 and 15D256. D comes from the bright block of each SLM Gram. The added opaque 15D128 supplies the fourth B-transfer interval in Figure3(a); the added orthogonal 6D128 supports the separate angular check. Full-size runtime of the expanded graph is unknown. Individual modal caps remain300 seconds at 6D,660 at 12D and 1200 at 15D; the reference cap is 720 seconds. Their sum is10,200 seconds. The example allows10,800 seconds overall so that the total allowance does not undercut those individual limits. This is a limit, not a runtime estimate or completion guarantee.

Each stage checks available RAM against twice its estimate. The15D estimate is 1,962,934,272 bytes, requiring about 3.93 GB currently available RAM; observed peak above 2 GB fails the run. Previous15D peak was1.547 GB. Retain more free RAM for the operating system. The workflow uses one FFT/BLAS worker, publishes completed cases atomically, and releases large arrays before the next case. Memory observations are high-water checks, not OS allocation limits.

## Explicit verified reuse and interruption

Every output directory must be new. Failed/interrupted builds retain `result.json`, completed case checksums and any unfinished artifact. No run automatically resumes or overwrites earlier output. To copy validated completed cases into a new run:

```powershell
python reproduce_paper.py build --profile all --output local-results/paper-recovered --reuse local-results/paper-all --wall-budget 10800
```

Reuse checks the fixed case definitions, source hashes, runtime versions and each artifact checksum. Canary and full-size runs cannot be mixed. Only manifest-recorded completed cases are reused; a matching filename alone is insufficient. Recovery requires diagnosing the failure and rechecking resources before another full run. Postprocessing refuses incomplete bundles, verifies source and artifact identities, checks inputs again before completion, and records its own source identity and output checksums.

## Current numeric output coverage

| Output | Intended manuscript material |
|---|---|
| `transfer-summary.csv` | Main Tables 1–3, Supp Tables S7–S8; domain gains, command/readout intervals, strip power and raw offsets |
| `phase-curves.csv` | Main Fig. 4 phase curves and offsets |
| `pairing-summary.csv`, `transfer-curves.csv` | Main Figs. 3/5, Supp Fig. S3/Table S3; full-phase gain intervals and 201-point target transfer curves |
| `sensitivity-summary.csv`, `sensitivity-curves.csv` | Supp Tables S1/S2 and Figs. S1/S2; original-command angular/domain comparisons |
| `finite-calibration.csv` | Main Fig. 7 and Supp Table S4; original 15D six-poke local calibration diagnostic |
| `finite-equal-poke.csv`, `finite-minima.csv` | Supp Table S5 and both minimum tables; equal-poke grid extends to 0.3,0.4,0.5 rad |

Readout comparisons use the fixed sampled reference, native 15D D vector and global-quadrant vector normalized at 15D. All remain fixed across the compared commands/domains. Sensitivity uses the original convention `abs(V_lower/V_higher-1)` and phase contrasts relative to phase zero. Finite diagnostics retain the local reciprocal approximation and its CV reporting cutoff. Their selected phases are not a full-phase finite-noise optimization. `finite-calibration.csv` keeps the original six local-slope pokes; the equal-poke CSV has nine. Boundary minima are explicitly marked.

The original 6D standalone route and the historical saved-array tools remain separate interfaces. Fresh manifests identify new executions and do not fabricate old test records or archived checksums. Exact historical byte identity is neither required nor promised across libraries/platforms.

## Figure and TeX rendering

`render_paper.py` writes seven main figures and three supplementary figures as PDF and PNG, plus eleven numeric TeX tabular fragments when every required input is present. It uses the current three-panel domain figure, including the common-mask local-gain diagnostic. Figure names match the manuscript assets. TeX filenames describe their content; the manuscript determines final table numbering. The fragments use standard `tabular` and can be placed inside the manuscript's table environments; they do not replace captions or the manuscript itself.

| Rendered artifact | Required fresh input |
|---|---|
| `optical-schematic`, `pair-tolerances` | Public scalar settings and analytic coherence formula; no numerical bundle |
| `transfer-B`, `transfer` | CSVs; matching build arrays additionally supply target variance for the photon-crossover panel |
| `carrier-comparison`, `phase-pair`, `finite-calibration` | Verified analysis CSVs |
| `domain6-to15`, `domain12-to15` | Verified sensitivity CSVs |
| `domain-gains` | CSVs plus matching common-mask diagnostic for the revised overlay |
| Readout, command, centered-domain, angular, sensitivity, pairing, finite-poke, equal-poke and carrier TeX tables | Verified analysis CSVs |
| `supp-common-mask.tex` | CSVs plus matching common-mask diagnostic |

The two analytic figures can be generated alone:

```powershell
python render_paper.py --analytic-only --output local-results/paper-analytic
```

Rendering performs no optical propagation. The optional `--build-input` must match the original build-manifest hash recorded by the analysis. It supplies existing freshly generated target Grams and the fixed sampled readout solely for target-variance contractions. The optional `--common-mask-input` must be a completed `reproduce_common_mask.py` output bound to that same build. Its source, summary and array hashes are checked. At 6D and 12D, the renderer reuses centered-control values only after checking exact equality of the two masks;15D comes from the new common-mask calculation. Only zero-piston local gains use this separate diagnostic. Finite-piston and noise plots keep the centered B convention.

Every output directory must be new. The renderer records current Python/NumPy/Matplotlib/platform versions separately from analysis/common-mask runtimes, source hashes, input-manifest hashes, output checksums and per-artifact coverage. It checks inputs and sources again before completion. `COMPLETE_RENDER` means all requested rendering inputs were available; it does not certify full-size numerical agreement, historical typography, or publication readiness. Missing controls, crossover variance or the common-mask diagnostic produce explicit `PARTIAL_COVERAGE` records. The common-mask overlay is visibly labelled unavailable when absent. Tiny numerical figures carry a canary banner and the table fragments carry a canary comment. A render has a 180-second Python-boundary deadline; a native operation can delay interruption. Interrupted output is retained as a failed record, never silently reused.

## Current tiny-chain validation

On 2026-10-16, the native Windows source-only 14-case canary completed build, analysis, common-mask calculation and rendering through the exact CLIs. The focused renderer suite passed 11 tests in 188.17 seconds. It verified all 31 output hashes and 21 artifact coverage entries, including the fourth B-transfer interval at 15D/128 and explicit partial coverage when that case is removed. Rendering took 75.793 seconds and reached a 225,746,944-byte peak working set. These are eight-pixel/eight-angle interface and coverage checks; full-size numerical validation remains pending.

## Remaining work and evidence limits

- The renderer regenerates numerical content and layouts. It does not reproduce historical figure bytes or provide a complete typeset manuscript. TeX fragments still require integration with captions and journal styling.
- The renderer requires all four B-transfer interval cases, including 15D128. Missing intervals produce `PARTIAL_COVERAGE` with the exact missing case name; a three-interval panel cannot silently count as complete.
- The revised reference and representative original 15D/256 chain have completed, with separate numerical acceptance. The full-size 14-case route and full-size public common-mask wrapper remain unexecuted. Tiny checks establish interfaces and coverage for that complete route; they do not establish full-size agreement for every case or rendered artifact.
- The expanded graph contains orthogonal 6D128 and opaque 15D128 arrays, but not every related prose diagnostic is exported by the postprocessor.
- No common-mask replacement, device leakage/fill-factor sweep, continuum convergence, hardware validity or submission-readiness claim follows.
- No full optical calculation on another operating system or fresh dependency installation is established by these commands. An external user's full-size 15D output must have its actual source/runtime/artifact evidence recorded separately.

All newly generated arrays and provenance stay in the chosen local directory. Publishing this source does not release the author's private numerical archive.
