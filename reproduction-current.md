# Reproducing the added readout and sensitivity analyses

Code version: **v0.4.0**, 2026-10-18. This is the command map for the added full-image state-built, weighted, slope-map, zero-offset, iris, geometry and leakage analyses. It supplements the earlier [reproduction-plan.md](reproduction-plan.md). The optical operators are unchanged from v0.3.0; the added analysis originated with the subsequent manuscript revisions and has been made portable here. Only source, tests and scalar settings are distributed. No historical numerical arrays or private configuration archive are included.

The commands below run from this checkout. Absolute script and input paths also work from an unrelated directory. Use the pinned dependencies in `requirements-reproduction.txt`. Every new output file or directory must be absent; existing outputs are refused. Keep generated files in `local-results/`. The scripts never download inputs. Phase extrema in the new full-image, weighted, slope, geometry and leakage contractions are **sampled extrema**, not certified continuous bounds. The iris fixed-readout contractions use the existing analytic gain-extremum routine and numerical polishing check.

## Inputs from source

The full-size commands can require minutes and substantial memory per propagation. These small analysis entrypoints do not enforce wall-time or memory limits: provision and supervise full-size runs explicitly. The older `reproduce_6d.py` retains its documented bounds. No new full-size propagation was run for this release.

```sh
python reproduce_6d.py --output local-results/current-reference --wall-budget 1200
python analysis/state-built-reconstructor/grams6d.py --output local-results/current-grams/grams6d.npz
python analysis/state-built-reconstructor/grams_dom.py 12 --output local-results/current-grams/grams12d.npz
python analysis/state-built-reconstructor/grams_dom.py 15 --output local-results/current-grams/grams15d.npz
```

`current-reference/readout.npz` is the fixed sampled D-built reference (three samples per physical pixel, 128 modulation angles); its original/orthogonal/opaque Grams use 256 angles. The separate native-Gram commands use 128 physical pupil pixels, an 8-pixel strip, carrier frequencies +/-0.125 cycles/pixel, iris radius 8 lambda/D, modulation radius 3 lambda/D, 256 angles, and a 512 by 512 detector. The 6D bundle includes both strip commands; 12D and 15D default to original plus opaque. `--orthogonal` adds the alternative at those domains. No archived input is required.

## Content-to-command map

| Current result family | Public entrypoints | Input and output |
|---|---|---|
| Full-image fixed/state-built readouts (main state-built table; Supplement readout diagnostic) | `recon.py`, `domains.py` | Regenerated native Grams; JSON values for every named readout/domain |
| Inverse-intensity weighted state-built readout | `weighted.py` | 6D native Grams; relative floor and phase count recorded in JSON |
| Native/quadrant zero offsets (supplement zero-offset table) | `offsets.py` | 6D native Grams; unit-D-gain offset decomposition |
| Sampled-reference zero offsets | `offsets_sref.py` | Fresh reference bundle; raw offsets, computed D gains and scaled offsets |
| Area-weighted/hard iris domains and radius sensitivity | `aa_iris.py`, `radius.py`, `statebuilt_aa.py` | Fresh sampled readout or source-built native readout; radius/domain/angle-labelled matrices and JSON summaries |
| Global/local slope readouts and zero offsets | `slopes.py`, `offsets_slopes.py` | Native Grams; fixed and state-built values, crop/mask settings and phase count |
| Geometry table | `sweep.py` | Source-only generation for five scalar geometries; native readout per geometry |
| Unswitched bright-shaped leakage example | `leak.py` | 6D native Grams; sampled phase ranges for four amplitudes |
| Target-only iris mismatch | `target_only.py` | Baseline target gain/pair residual plus shifted target gain; unchanged calibration gain and residual |

Names in this table identify content rather than unstable manuscript table numbers. New scripts produce numerical results; they do not automatically rebuild every new manuscript table or figure.

## Full-image readouts and offsets

```sh
python analysis/state-built-reconstructor/recon.py local-results/current-grams/grams6d.npz --output local-results/recon.json
python analysis/state-built-reconstructor/domains.py local-results/current-grams --domains 6 12 15 --output local-results/domains.json
python analysis/state-built-reconstructor/weighted.py local-results/current-grams/grams6d.npz --floor 1e-6 --output local-results/weighted.json
python analysis/state-built-reconstructor/offsets.py local-results/current-grams/grams6d.npz --output local-results/offsets.json
python analysis/iris-sampling/offsets_sref.py local-results/current-reference --output local-results/offsets-sampled.json
```

`recon.py` uses 720 coherent state-built phases and 1800 fixed-readout phases. `domains.py` uses 1800 for both and fails if **any requested domain is missing**. Weighted readouts use 720 phases; native/quadrant offsets use 1800; sampled-reference offsets use 7200. All counts can be set explicitly with `--phases` (`--fixed-phases` separately for `recon.py`). The weighted floor is relative to each image's maximum; repeat with another `--floor` and a new output to inspect floor sensitivity. A floor-insensitivity claim is not established by the release tests.

Expected full-size scalar checkpoints (reported manuscript calculations, **not regenerated in this release**): original coherent state-built 6D interval [-19.070514, -11.880390]%, paired +0.07334745%; original fixed native interval [-1.448933,+2.887815]%, paired +0.07331897%. Weighted intervals: original [-22.612,-10.660]%, orthogonal [-0.205,-0.160]%. Original native zero offset interval [-0.5539675,+0.6953130] nominal readout radians, pair +0.0722431. Sampled-reference offsets divided by computed D gain: original [-0.5546,+0.6960], pair +0.0723; orthogonal [+0.0623,+0.0677], pair +0.0650. Allow half a unit of the last quoted decimal for comparisons; floating-point/platform agreement must be checked rather than presumed.

## Iris quadrature, domains and radii

```sh
python analysis/iris-sampling/aa_iris.py local-results/current-reference/readout.npz --domains 6 8 10 12 15 --angles 128 --orthogonal --output local-results/iris-area
python analysis/iris-sampling/aa_iris.py local-results/current-reference/readout.npz --domains 6 8 10 12 15 --angles 128 --hard --output local-results/iris-hard
python analysis/iris-sampling/radius.py local-results/current-reference/readout.npz --domains 6 12 --angles 128 --radius 7.98 --output local-results/iris-r798
python analysis/iris-sampling/radius.py local-results/current-reference/readout.npz --domains 6 12 --angles 128 --radius 8.02 --output local-results/iris-r802
python analysis/iris-sampling/statebuilt_aa.py --output local-results/statebuilt-iris.json
```

For the hard-iris radius comparison, repeat the two radius commands with `--hard` and different output directories. The area iris uses 32 by 32 midpoint samples in boundary cells to approximate their area fractions; these fractions multiply field amplitudes. `statebuilt_aa.py` compares hard/area original-command native readouts at 6D, 256 angles and 720 phase samples. Keep its 256-angle values distinct from the 128-angle area/radius series. Radius is explicit, appears in each matrix filename and every summary row, and a new directory is required. Every domain, including 6D, is saved in JSON.

Expected checkpoints: area 6D/128 original B interval [-1.28977,+2.77660]%, pair +0.08066%, gB=0.985257; area 15D/128 original pair +0.08113995%, gB=0.9869524162. Original state-built area 6D/256 interval [-19.627172,-12.354791]%, pair +0.07131866%. These checks address local gain only; neither the area-weighted iris nor its apparent stability establishes convergence of the full finite-piston response.

## Slope maps, domain sensitivity and crop/mask variants

```sh
python analysis/slope-readouts/slopes.py local-results/current-grams/grams6d.npz --output local-results/slopes6.json
python analysis/slope-readouts/slopes.py local-results/current-grams/grams12d.npz --output local-results/slopes12.json
python analysis/slope-readouts/slopes.py local-results/current-grams/grams15d.npz --output local-results/slopes15.json
python analysis/slope-readouts/offsets_slopes.py local-results/current-grams/grams6d.npz --output local-results/slope-offsets.json
python analysis/slope-readouts/slopes.py local-results/current-grams/grams6d.npz --offset 95 --output local-results/slopes-offset95.json
python analysis/slope-readouts/slopes.py local-results/current-grams/grams6d.npz --offset 97 --output local-results/slopes-offset97.json
python analysis/slope-readouts/slopes.py local-results/current-grams/grams6d.npz --mask-radius 54 --output local-results/slopes-mask54.json
```

Default pupil crop radius is 64 pixels, centres are +/-96 pixels in each coordinate, circular mask radius 64, phase count 720 and detector size 512. The offset variants move the four crop centres radially by one pixel; they are not a common translation of all four images. Global and local normalizations are both produced. Offset calculations accept the same crop/mask options. Expected original 6D paired errors: global +0.1468%, local +0.9307%; original 15D +0.1464% and +0.9260%. Original 6D slope zero offsets: global interval [-0.5666,+0.7099], pair +0.0733; local [-0.4691,+0.5536], pair +0.0481. One-pixel crop variants give local paired values about 0.85–0.86%; reducing mask radius changes paired values by up to about 0.022 percentage points. These coarse values are diagnostics, not rigorous tolerances.

## Geometry and leakage

```sh
python analysis/geometry-sweep/sweep.py base 128 8 0.125 8 --output local-results/geometry-base.json
python analysis/geometry-sweep/sweep.py iris6 128 8 0.125 6 --output local-results/geometry-iris6.json
python analysis/geometry-sweep/sweep.py iris12 128 8 0.125 12 --output local-results/geometry-iris12.json
python analysis/geometry-sweep/sweep.py carrier4p 128 8 0.25 8 --output local-results/geometry-carrier4p.json
python analysis/geometry-sweep/sweep.py pupil256 256 16 0.125 8 --output local-results/geometry-pupil256.json
python analysis/leakage/leak.py local-results/current-grams/grams6d.npz --output local-results/leakage.json
```

Geometry tuples are pupil diameter pixels, strip width pixels, carrier cycles/pixel and iris radius lambda/D. Each uses 6D, 128 modulation angles, modulation radius 3 lambda/D, 360 phase samples and its own native D-built readout. Expected original paired errors by listed case: +0.07334%, +0.06325%, +0.01050%, -0.24836%, +0.01607% (5e-6 percentage-point rounding allowance). Different geometry/readout and angular conventions explain why these are not the primary fixed-reference values.

Leakage uses amplitudes 0, 0.003, 0.01 and 0.03; 72 leakage phases and 90 strip phases. The assumed unswitched field has the bright-field shape and does not respond to piston. Expected paired intervals: [0.0733,0.0733]%, [-0.2278,+0.3745]%, [-0.9305,+1.0773]%, [-2.9375,+3.0858]%. These sampled ranges apply to this one spatial pattern. They are not a general worst-case leakage bound.

## Target-only iris change

`radius.py` shifts both optical models. To hold the calibration fixed, use its shifted B gain with the baseline calibration gain, gCal = gB_baseline/(1+ePair_baseline/100):

```sh
python analysis/iris-sampling/target_only.py --baseline-target 0.985257 --baseline-pair-percent 0.08066 --shifted-target 0.991992 --output local-results/target-r798.json
python analysis/iris-sampling/target_only.py --baseline-target 0.985257 --baseline-pair-percent 0.08066 --shifted-target 0.978237 --output local-results/target-r802.json
```

These rounded 6D/128 values give +0.764789% and -0.632419% respectively (1e-6 percentage-point rounding allowance). For an exact freshly generated comparison, use full-precision `g_target` from the corresponding B summary rows and the baseline `paired_percent`, rather than these rounded examples. No extra optical propagation is needed for the mixing step.

## Verification levels

```sh
python -m pytest -q -p no:cacheprovider test_current_analysis.py
```

Eight focused tests pass on Windows/Python 3.11.15, NumPy 2.4.6 and SciPy 1.17.1. They check normalized-image and slope derivatives by finite differences, raw pooling, area weights/translation, missing-domain failure, exclusive outputs and target-only arithmetic. Every new entrypoint's help and every result family's real tiny workflow runs from a fresh directory outside the checkout. The optical tiny cases use 8 pupil pixels, a 2-pixel strip, 8 angles, radius 1 and modulation 0.73; tiny slope crops use radius 4 and offset 6. Those results are interface canaries, not paper values.

The scripts were ported without changing the production optical operators. Historical reported scalar checkpoints above came from the manuscript calculations (Linux/Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1 for the new native-Gram analyses). Source availability and tiny tests do not demonstrate a full-size rerun in the pinned Windows environment. No full-size new-family regeneration, clean dependency installation, every new table/figure rendering, or external hardware validation was performed for this release.
