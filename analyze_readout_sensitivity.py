"""Readout and domain sensitivity from frozen detector Grams; no propagation."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parent
import numpy as np
import psutil
from analyze_transfer import load_gram, B_TARGET_RUNS
from gain_phase import gain_coefficients, gain_phase_range, gain_value
from phase_pair import paired_matrices
from piston_basis import evaluate, readout_matrices, readout_moments
from run_pilot import WallDeadline
from simulation import normalize_response


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dark(matrices):
    result = matrices.copy()
    result[..., 2, :] = 0
    result[..., :, 2] = 0
    return result


def native(gram):
    intensity = gram[..., 0, 0].real
    tangent = 2 * gram[..., 0, 1].real
    y, j = normalize_response(intensity, tangent)
    weights = j / np.sum(j * j)
    assert abs(np.sum(weights * j) - 1) < 2e-14
    return weights, j


def selfcheck():
    # Independent pixel-level quotient and finite differences test contractions.
    rng = np.random.default_rng(61215)
    z = rng.normal(size=(4, 4, 3, 3)) + 1j * rng.normal(size=(4, 4, 3, 3))
    gram = np.einsum("...ki,...kj->...ij", z.conj(), z)
    w = rng.normal(size=(4, 4))
    matrices = readout_matrices(gram, w)
    a, phi, h = .31, .72, 1e-5
    y, j = normalize_response(*evaluate(gram, a, phi))
    result = readout_moments(matrices, a, phi)
    assert abs(np.sum(w * y) - result["weighted_mean"]) < 1e-13
    assert abs(np.sum(w * j) - result["local_gain"]) < 1e-13
    slope = (readout_moments(matrices, a+h, phi)["weighted_mean"] -
             readout_moments(matrices, a-h, phi)["weighted_mean"]) / (2*h)
    assert abs(slope - result["local_gain"]) < 1e-9
    raw_pair = (evaluate(gram, a, phi)[0] + evaluate(gram, a, phi+np.pi)[0])/2
    pair_mean = np.sum(w * raw_pair) / raw_pair.sum()
    assert abs(pair_mean-readout_moments(paired_matrices(matrices, 0), a)["weighted_mean"]) < 1e-13
    assert abs(gain_value(gain_coefficients(matrices, a), phi)-result["local_gain"]) < 1e-13


def row(domain, label, gram, opaque, weights):
    matrices = readout_matrices(gram, weights)
    d = dark(matrices)
    b = readout_matrices(opaque, weights)
    g_d = readout_moments(d, 0)["local_gain"]
    g_b = readout_moments(b, 0)["local_gain"]
    ext = gain_phase_range(gain_coefficients(matrices))
    assert ext["inversion_status"] == "WELL_CONDITIONED"
    pair_gain = readout_moments(paired_matrices(matrices, 0), 0)["local_gain"]
    e_b = sorted(g_b/ext[k]-1 for k in ("minimum", "maximum"))
    e_d = sorted(g_d/ext[k]-1 for k in ("minimum", "maximum"))
    assert abs((1+(g_d/pair_gain-1))*(g_b/g_d)-1-(g_b/pair_gain-1)) < 2e-14
    return {
        "domain_D": domain, "readout": label, "g_D": g_d, "g_B": g_b,
        "g_C_phi0": readout_moments(matrices, 0)["local_gain"],
        "g_C_min": ext["minimum"], "g_C_max": ext["maximum"],
        "g_C_min_phase_rad": ext["minimum_phase"], "g_C_max_phase_rad": ext["maximum_phase"],
        "g_pair": pair_gain, "e_B_min_percent": 100*e_b[0], "e_B_max_percent": 100*e_b[1],
        "e_B_pair_percent": 100*(g_b/pair_gain-1),
        "e_D_min_percent": 100*e_d[0], "e_D_max_percent": 100*e_d[1],
        "e_D_pair_percent": 100*(g_d/pair_gain-1),
        "B_over_D_minus_one_percent": 100*(g_b/g_d-1),
        "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
    }, matrices, d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    folder = (ROOT/args.run_dir).resolve()
    output_root = (ROOT/"local-results").resolve()
    if folder == output_root or not folder.is_relative_to(output_root) or folder.exists():
        parser.error("Require a fresh output directory below local-results")
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    sources = ["analyze_transfer.py", "gain_phase.py", "phase_pair.py", "piston_basis.py",
               "simulation.py", "modal_simulation.py", "run_pilot.py"]
    source_hashes = {name: digest(ROOT/name) for name in sources}
    identities = {}

    def load(name, family="SLM"):
        manifest, gram, hashes = load_gram(name, family)
        assert gram.shape == (512, 512, 3, 3) and np.isfinite(gram).all()
        identities[name] = {**hashes, "configuration": manifest["config"], "generator_identity": manifest["identity"]}
        return gram

    with WallDeadline(60):
        selfcheck()
        legacy = dark(load("finite-s3-domain6-128"))
        _, j = normalize_response(*evaluate(legacy, 0))
        archived = j/np.sum(j*j)
        del legacy
        assert hashlib.sha256(archived.tobytes()).hexdigest() == "81fc8a8066a6af2613e8b05ca4657368bf9562ea95bb746f178fb7e6ca3f4972"
        gram15 = load("modal-domain15-256")
        native15, j15 = native(gram15)
        pixel_x = np.arange(512) + .5 - 256
        quadrant = np.broadcast_to(np.sign(pixel_x)[None, :], (512, 512)).copy()
        quadrant_sensitivity = float(np.sum(quadrant*j15))
        assert np.isfinite(quadrant_sensitivity) and abs(quadrant_sensitivity) > 1e-5
        quadrant /= quadrant_sensitivity
        rows = []
        baseline_matrices = baseline_d = None
        for domain in (6, 12, 15):
            name = f"modal-domain{domain}-256"
            gram = gram15 if domain == 15 else load(name)
            opaque = load(B_TARGET_RUNS[name], "B")
            assert not np.any(opaque[..., 2, :]) and not np.any(opaque[..., :, 2])
            native_w, _ = native(gram)
            weights = [("archived_fixed", archived), ("native15_fixed", native15), ("domain_native", native_w)]
            if domain == 15:
                weights.append(("global_quadrant_x", quadrant))
            for label, w in weights:
                record, matrices, d = row(domain, label, gram, opaque, w)
                rows.append(record)
                if label == "domain_native":
                    assert abs(record["g_D"]-1) < 2e-12
                if domain == 15 and label == "archived_fixed":
                    baseline_matrices, baseline_d = matrices, d
                    assert abs(record["e_B_min_percent"] + 1.3144) < 5e-5
                    assert abs(record["e_B_max_percent"] - 2.7810) < 5e-5
                    assert abs(record["e_B_pair_percent"] - .07943) < 5e-6
            del gram, opaque
        baseline = next(r for r in rows if r["domain_D"] == 15 and r["readout"] == "archived_fixed")
        a_grid = np.linspace(-1, 1, 201)
        zero = readout_moments(baseline_d, 0)
        t_d = np.array([(readout_moments(baseline_d, float(a))["weighted_mean"]-zero["weighted_mean"])/zero["local_gain"] for a in a_grid])
        peak = int(np.argmax(abs(t_d)))
        flux_rows = []
        for a in a_grid:
            coeff = gain_coefficients(baseline_matrices, float(a))
            flux_rows.append({"piston_rad": float(a), "strip_fraction_of_pooled_detected_flux": float(baseline_matrices[0, 2, 2].real/coeff[0]),
                              "relative_flux_interference_amplitude": float(2*abs(coeff[2])/coeff[0])})
        assert all(r["relative_flux_interference_amplitude"] < 1 for r in flux_rows)
        d_check = {"T_D_minus1_rad": float(t_d[0]), "T_D_plus1_rad": float(t_d[-1]),
                   "sampled_peak_abs_T_D_rad": float(abs(t_d[peak])), "sampled_peak_piston_rad": float(a_grid[peak]),
                   "sample_count": len(a_grid), "piston_step_rad": .01,
                   "max_abs_transfer_mrad": float(abs(t_d[peak])*max(abs(baseline[k]) for k in ("e_D_min_percent", "e_D_max_percent"))*10),
                   "max_transfer_calibration_phase_rad": baseline["g_C_min_phase_rad"]}
        assert abs(d_check["max_abs_transfer_mrad"]-24.106) < .001
        for name, old in source_hashes.items():
            assert digest(ROOT/name) == old
        peak_bytes = psutil.Process().memory_info().peak_wset
        assert peak_bytes < 2_000_000_000
        folder.mkdir(parents=True)
        for filename, data in (("readout-domain-results.csv", rows), ("flux-versus-piston.csv", flux_rows)):
            with (folder/filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        result = {"status": "PASS", "started_utc": started, "elapsed_seconds": time.perf_counter()-start,
                  "peak_rss_bytes": peak_bytes, "process_id": os.getpid(), "python": sys.version,
                  "platform": platform.platform(), "numpy": np.__version__, "psutil": psutil.__version__,
                  "classification": "ordinary contraction only; no optical propagation",
                  "input_identities": identities,
                  "source_sha256_raw": source_hashes, "script_sha256": digest(Path(__file__)),
                  "quadrant_definition": "sign(x_detector), all cropped pixels, normalized by analytic native15 D zero-piston sensitivity",
                  "quadrant_raw_D_sensitivity_per_rad": quadrant_sensitivity,
                  "results": rows, "D_transfer_endpoints": d_check,
                  "flux_at_zero": flux_rows[100],
                  "flux_sampled_max_interference": max(flux_rows, key=lambda r:r["relative_flux_interference_amplitude"]),
                  "flux_sampled_max_strip_fraction": max(flux_rows, key=lambda r:r["strip_fraction_of_pooled_detected_flux"]),
                  "checks": ["synthetic pixel/contraction quotient", "finite-difference derivative", "raw-pair identity", "gain coefficient identity", "archived weight checksum", "baseline published B reproduction", "native D normalization", "B/D decomposition", "D peak reconciliation", "source identity and resource cap"],
                  "output_sha256": {n:digest(folder/n) for n in ("readout-domain-results.csv", "flux-versus-piston.csv")}}
        (folder/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status", "elapsed_seconds", "peak_rss_bytes", "quadrant_raw_D_sensitivity_per_rad", "D_transfer_endpoints", "flux_at_zero", "flux_sampled_max_interference")}, indent=2))
    for record in rows:
        print({k:record[k] for k in ("domain_D", "readout", "g_D", "g_B", "e_B_min_percent", "e_B_max_percent", "e_B_pair_percent")})


if __name__ == "__main__":
    main()
