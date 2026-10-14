"""Source coefficient identity and iris-boundary audit; no optical propagation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parent
import numpy as np
import psutil
from modal_simulation import ModalModel
from simulation import Configuration, OpticalModel, ft
from run_pilot import WallDeadline


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    folder = (ROOT/args.run_dir).resolve()
    output_root = (ROOT/"local-results").resolve()
    if folder == output_root or not folder.is_relative_to(output_root) or folder.exists():
        parser.error("Require a fresh output directory below local-results")
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    sources = {name:digest(ROOT/name) for name in ("modal_simulation.py", "simulation.py", "run_pilot.py", "piston_basis.py")}
    records, checks = [], []
    with WallDeadline(60):
        for domain in (6, 12, 15):
            model = OpticalModel(Configuration(samples_per_pixel=1, domain_diameters=domain))
            centered = model.iris(8.)
            shifted = np.roll(model.iris(8., .125), -round(.125*model.n), axis=1)
            for name, mask in (("centered_only", centered & ~shifted), ("shifted_only", shifted & ~centered)):
                ij = np.argwhere(mask)
                modes = [[int(round(model.f[x]*model.n)), int(round(model.f[y]*model.n))] for y, x in ij]
                if name == "centered_only":
                    centered_only = modes
                else:
                    shifted_only = modes
            records.append({"domain_D": domain, "centered_mode_count": int(centered.sum()),
                            "translated_shifted_mode_count": int(shifted.sum()),
                            "centered_only_kx_ky": centered_only, "shifted_only_kx_ky": shifted_only})
            del model, centered, shifted
        assert [(r["centered_mode_count"],r["translated_shifted_mode_count"]) for r in records] == [(7213,7213),(28917,28917),(45225,45221)]
        assert sorted(records[-1]["centered_only_kx_ky"]) == [[-72,-96],[-72,96],[72,-96],[72,96]]
        assert all(not r["shifted_only_kx_ky"] for r in records)
        model = ModalModel(Configuration(samples_per_pixel=1, domain_diameters=6), iris_radius=8.)
        g = model.grid
        box_hash_before = hashlib.sha256(model.box.tobytes()).hexdigest()
        common = g.iris(8.)
        fx, q = g.f[None, :], g.config.bright_carrier
        source_ratio = np.sinc(fx+q)/np.sinc(fx)
        # Remove the source sinc only; model.box remains the detector integration factor.
        def no_source_hold(command):
            return ft(command)*model.origin.conj()/g.n
        for a in (-.37, 0., 1.):
            command_b = g.bright*np.exp(1j*a*g.piston_shape)
            command_d = command_b*np.exp(2j*np.pi*q*g.pixel_x[None, :])
            for label, cb, cd in (("field", command_b, command_d),
                                  ("piston_tangent", 1j*g.piston_shape*command_b, 1j*g.piston_shape*command_d)):
                b = no_source_hold(cb)*common
                d = np.roll(no_source_hold(cd), -model.carrier_index, axis=1)*common
                relative = float(np.max(abs(b-d))/np.max(abs(b)))
                assert relative < 1e-12
                held_b = model.held_coefficients(cb)*common
                held_d = np.roll(model.held_coefficients(cd), -model.carrier_index, axis=1)*common
                held_error = float(np.max(abs(held_d-held_b*source_ratio))/np.max(abs(held_b)))
                assert held_error < 1e-12
                checks.append({"piston_rad":a, "quantity":label,
                               "common_mask_no_source_hold_relative_Linf_error":relative,
                               "held_sinc_ratio_relative_Linf_error":held_error})
        box_hash_after = hashlib.sha256(model.box.tobytes()).hexdigest()
        assert box_hash_before == box_hash_after
        values = [{"envelope_fx_per_pitch":x, "sinc_fx_plus_q_over_sinc_fx":float(np.sinc(x+.125)/np.sinc(x))} for x in (-.0625,0.,.0625)]
        assert np.allclose([r["sinc_fx_plus_q_over_sinc_fx"] for r in values], [1.,.9744953584044327,.949253021674191], atol=1e-12, rtol=0)
        assert all(digest(ROOT/name) == value for name,value in sources.items())
        peak_bytes = psutil.Process().memory_info().peak_wset
        assert peak_bytes < 2_000_000_000
        result = {"status":"PASS", "started_utc":started, "elapsed_seconds":time.perf_counter()-start,
                  "peak_rss_bytes":peak_bytes, "python":sys.version, "numpy":np.__version__,
                  "source_sha256_raw":sources, "script_sha256":digest(Path(__file__)),
                  "mask_counts":records, "held_source_ratio":values, "coefficient_checks":checks,
                  "detector_box_sha256_before":box_hash_before,"detector_box_sha256_after":box_hash_after,
                  "piston_shape":"-1/2 on bright left, +1/2 on bright right; actual physical pixel centers and origin",
                  "conclusion":"With source hold removed and one common translated iris, B and D field/tangent coefficients agree to roundoff. Subsequent identical operators imply g_B/g_D=1 analytically when the common readout gain is nonzero. This is not an archived 15D runtime gain calculation.",
                  "limitations":["No sensor propagation or new Gram construction", "No measured decomposition of archived 15D gain into hold and four-mode mask effects", "Only the 6D source coefficient canary executed; common-mask identity is analytic", "model.box unchanged; source coefficient sinc removed locally only"]}
        folder.mkdir(parents=True)
        (folder/"source-hold-results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
