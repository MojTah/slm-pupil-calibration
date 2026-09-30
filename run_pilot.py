"""Run bounded, resumable optical cases; each completed case is immutable.

Example: python run_pilot.py --run-dir local-results/pilot-v1
Use --resume only with unchanged source/config identities. Budget admission is
checked between cases; this is not an OS-enforced time or memory limit.
"""
import argparse
import _thread
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import median
import threading
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import psutil

from simulation import Configuration, OpticalModel, compare_response, crop_detector, normalize_response

ROOT = Path(__file__).resolve().parent


class WallDeadline:
    """Request interruption at a Python boundary; not an OS-enforced limit."""
    def __init__(self, seconds):
        self.expired = threading.Event()
        self.timer = threading.Timer(max(0.0, seconds), self._expire)
        self.timer.daemon = True

    def _expire(self):
        self.expired.set()
        _thread.interrupt_main()

    def __enter__(self):
        self.timer.start()
        return self

    def __exit__(self, *error):
        self.timer.cancel()
        self.timer.join()


def forecast_rate(rates, modulation):
    """Robust admission estimate; old durations remain in their case records."""
    return median(rates.get(modulation) or [v for group in rates.values() for v in group] or [0.0])


def text_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "pilot-config.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wall-budget", type=float, default=110)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-run", type=Path, help="Copy verified cases with identical physics/config into a new run")
    args = parser.parse_args()
    if args.resume and args.reuse_run:
        parser.error("Choose resume or reuse-run, not both")
    if not 0 < args.wall_budget <= 600:
        parser.error("Wall budget must be positive and at most 600 seconds; obey the recorded phase allocation")
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to((ROOT / "local-results").resolve()):
        parser.error("Run directory must be under this study's local-results")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["schema_version"] != 1:
        parser.error("Unsupported configuration schema")
    # Construct all geometries before writing a run directory.
    geometries = [Configuration(**config["geometry"], samples_per_pixel=s) for s in config["samples_per_pixel"]]
    filtered_cases = config.get("filtered_cases", ["B", "C"])
    if filtered_cases not in (["B", "C"], ["B", "D", "C"]):
        parser.error("Filtered cases must be B,C or B,D,C in that order")
    detector_diameters = config.get("detector_diameters", config["geometry"]["domain_diameters"])
    if (not isinstance(detector_diameters, int) or isinstance(detector_diameters, bool) or
            not 1 <= detector_diameters <= config["geometry"]["domain_diameters"]):
        parser.error("Detector diameters must be a positive integer within the propagation domain")
    if not geometries or len(set(config["samples_per_pixel"])) != len(geometries):
        parser.error("Require nonempty, unique resolutions")
    preflight = json.loads((ROOT / "preflight.json").read_text(encoding="utf-8"))
    if preflight["exit_code"] != 0:
        parser.error("Preflight did not pass")
    for name, value in preflight["source_hashes"].items():
        if value != text_hash(ROOT / name):
            parser.error("Preflight source identity is stale: " + name)
    identity = {name: text_hash(ROOT / name) for name in ("simulation.py", "run_pilot.py", "test_simulation.py")}
    identity["config"] = text_hash(args.config)
    output = run_dir / "result.json"
    if args.resume:
        state = json.loads(output.read_text(encoding="utf-8"))
        if state["identity"] != identity:
            parser.error("Resume requires the exact saved source/config identity")
        for case in state["cases"]:
            if hashlib.sha256((run_dir / case["artifact"]).read_bytes()).hexdigest() != case["sha256"]:
                parser.error("Saved case artifact failed its checksum")
    else:
        if run_dir.exists():
            parser.error("Run directory already exists; use unchanged --resume or a new name")
        run_dir.mkdir(parents=True)
        state = {"started_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
                 "config": config, "cases": [], "elapsed_wall_seconds": 0.0,
                 "status": "RUNNING", "independent_scientific_review": "PENDING",
                 "scope": "Exploratory scalar model; no lab validation or novelty verdict"}
        if args.reuse_run:
            previous_dir = args.reuse_run if args.reuse_run.is_absolute() else ROOT / args.reuse_run
            previous_dir = previous_dir.resolve()
            if not previous_dir.is_relative_to((ROOT / "local-results").resolve()):
                parser.error("Reuse source must be under this study's local-results")
            previous_path = previous_dir / "result.json"
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            for name in ("simulation.py", "test_simulation.py", "config"):
                if previous["identity"][name] != identity[name]:
                    parser.error("Reuse requires identical physics, tests and configuration: " + name)
            # A changed runner is recorded, never retroactively attributed to old cases.
            for old in previous["cases"]:
                source = (previous_dir / old["artifact"]).resolve()
                if source.parent != previous_dir or source.name != old["artifact"]:
                    parser.error("Invalid saved artifact path")
                if hashlib.sha256(source.read_bytes()).hexdigest() != old["sha256"]:
                    parser.error("Reuse artifact failed checksum")
                shutil.copy2(source, run_dir / source.name)
                state["cases"].append({**old, "execution_identity": old.get("execution_identity", previous["identity"]),
                                       "reused_from": str(previous_dir)})
            state["reuse"] = {"run": str(previous_dir), "cases": len(previous["cases"]),
                              "result_sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                              "original_execution_seconds": previous["elapsed_wall_seconds"]}
    completed = {c["name"]: c for c in state["cases"]}
    previous_elapsed = state["elapsed_wall_seconds"]
    previous_cpu = state.get("cpu_seconds", 0.0)
    start = time.perf_counter()
    process = psutil.Process()
    cpu_start = sum(process.cpu_times()[:2])
    stop_reason = None
    rates = {}
    for c in state["cases"]:
        rates.setdefault(c["modulation"], []).append(c["elapsed_seconds"] / c["work_proxy"])

    def checkpoint():
        memory = process.memory_info()
        state.update(elapsed_wall_seconds=previous_elapsed + time.perf_counter() - start,
                     peak_rss_bytes=max(state.get("peak_rss_bytes", 0), getattr(memory, "peak_wset", memory.rss)),
                     cpu_seconds=previous_cpu + sum(process.cpu_times()[:2]) - cpu_start,
                     updated_utc=datetime.now(timezone.utc).isoformat(),
                     process_id=os.getpid())
        write_json(output, state)

    deadline = None
    try:
        state["status"] = "RUNNING"
        checkpoint()
        for geometry in geometries:
            if stop_reason:
                break
            # Conservative field-array working-set estimate, checked before allocation.
            n = geometry.diameter_pixels * geometry.domain_diameters * geometry.samples_per_pixel
            estimated_bytes = n * n * 16 * 20
            if estimated_bytes > 2_000_000_000 or psutil.virtual_memory().available < estimated_bytes * 2:
                stop_reason = "Memory admission check"
                break
            model = OpticalModel(geometry)
            for modulation in config["modulations"]:
                if stop_reason:
                    break
                specs = [("A", None)] + [(c, r) for r in config["iris_radii"] for c in filtered_cases]
                for case, radius in specs:
                    name = f"s{geometry.samples_per_pixel}_m{modulation:g}_{case}" + ("" if radius is None else f"_r{radius:g}")
                    if name in completed:
                        continue
                    work = n * n * (1 if modulation == 0 else config["modulation_angles"])
                    estimate = max(1.0, 1.5 * forecast_rate(rates, modulation) * work)
                    if previous_elapsed + time.perf_counter() - start + estimate > args.wall_budget:
                        stop_reason = "Time admission check; completed cases preserved"
                        break
                    t0 = time.perf_counter()
                    state["active_case"] = name
                    checkpoint()
                    print(json.dumps({"case_started": name, "elapsed_total": state["elapsed_wall_seconds"]}), flush=True)
                    deadline = WallDeadline(args.wall_budget - previous_elapsed - (time.perf_counter() - start))
                    with deadline:
                        field, derivative = model.relay(case, iris_radius=radius or config["iris_radii"][0])
                        full_image, full_j = model.sensor(field, derivative, modulation, config["modulation_angles"])
                    # Deadline is inactive before metric/artifact publication.
                    side = geometry.diameter_pixels * detector_diameters
                    image, raw_j = crop_detector(full_image, side), crop_detector(full_j, side)
                    y, j = normalize_response(image, raw_j)
                    reference_name = f"s{geometry.samples_per_pixel}_m{modulation:g}_A"
                    if case == "A":
                        y_ref, j_ref = y, j
                    else:
                        with np.load(run_dir / completed[reference_name]["artifact"]) as saved:
                            y_ref, j_ref = saved["intensity"], saved["response"]
                    metrics = compare_response(j_ref, j)
                    if not metrics["reference_near_zero"]:
                        metrics["zero_piston_linear_bias_rad"] = float(np.sum(j_ref * (y - y_ref)) / np.sum(j_ref ** 2))
                    metrics.update(throughput_relative_to_opaque=float(image.sum() / model.power(model.bright)),
                                   physical_detector_flux=float(image.sum()),
                                   physical_flux_derivative=float(raw_j.sum()),
                                   total_propagated_flux=float(full_image.sum()),
                                   exterior_flux=float(full_image.sum() - image.sum()),
                                   captured_fraction=float(image.sum() / full_image.sum()),
                                   shadow_to_bright_intensity=float(np.mean(abs(field[model.shadow]) ** 2) / np.mean(abs(field[model.bright]) ** 2)))
                    if case in ("C", "D"):
                        bname = f"s{geometry.samples_per_pixel}_m{modulation:g}_B_r{radius:g}"
                        with np.load(run_dir / completed[bname]["artifact"]) as saved:
                            metrics["relative_to_same_iris_opaque"] = compare_response(saved["response"], j)
                    if case == "C" and "D" in filtered_cases:
                        dname = f"s{geometry.samples_per_pixel}_m{modulation:g}_D_r{radius:g}"
                        with np.load(run_dir / completed[dname]["artifact"]) as saved:
                            metrics["relative_to_bright_carrier"] = compare_response(saved["response"], j)
                    artifact = name + ".npz"
                    target = run_dir / artifact
                    if target.exists():
                        raise RuntimeError("Refusing to overwrite an unrecorded case: " + artifact)
                    temporary = target.with_suffix(".tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, intensity=y, response=j,
                                            pupil_intensity=model.bin_intensity(abs(field) ** 2),
                                            profile_x=model.x, profile_field=field[model.n // 2])
                    temporary.replace(target)
                    elapsed = time.perf_counter() - t0
                    record = {"name": name, "geometry": asdict(geometry), "modulation": modulation,
                              "iris_radius": radius, "case": case, "metrics": metrics,
                              "detector_diameters": detector_diameters,
                              "artifact": artifact, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                              "artifact_bytes": target.stat().st_size, "elapsed_seconds": elapsed, "work_proxy": work,
                              "execution_identity": identity, "profile_y_pixels": float(model.x[model.n // 2])}
                    state["cases"].append(record)
                    completed[name] = record
                    state.pop("active_case", None)
                    rates.setdefault(modulation, []).append(elapsed / work)
                    checkpoint()
                    print(json.dumps({"case": name, "gain": metrics["gain"], "shape_error": metrics["shape_error"],
                                      "elapsed_total": state["elapsed_wall_seconds"]}), flush=True)
                    if state["peak_rss_bytes"] > 2_000_000_000:
                        stop_reason = "Measured memory ceiling exceeded"
                        break
                    del field, derivative, image, raw_j, full_image, full_j, y, j
            del model
        state["status"] = "PARTIAL" if stop_reason else "COMPLETE_CASES"
        state["stop_reason"] = stop_reason
    except KeyboardInterrupt:
        timed_out = deadline is not None and deadline.expired.is_set()
        state["status"] = "PARTIAL" if timed_out else "INTERRUPTED"
        state["stop_reason"] = "Wall deadline expired inside execution; completed cases preserved" if timed_out else "User interrupt; completed cases preserved"
        state["unfinished_case"] = state.pop("active_case", None)
        if not timed_out:
            raise
    except BaseException as error:
        state["status"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # The context has canceled/joined the timer before this durable write.
        checkpoint()
    print(json.dumps({"status": state["status"], "completed_cases": len(state["cases"]),
                      "elapsed_seconds": state["elapsed_wall_seconds"], "peak_rss_bytes": state["peak_rss_bytes"],
                      "stop_reason": state.get("stop_reason")}), flush=True)


if __name__ == "__main__":
    main()
