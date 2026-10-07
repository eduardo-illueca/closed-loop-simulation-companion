"""Demonstrator Performance Benchmarking (Action A33).

Benchmarks runtime, memory usage, and hardware parameters for running the
closed-loop process mining and causal refinement workflow on the 294,734
trajectory dataset (matching the published SCREAM tutorial cohort scale).

Also explicitly records that the framework operates as a cycle-based diagnostic
workflow for iterative batched analysis, not real-time continuous stream processing.

Usage:
    python -m closed_loop_sim.benchmark
    python -m closed_loop_sim.benchmark --n 10000
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path

from closed_loop_sim.config import base_cell
from closed_loop_sim.discovery import discover
from closed_loop_sim.estimation import estimate, estimate_arm
from closed_loop_sim.eventlog import build_event_log_data
from closed_loop_sim.metrics import structural_metrics
from closed_loop_sim.refinement import ARMS, run_iterative_refinement, MechanicalProposer, RulesGate

OUT = Path("results")
OUT.mkdir(exist_ok=True)


def benchmark_demonstrator(n: int = 294734, seed: int = 20260610, out_dir: str | Path = OUT) -> dict:
    """Benchmark the full closed-loop pipeline for n trajectories."""
    print(f"[benchmark] Starting demonstrator benchmark on n={n:,} trajectories...")
    
    tracemalloc.start()
    t_start = time.perf_counter()

    # 1. Data Generation & Event Log Construction
    t0 = time.perf_counter()
    cfg = base_cell()
    cfg.n = n
    data = build_event_log_data(cfg.n, cfg, seed)
    t_datagen = time.perf_counter() - t0
    print(f"  - Event log constructed ({len(data.log):,} event rows): {t_datagen:.2f} s")

    # 2. Process Discovery (DFG extraction & transition risk calculation)
    t0 = time.perf_counter()
    disc = discover(data.log, data.visit_counts, compute_activity_dfg=False)
    t_discovery = time.perf_counter() - t0
    print(f"  - Process discovery (DFG & state risks): {t_discovery:.2f} s")

    # 3. Iterative Refinement Cycles across Arms
    t0 = time.perf_counter()
    ref_a = ARMS["A"](disc)
    ref_b = run_iterative_refinement(MechanicalProposer(), RulesGate(), disc, max_cycles=5)
    ref_c = ARMS["C"](disc)
    struct_b = structural_metrics(ref_b)
    t_refinement = time.perf_counter() - t0
    print(f"  - Refinement cycles (Arms A, B, C; B cycles={ref_b.cycles_run}): {t_refinement:.2f} s")

    # 4. Phase 4 Estimand Estimation
    t0 = time.perf_counter()
    est_b = estimate_arm(data, cfg.dgp.t_max, ref_b)
    t_estimation = time.perf_counter() - t0
    print(f"  - Phase 4 estimation & estimand recovery: {t_estimation:.2f} s")

    t_total = time.perf_counter() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()


    peak_mb = peak_mem / (1024 * 1024)
    peak_gb = peak_mem / (1024 * 1024 * 1024)

    # Hardware & System Metadata
    hw_info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "architecture": platform.architecture()[0],
        "cpu_count_logical": os.cpu_count(),
    }

    result = {
        "action_item": "A33",
        "n_trajectories": n,
        "n_event_log_rows": len(data.log),
        "seed": seed,
        "framework_operating_mode": (
            "Cycle-based diagnostic workflow (iterative batched analysis of longitudinal event logs), "
            "not real-time continuous stream processing."
        ),
        "timing_sec": {
            "data_generation": round(t_datagen, 3),
            "process_discovery": round(t_discovery, 3),
            "refinement_cycles": round(t_refinement, 3),
            "estimation": round(t_estimation, 3),
            "total_runtime": round(t_total, 3),
        },
        "memory": {
            "peak_memory_mb": round(peak_mb, 2),
            "peak_memory_gb": round(peak_gb, 4),
        },
        "refinement_metrics": {
            "arm_b_cycles_run": ref_b.cycles_run,
            "arm_b_stopped_reason": ref_b.stopped_reason,
            "arm_b_precision": struct_b["precision"],
            "arm_b_recall": struct_b["recall"],
            "arm_b_f1": struct_b["f1"],
        },
        "hardware": hw_info,
    }

    out_path = Path(out_dir) / "benchmark.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[benchmark] Completed benchmark on {n:,} trajectories in {t_total:.2f} s (Peak Memory: {peak_mb:.2f} MB)")
    print(f"[benchmark] Output written to: {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Action A33 Benchmarking Script")
    parser.add_argument("--n", type=int, default=294734, help="Number of trajectories (default: 294734)")
    args = parser.parse_args()
    benchmark_demonstrator(n=args.n)


if __name__ == "__main__":
    main()
