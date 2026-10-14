#!/usr/bin/env python3
"""Measure Kramer evaluator accuracy, setup and warmed array timings.

No finite-element model is run. Use one process, no concurrent model jobs, and
the exact feature runtime. Output is an explicitly supplied JSON path; nothing
is written to a benchmark-specific package directory. SymPy substitution is
limited to the smallest batch and cannot be compared as a large-array result.
"""

import argparse
import gc
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
import tracemalloc
from types import SimpleNamespace

from mpi4py import MPI
import numpy as np
import sympy as sp
import underworld3 as uw


def measure(call, repetitions):
    values = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repetitions):
            start = perf_counter()
            call()
            values.append(perf_counter() - start)
    finally:
        if was_enabled:
            gc.enable()
    return dict(
        median_seconds=float(np.median(values)),
        minimum_seconds=min(values),
        maximum_seconds=max(values),
        iqr_seconds=float(np.percentile(values, 75) - np.percentile(values, 25)),
        samples_seconds=values,
    )


def scaled_error(actual, expected):
    scale = max(float(np.max(np.abs(expected))), 1e-14)
    absolute = float(np.max(np.abs(actual - expected)))
    if absolute > 2e-11 * scale + 2e-14:
        raise AssertionError(f"Evaluator disagrees with assess: absolute={absolute}, scale={scale}")
    return dict(maximum_absolute_error=absolute, maximum_magnitude_scaled_error=absolute / scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[100, 10000, 100000])
    parser.add_argument("--repetitions", type=int, default=7)
    arguments = parser.parse_args()
    if MPI.COMM_WORLD.size != 1 or arguments.repetitions < 7 or min(arguments.counts) < 1:
        raise ValueError(
            "Use one MPI process, at least seven repetitions and positive batch sizes."
        )
    x, y = sp.symbols("x y", real=True)
    mesh = SimpleNamespace(dim=2, X=sp.Matrix([[x, y]]))
    records = []
    for density, boundary in (
        ("delta", "free"),
        ("delta", "zero"),
        ("smooth", "free"),
        ("smooth", "zero"),
    ):
        case = f"{density}-{boundary}"
        for count in arguments.counts:
            radii = 0.5 + 0.5 * (np.arange(count) + 0.37) / count
            angles = np.arange(count) * 2.399963229728653
            points = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
            for field in ("velocity", "pressure", "radial_normal_stress"):
                start = perf_counter()
                oracle = uw.analytic.AssessCylindricalStokes(
                    mesh, density=density, boundary=boundary
                )
                assess_setup = perf_counter() - start
                if field == "radial_normal_stress":

                    def assess_evaluate():
                        return np.array(
                            [
                                (
                                    oracle._above if r >= oracle.r_int else oracle._below
                                ).radial_stress_cartesian(point)
                                for r, point in zip(radii, points)
                            ]
                        )

                else:

                    def assess_evaluate():
                        return oracle.evaluate(field, points)

                start = perf_counter()
                expected = assess_evaluate()
                assess_first = perf_counter() - start
                methods = [("assess_point_loop", assess_evaluate, assess_setup, 0.0, assess_first)]
                for cse in (False, True):
                    start = perf_counter()
                    solution = uw.analytic.CylindricalStokes(
                        mesh, density=density, boundary=boundary
                    )
                    construction = perf_counter() - start
                    solution.evaluation_cse = cse
                    start = perf_counter()
                    solution.compile_evaluators(field)
                    compilation = perf_counter() - start
                    evaluate = lambda sol=solution: sol.evaluate(field, points)
                    start = perf_counter()
                    actual = evaluate()
                    first = perf_counter() - start
                    scaled_error(actual, expected)
                    methods.append((f"numpy_cse_{cse}", evaluate, construction, compilation, first))
                if count == min(arguments.counts):
                    solution = uw.analytic.CylindricalStokes(
                        mesh, density=density, boundary=boundary
                    )

                    def substitute():
                        values = []
                        for radius, theta in zip(radii, angles):
                            branch = 1 if density == "delta" and radius >= solution.r_int else 0
                            expression = solution._polar_fields[branch][field]
                            terms = (
                                list(expression)
                                if isinstance(expression, sp.MatrixBase)
                                else [expression]
                            )
                            values.append(
                                [
                                    float(
                                        term.evalf(
                                            subs={solution._r: radius, solution._theta: theta}
                                        )
                                    )
                                    for term in terms
                                ]
                            )
                        return np.asarray(values).reshape(expected.shape)

                    scaled_error(substitute(), expected)
                    methods.append(("sympy_substitution_small_batch", substitute, None, None, None))
                for method, evaluate, construction, compilation, first in methods:
                    errors = scaled_error(evaluate(), expected)
                    timing = measure(evaluate, arguments.repetitions)
                    tracemalloc.start()
                    evaluate()
                    _current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    record = dict(
                        case=case,
                        geometry="annulus",
                        field=field,
                        points=count,
                        method=method,
                        construction_seconds=construction,
                        evaluator_generation_seconds=compilation,
                        first_evaluation_seconds=first,
                        python_heap_peak_bytes=peak,
                        **errors,
                        **timing,
                    )
                    records.append(record)
                    print(
                        f"{case:12s} {field:20s} N={count:6d} {method:30s} "
                        f"median={timing['median_seconds']:.6g}s",
                        flush=True,
                    )
    root = Path(uw.__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = dict(
        schema_version=1,
        records=records,
        uw_source=str(uw.__file__),
        uw_revision=revision,
        uw_dirty=bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        ),
        python=sys.version,
        numpy=np.__version__,
        sympy=sp.__version__,
        platform=platform.platform(),
        repetitions=arguments.repetitions,
        whole_run_peak_rss_bytes=peak if sys.platform == "darwin" else peak * 1024,
        note="Heap peaks are tracemalloc allocations, not per-method process RSS. Setup is reported separately; no FE solve.",
    )
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
