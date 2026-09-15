from __future__ import annotations

import inspect

import numpy as np

import config
import methods


def test_demo_and_full_differ_only_in_replications() -> None:
    demo = config.get_experiment_config("demo")
    full = config.get_experiment_config("full")
    assert (demo.replications, full.replications) == (10, 1000)
    assert demo.scientific_payload() == full.scientific_payload()


def test_public_methods_default_to_strict_c() -> None:
    functions = (
        methods.make_sampler,
        methods.score_scan,
        methods.multiplier_statistics_from_draws,
        methods.multiplier_statistics,
        methods.fixed_look_test,
    )
    assert all(
        inspect.signature(function).parameters["backend"].default == "c"
        for function in functions
    )
    assert inspect.signature(methods.OnlineGaussianProcessScore).parameters[
        "backend"
    ].default == "c"


def test_python_and_c_backends_are_numerically_matched() -> None:
    if not methods.fast_available():
        methods.build_native()
    report = methods.backend_parity_report()
    assert report["passed"] is True
    assert report["score"]["max_abs_z_error"] <= 1e-10
    assert np.isfinite(report["score"]["statistic_abs_error"])
