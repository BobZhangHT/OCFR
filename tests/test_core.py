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


def test_zero_lag_weekly_exposure_has_native_parity() -> None:
    cases = np.linspace(100.0, 1000.0, 20)
    delay = np.array([0.6, 0.3, 0.08, 0.02])
    total_python, post_python = methods.exposure_components(
        cases, delay, lag_offset=0
    )
    total_c, post_c = methods.FastKernels().gp_exposure_components(
        cases, delay, lag_offset=0
    )
    assert np.isclose(total_python[0], cases[0] * delay[0])
    np.testing.assert_allclose(total_c, total_python, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(post_c, post_python, atol=1e-12, rtol=0.0)
