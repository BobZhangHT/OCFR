"""Independent numerical checks for the opt-in log-risk-ratio penalty."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit, gammaln
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import methods as m


def sample():
    rng = np.random.default_rng(701)
    design = np.column_stack([np.r_[np.full(40, 700.), np.full(30, 100.)],
                              np.r_[np.zeros(40), np.full(30, 600.)]])
    offset = np.linspace(.1, .5, len(design))
    mean = offset + design @ np.array([.02, .06])
    y = rng.negative_binomial(8., 8. / (8. + mean))
    return y, offset, design


def independent_objective(theta, y, offset, design, precision):
    rates, phi = np.exp(theta[:2]), np.exp(theta[2])
    mu = np.maximum(offset + design @ rates, m.MU_FLOOR)
    ll = np.sum(gammaln(y + phi) - gammaln(phi) - gammaln(y + 1)
                + phi * np.log(phi) + y * np.log(mu)
                - (y + phi) * np.log(phi + mu))
    return -ll + .5 * precision * (theta[1] - theta[0])**2


@pytest.mark.parametrize('precision', [0., 1., 4.])
def test_gradient_against_independent_finite_differences(precision):
    y, offset, design = sample()
    theta = np.log([.025, .05, 12.])
    value, gradient = m._regularized_nll_and_grad(theta, y, offset, design, precision)
    h = 1e-5
    numerical = np.array([(independent_objective(theta + h * e, y, offset, design, precision)
                           - independent_objective(theta - h * e, y, offset, design, precision)) / (2*h)
                          for e in np.eye(3)])
    assert value == pytest.approx(independent_objective(theta, y, offset, design, precision))
    np.testing.assert_allclose(gradient, numerical, atol=2e-6, rtol=2e-5)


def test_fit_improves_penalized_objective_and_shrinks_contrast():
    y, offset, design = sample()
    ordinary = m.fit_nb_identity(y, offset, design)
    fit = m.fit_nb_regularized_identity(y, offset, design, initial_fit=ordinary)
    assert fit.success
    penalty = lambda f: -f.loglik + .5 * np.log(f.rates[1] / f.rates[0])**2
    assert penalty(fit) <= penalty(ordinary) + 1e-10
    assert abs(np.log(fit.rates[1] / fit.rates[0])) < abs(np.log(ordinary.rates[1] / ordinary.rates[0]))
    assert fit.loglik == pytest.approx(m.nb_loglik(y, offset + design @ fit.rates, fit.phi))


def test_zero_precision_returns_identical_baseline_and_default_is_unchanged(monkeypatch):
    y, offset, design = sample()
    ordinary = m.fit_nb_identity(y, offset, design)
    assert m.fit_nb_regularized_identity(y, offset, design, precision=0, initial_fit=ordinary) is ordinary
    def forbidden(*args, **kwargs):
        raise AssertionError('ordinary fitting called the opt-in regularizer')
    monkeypatch.setattr(m, 'fit_nb_regularized_identity', forbidden)
    repeated = m.fit_nb_identity(y, offset, design)
    np.testing.assert_array_equal(repeated.rates, ordinary.rates)
    assert repeated.phi == ordinary.phi
    assert repeated.status == ordinary.status


def test_optimizer_failure_cannot_promote_unsuccessful_baseline(monkeypatch):
    y, offset, design = sample()
    ordinary = m.fit_nb_identity(y, offset, design)
    unsuccessful = m.NBFit(ordinary.rates, ordinary.phi, ordinary.loglik, False, 'failed', 0)
    monkeypatch.setattr(m, 'minimize', lambda *a, **kw: SimpleNamespace(
        success=False, x=a[1], fun=1., nit=1))
    fit = m.fit_nb_regularized_identity(y, offset, design, initial_fit=unsuccessful)
    assert not fit.success
    assert fit.status == 'nonconvergence:regularized_baseline_fallback'
    np.testing.assert_array_equal(fit.rates, ordinary.rates)


def test_optimization_worse_than_baseline_is_explicit_failure(monkeypatch):
    y, offset, design = sample()
    ordinary = m.fit_nb_identity(y, offset, design)
    monkeypatch.setattr(m, 'minimize', lambda *a, **kw: SimpleNamespace(
        success=True, x=a[1], fun=1e100, nit=1))
    fit = m.fit_nb_regularized_identity(y, offset, design, initial_fit=ordinary)
    assert not fit.success
    assert fit.loglik == pytest.approx(ordinary.loglik)


def test_powell_retry_after_iteration_limit_preserves_objective():
    y, offset, design = sample()
    ordinary = m.fit_nb_identity(y, offset, design)
    fit = m.fit_nb_regularized_identity(y, offset, design,
                                        initial_fit=ordinary, maxiter=1)
    assert fit.success
    assert fit.status == 'regularized_log_rr:powell_fallback'
    penalty = lambda f: -f.loglik + .5 * np.log(f.rates[1] / f.rates[0])**2
    assert penalty(fit) <= penalty(ordinary)
    complete = m.fit_nb_regularized_identity(y, offset, design, initial_fit=ordinary)
    np.testing.assert_allclose(fit.rates, complete.rates, rtol=2e-5)


@pytest.mark.parametrize('left,right', [(.02, .04), (.04, .02), (.03, .03)])
def test_large_information_mean_data_approximately_recovers_risks(left, right):
    design = np.zeros((100, 2))
    design[:50, 0] = 1e5
    design[50:, 1] = 1e5
    y = design @ [left, right]
    fit = m.fit_nb_regularized_identity(y, np.zeros(100), design)
    assert fit.success
    np.testing.assert_allclose(fit.rates, [left, right], rtol=.002)


def test_sandwich_covariance_and_transforms_against_independent_matrix_arithmetic():
    y, offset, design = sample()
    fit = m.fit_nb_regularized_identity(y, offset, design)
    ci = m.regularized_parameter_intervals(fit, offset, design, confidence=.9)
    ordinary = m.regularized_parameter_intervals(fit, offset, design, covariance='unpenalized')
    mu = offset + design @ fit.rates
    variance = mu + mu**2 / fit.phi
    info = (design.T / variance) @ design
    p = np.diag(fit.rates)
    info_log = p @ info @ p
    c = np.array([-1., 1.])
    a = info_log + np.outer(c, c)
    inverse = np.linalg.solve(a, np.eye(2))
    expected_log = inverse @ info_log @ inverse.T
    expected = p @ expected_log @ p
    np.testing.assert_allclose(ci.covariance, expected, rtol=1e-10)
    np.testing.assert_allclose(ci.information, info)
    assert np.linalg.eigvalsh(ci.covariance)[0] > 0
    assert np.linalg.eigvalsh(ordinary.covariance - ci.covariance)[0] > -1e-14
    assert ci.log_risk_ratio_standard_error**2 == pytest.approx(c @ expected_log @ c)
    assert ci.log_risk_ratio_standard_error < ordinary.log_risk_ratio_standard_error
    z = norm.ppf(.95)
    np.testing.assert_allclose(ci.cfr_lower, expit(ci.cfr_logit - z * ci.cfr_logit_standard_errors))
    np.testing.assert_allclose(ci.cfr_upper, expit(ci.cfr_logit + z * ci.cfr_logit_standard_errors))
    assert ci.risk_ratio_lower == pytest.approx(np.exp(ci.log_risk_ratio - z * ci.log_risk_ratio_standard_error))
    assert ci.risk_ratio_upper == pytest.approx(np.exp(ci.log_risk_ratio + z * ci.log_risk_ratio_standard_error))
    zero = m.regularized_parameter_intervals(fit, offset, design, precision=0)
    np.testing.assert_array_equal(zero.covariance, ordinary.covariance)


@pytest.mark.parametrize('design', [np.zeros((5, 2)), np.ones((5, 2)),
                                    np.column_stack([np.ones(5), np.full(5, 1e-20)])])
def test_zero_or_unidentifiable_information_has_no_interval(design):
    fit = m.NBFit(np.array([.02, .04]), 10., 0., True, 'test', 0)
    with pytest.raises(np.linalg.LinAlgError):
        m.regularized_parameter_intervals(fit, np.ones(5), design)


@pytest.mark.parametrize('precision', [-1., np.nan, np.inf])
def test_invalid_precision_rejected(precision):
    y, offset, design = sample()
    with pytest.raises(ValueError):
        m.fit_nb_regularized_identity(y, offset, design, precision=precision)


@pytest.mark.parametrize('bad', ['negative_y', 'negative_offset', 'negative_design', 'nan', 'one_column'])
def test_invalid_fit_inputs_rejected(bad):
    y, offset, design = sample()
    y = y.astype(float)
    if bad == 'negative_y':
        y[0] = -1
    elif bad == 'negative_offset':
        offset[0] = -1
    elif bad == 'negative_design':
        design[0, 0] = -1
    elif bad == 'nan':
        design[0, 0] = np.nan
    else:
        design = design[:, :1]
    with pytest.raises(ValueError):
        m.fit_nb_regularized_identity(y, offset, design)


def test_portable_exports():
    assert 'fit_nb_regularized_identity' in m.__all__
    assert 'regularized_parameter_intervals' in m.__all__


def test_swapping_left_and_right_swaps_regularized_fit():
    y, offset, design = sample()
    fit = m.fit_nb_regularized_identity(y, offset, design)
    swapped = m.fit_nb_regularized_identity(y, offset, design[:, ::-1])
    assert fit.success and swapped.success
    np.testing.assert_allclose(fit.rates, swapped.rates[::-1], rtol=2e-6)
    assert fit.phi == pytest.approx(swapped.phi, rel=2e-5)


def test_zero_exposure_fit_and_weak_identifiable_information():
    zero = m.fit_nb_regularized_identity(np.zeros(5), np.zeros(5), np.zeros((5, 2)))
    assert np.all(np.isfinite(zero.rates))
    assert np.isfinite(zero.loglik)
    fit = m.NBFit(np.array([.02, .04]), 10., 0., True, 'test', 0)
    design = 1e-6 * np.array([[1., 0.], [0., 1.], [1., 1.]])
    interval = m.regularized_parameter_intervals(fit, np.ones(3), design)
    assert np.all(np.isfinite(interval.covariance))
    assert np.linalg.eigvalsh(interval.covariance)[0] >= -1e-8


def test_module_imports_alone_without_project_package(tmp_path):
    import shutil
    import subprocess
    shutil.copyfile(Path(m.__file__), tmp_path / 'methods.py')
    result = subprocess.run([sys.executable, '-c',
        'import methods; assert callable(methods.fit_nb_regularized_identity); '
        'assert callable(methods.regularized_parameter_intervals)'],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
