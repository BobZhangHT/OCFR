"""Exact two-risk Poisson point estimator: mathematics and explicit failures."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import methods as m


@pytest.mark.parametrize('precision', [0., 1.])
def test_exact_objective_gradient_and_offset(precision):
    x = np.array([[400., 0.], [200., 120.], [30., 250.], [0., 80.]])
    y, offset, theta = np.array([6., 10., 8., 0.]), np.array([.2, .5, .8, .1]), np.log([.015, .035])
    value, gradient = m._poisson_lograte_nll_and_grad(theta, y, offset, x, precision)
    numerical = approx_fprime(theta, lambda t: m._poisson_lograte_nll_and_grad(t, y, offset, x, precision)[0], 1e-6)
    assert np.allclose(gradient, numerical, rtol=2e-5, atol=1e-5)
    mu = offset+x@np.exp(theta)
    exact = np.sum(mu-y*np.log(mu)+gammaln(y+1))+.5*precision*(theta[1]-theta[0])**2
    assert value == pytest.approx(exact, abs=1e-12)


def test_oracle_recovery_same_cfr_and_exact_likelihood(monkeypatch):
    rng = np.random.default_rng(31)
    x = rng.uniform(0., 1200., (70, 2))
    offset = rng.uniform(0., 3., len(x))
    truth = np.array([.017, .043])
    y = offset+x@truth
    monkeypatch.setattr(m, 'nb_loglik', lambda *a: pytest.fail('Poisson used NB likelihood'))
    fit = m.fit_poisson_identity(y, offset, x)
    assert isinstance(fit, m.PoissonFit) and not hasattr(fit, 'phi')
    assert fit.success
    assert np.allclose(fit.rates, truth, rtol=1e-6)
    assert fit.loglik == pytest.approx(np.sum(y*np.log(y)-y-gammaln(y+1)), abs=1e-9)
    for precision in (0., 1.):
        equal = m.fit_poisson_identity(x@np.array([.02, .02]), np.zeros(len(x)), x, precision=precision)
        assert equal.success
        assert np.allclose(equal.rates, [.02, .02], rtol=1e-6)


def test_disjoint_exposure_matches_single_rate_analytic_mle_and_penalty_shrinks():
    x = np.r_[np.tile([300., 0.], (20, 1)), np.tile([0., 300.], (20, 1))]
    y = np.r_[np.full(20, 2.), np.full(20, 6.)]
    ordinary = m.fit_poisson_identity(y, np.zeros(40), x, precision=0.)
    regularized = m.fit_poisson_identity(y, np.zeros(40), x, precision=1.)
    assert ordinary.success and regularized.success
    assert np.allclose(ordinary.rates, [2/300, 6/300], rtol=1e-6)
    assert 0 < np.log(regularized.rates[1]/regularized.rates[0]) < np.log(3.)
    assert ordinary.loglik >= regularized.loglik-1e-8
    penalized = lambda f: -f.loglik+.5*np.log(f.rates[1]/f.rates[0])**2
    assert penalized(regularized) <= penalized(ordinary)


def test_probability_bounds_zero_deaths_and_nonidentification():
    x = np.array([[300., 0.], [0., 300.]])
    for y in (np.zeros(2), np.full(2, 10000.)):
        fit = m.fit_poisson_identity(y, np.zeros(2), x, precision=1.)
        assert fit.success and np.isfinite(fit.loglik)
        assert np.all(fit.rates >= m.P_BOUNDS[0]) and np.all(fit.rates <= m.P_BOUNDS[1])
    fit = m.fit_poisson_identity(np.ones(3), np.zeros(3), np.ones((3, 2)))
    assert not fit.success and fit.status.startswith('unidentifiable:')


def test_structural_zero_rows_have_exact_likelihood_and_impossible_counts_rejected():
    x = np.array([[100., 0.], [0., 100.], [0., 0.]])
    y, offset = np.array([1., 2., 0.]), np.zeros(3)
    fit = m.fit_poisson_identity(y, offset, x)
    assert fit.success
    assert fit.loglik == pytest.approx(m.fit_poisson_identity(y[:2], offset[:2], x[:2]).loglik, abs=1e-12)
    y[-1] = 1.
    with pytest.raises(ValueError, match='structurally zero'):
        m.fit_poisson_identity(y, offset, x)


def test_nonconvergence_does_not_silently_become_a_success(monkeypatch):
    monkeypatch.setattr(m, 'minimize', lambda fun, x, *a, **kw:
        SimpleNamespace(x=x, fun=100., success=False, nit=2, message='forced failure'))
    fit = m.fit_poisson_identity(np.ones(2), np.zeros(2), np.eye(2)*100.)
    assert not fit.success and fit.status == 'nonconvergence:forced failure'


def test_optimizer_success_cannot_accept_a_worse_common_rate_objective(monkeypatch):
    monkeypatch.setattr(m, 'minimize', lambda *a, **kw:
        SimpleNamespace(x=np.log([.9, .9]), fun=0., success=True, nit=1, message='claimed success'))
    fit = m.fit_poisson_identity(np.ones(2), np.zeros(2), np.eye(2)*100.)
    assert not fit.success
    assert fit.status == 'nonconvergence:nonfinite_or_worse_than_common_rate'


def test_claimed_success_improving_common_start_still_requires_stationarity(monkeypatch):
    calls = []
    def nonstationary(*args, **kwargs):
        calls.append(kwargs['method'])
        return SimpleNamespace(x=np.log([.025, .045]), fun=0., success=True,
                               nit=1, message='claimed success')
    monkeypatch.setattr(m, 'minimize', nonstationary)
    x, y, offset = 100.*np.eye(2), np.array([2., 6.]), np.zeros(2)
    candidate_value, gradient = m._poisson_lograte_nll_and_grad(np.log([.025, .045]), y, offset, x, 0.)
    common_value = m._poisson_lograte_nll_and_grad(np.log([.04, .04]), y, offset, x, 0.)[0]
    assert candidate_value < common_value and np.allclose(gradient, [.5, -1.5])
    fit = m.fit_poisson_identity(y, offset, x)
    assert not fit.success and fit.status == 'nonconvergence:projected_gradient'
    assert 'L-BFGS-B' in calls and calls[-1] == 'Powell'


def test_stationary_powell_retry_is_accepted_after_nonstationary_successes(monkeypatch):
    def optimizer(*args, **kwargs):
        rates = [.02, .06] if kwargs['method'] == 'Powell' else [.025, .045]
        return SimpleNamespace(x=np.log(rates), fun=0., success=True, nit=1,
                               message='claimed success')
    monkeypatch.setattr(m, 'minimize', optimizer)
    fit = m.fit_poisson_identity(np.array([2., 6.]), np.zeros(2), 100.*np.eye(2))
    assert fit.success and fit.status == 'success:powell_retry'
    assert np.allclose(fit.rates, [.02, .06])


@pytest.mark.parametrize('precision', [0., 1.])
def test_interior_solution_passes_independent_score_check(precision):
    x = np.array([[400., 0.], [200., 120.], [30., 250.], [0., 80.]])
    y, offset = np.array([6., 10., 8., 0.]), np.array([.2, .5, .8, .1])
    fit = m.fit_poisson_identity(y, offset, x, precision=precision)
    _, gradient, residual, tolerance, stationary = m._poisson_candidate_check(np.log(fit.rates), y, offset, x, precision)
    assert fit.success and stationary
    assert np.max(np.abs(gradient)) == residual <= tolerance
    assert np.all(fit.rates > m.P_BOUNDS[0]) and np.all(fit.rates < m.P_BOUNDS[1])


@pytest.mark.parametrize('y,which,sign', [(np.zeros(2), 0, 1), (np.full(2, 10000.), 1, -1)])
def test_bound_solutions_allow_correct_outward_score_only(y, which, sign):
    x, offset = 300.*np.eye(2), np.zeros(2)
    fit = m.fit_poisson_identity(y, offset, x, precision=1.)
    _, gradient, residual, tolerance, stationary = m._poisson_candidate_check(np.log(fit.rates), y, offset, x, 1.)
    assert fit.success and stationary and residual <= tolerance
    assert np.allclose(fit.rates, m.P_BOUNDS[which], rtol=1e-9, atol=0.)
    assert np.all(sign*gradient > 0.)
    wrong_y = np.ones(2) if which == 0 else np.zeros(2)
    assert not m._poisson_candidate_check(np.log(fit.rates), wrong_y, offset, x, 0.)[-1]


def test_no_finite_acceptable_candidate_remains_explicitly_unsuccessful(monkeypatch):
    monkeypatch.setattr(m, 'minimize', lambda *a, **kw:
        SimpleNamespace(x=np.full(2, np.nan), fun=0., success=True, nit=1, message='claimed success'))
    fit = m.fit_poisson_identity(np.array([2., 6.]), np.zeros(2), 100.*np.eye(2))
    assert not fit.success and fit.status == 'nonconvergence:no_finite_candidate'


@pytest.mark.parametrize('kind', ['shape', 'negative', 'nan', 'precision'])
def test_invalid_inputs(kind):
    y, offset, x, precision = np.ones(3), np.zeros(3), np.array([[1., 0.], [0., 1.], [1., 1.]]), 0.
    if kind == 'shape': x = np.ones((3, 1))
    if kind == 'negative': y[0] = -1
    if kind == 'nan': x[0, 0] = np.nan
    if kind == 'precision': precision = -1
    with pytest.raises(ValueError):
        m.fit_poisson_identity(y, offset, x, precision=precision)
