"""Analytic and independent numeric checks of the daily bounded RSS fit."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import methods as m


def test_oracle_float_recovery_offset_zero_rss_and_no_optimizer(monkeypatch):
    monkeypatch.setattr(m, 'minimize', lambda *a, **kw: pytest.fail('iterative optimizer used'))
    x = np.array([[4., 0.], [0., 8.], [4., 8.]])
    truth, offset = np.array([.125, .25]), np.array([.5, .25, 1.])
    fit = m.fit_ls_identity(offset+x@truth, offset, x)
    assert fit.success and fit.rss < 1e-27
    np.testing.assert_allclose(fit.rates, truth, atol=1e-15)
    exact = m.fit_ls_identity(np.array([.25, .5]), np.zeros(2), np.eye(2))
    assert exact.success and exact.rss == 0.
    assert not hasattr(exact, 'phi') and not hasattr(exact, 'loglik')


@pytest.mark.parametrize('target', [[.2, .5], [-.2, .5], [2., .5], [-2., 3.], [0., 0.]])
def test_disjoint_analytic_interior_edges_corners_and_kkt(target):
    x = np.diag([2., 3.])
    # A nonnegative offset permits negative effective targets.
    offset = np.array([10., 10.])
    y = offset+x@target
    fit = m.fit_ls_identity(y, offset, x)
    expected = np.clip(target, *m.P_BOUNDS)
    assert fit.success
    np.testing.assert_allclose(fit.rates, expected, atol=2e-15)
    assert fit.rss == pytest.approx(np.sum((y-offset-x@expected)**2))
    assert fit.kkt_residual <= fit.kkt_tolerance
    gradient = 2*x.T@(offset+x@fit.rates-y)
    for j in range(2):
        if fit.rates[j] == m.P_BOUNDS[0]: assert gradient[j] >= 0.
        elif fit.rates[j] == m.P_BOUNDS[1]: assert gradient[j] <= 0.
        else: assert abs(gradient[j]) < 1e-12


def test_random_correlated_design_independent_bounded_linear_solver():
    rng = np.random.default_rng(23)
    for _ in range(30):
        x = rng.uniform(0., 300., (35, 2))
        offset = rng.uniform(0., 10., 35)
        y = rng.uniform(0., 450., 35)
        fit = m.fit_ls_identity(y, offset, x)
        reference = lsq_linear(x, y-offset, bounds=m.P_BOUNDS, tol=1e-13)
        assert fit.success
        np.testing.assert_allclose(fit.rates, reference.x, atol=1e-9)
        assert fit.rss == pytest.approx(np.sum((y-offset-x@reference.x)**2), rel=1e-12)


def test_kkt_rejects_wrong_bound_sign_and_nonstationary_interior():
    x, y, offset = np.eye(2), np.array([.2, .4]), np.zeros(2)
    for point in ([m.P_BOUNDS[0]]*2, [m.P_BOUNDS[1]]*2, [.3, .4]):
        assert not m._ls_candidate_check(point, y, offset, x)[-1]
    for scale in (1e-6, 1., 1e6):
        fit = m.fit_ls_identity(y*scale, offset, x*scale)
        assert fit.success
        np.testing.assert_allclose(fit.rates, y, atol=1e-14)


def test_zero_rows_allowed_and_rank_deficiency_explicit():
    fit = m.fit_ls_identity(np.array([.2, .4, 3.]), np.zeros(3),
                            np.array([[1., 0.], [0., 1.], [0., 0.]]))
    assert fit.success and fit.rss == pytest.approx(9.)
    fit = m.fit_ls_identity(np.ones(3), np.zeros(3), np.ones((3, 2)))
    assert not fit.success and fit.status.startswith('unidentifiable:')


@pytest.mark.parametrize('kind', ['shape', 'negative_y', 'negative_offset', 'negative_x', 'nan', 'empty'])
def test_invalid_inputs(kind):
    y, offset, x = np.ones(3), np.zeros(3), np.array([[1., 0.], [0., 1.], [1., 1.]])
    if kind == 'shape': x = np.ones((3, 1))
    if kind == 'negative_y': y[0] = -1.
    if kind == 'negative_offset': offset[0] = -1.
    if kind == 'negative_x': x[0, 0] = -1.
    if kind == 'nan': x[0, 0] = np.nan
    if kind == 'empty': y, offset, x = np.empty(0), np.empty(0), np.empty((0, 2))
    with pytest.raises(ValueError): m.fit_ls_identity(y, offset, x)
