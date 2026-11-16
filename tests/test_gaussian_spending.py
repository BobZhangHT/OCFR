import numpy as np
import pytest
import methods as m


def test_support_is_removed_without_strict_alarm_and_cannot_return():
    state = m.GaussianSpendingBoundary(19, 2, .2)
    first = state.advance(np.arange(19.))
    assert (first.rank, first.boundary, first.eligible_after) == (18, 17., 17)
    assert not state.strict_crossed[17] and state.support_discarded[17]
    x = np.arange(19.)
    x[17] = 100
    second = state.advance(x)
    assert (second.rank, second.eligible_after) == (16, 15)
    assert not state.eligible[17] and state.strict_crossed[17]


def test_skips_repeated_rank_ties_and_invalid_input():
    state = m.GaussianSpendingBoundary(19, 4, .1)
    with pytest.raises(ValueError):
        state.advance(np.zeros(18))
    assert state.look == 0
    first = state.advance(available=False)
    assert not first.resolved and state.eligible.all()
    second = state.advance(np.arange(19.))
    assert second.rank == 19 and second.eligible_after == 18
    third = state.advance(np.arange(19.))
    assert not third.resolved and np.isinf(third.boundary)
    tied = m.GaussianSpendingBoundary(19, 1, .1).advance(np.zeros(19))
    assert tied.boundary_ties == 19 and tied.strict_crossed_total == 0


def test_single_look_matches_existing_critical_value():
    x = np.random.default_rng(11).normal(size=399)
    look = m.GaussianSpendingBoundary(399, 1, .05).advance(x)
    assert look.boundary == m.monte_carlo_critical_value(x, .05)


@pytest.mark.parametrize('seed,stream', [(0, 0), (123, 19), ((1 << 64)-1, (1 << 64)-1)])
def test_portable_bank_words_normals_and_stream(seed, stream):
    rng = m._PCG32(seed, stream)
    actual = m.pcg_uint32_from_state(rng.state, rng.increment, 1025)
    expected = [rng._next_uint32() for _ in range(1025)]
    np.testing.assert_array_equal(actual, expected)
    fast, scalar = m.PythonSampler(seed, stream), m.PythonSampler(seed, stream)
    left = m.gaussian_multiplier_bank(fast, (100, 1000), normal_mode='pcg32-numpy-box-muller-v1')
    right = scalar.standard_normal((100, 1000))
    np.testing.assert_allclose(left, right, atol=1e-13, rtol=0)
    assert fast.stream == scalar.stream == ((stream + 1) & ((1 << 64)-1))
    np.testing.assert_array_equal(fast.negative_binomial(10., .7, size=12), scalar.negative_binomial(10., .7, size=12))


@pytest.mark.parametrize('seed,stream', [(0, 0), (123, 19), ((1 << 64)-1, (1 << 64)-1)])
@pytest.mark.parametrize('count', [100_000, 200_017])
def test_chunked_bank_exact_previous_vectorized_output(monkeypatch, seed, stream, count):
    import math
    reference = m._PCG32(seed, stream)
    words = m.pcg_uint32_from_state(reference.state, reference.increment, 2*count)
    uniforms = (words.astype(np.float64)+.5)/4294967296.
    previous = np.sqrt(-2.*np.log(uniforms[::2])) * np.cos((2.*math.pi)*uniforms[1::2])
    actual_words, requested = m.pcg_uint32_from_state, []
    def bounded_words(state, increment, size):
        requested.append(size)
        return actual_words(state, increment, size)
    monkeypatch.setattr(m, 'pcg_uint32_from_state', bounded_words)
    rng, scalar = m.PythonSampler(seed, stream), m.PythonSampler(seed, stream)
    actual = m.gaussian_multiplier_bank(rng, (count,), normal_mode='pcg32-numpy-box-muller-v1')
    np.testing.assert_array_equal(actual, previous)
    np.testing.assert_allclose(actual, scalar.standard_normal(count), atol=1e-13, rtol=0)
    assert max(requested) <= 2*m.GAUSSIAN_BANK_CHUNK_NORMALS
    assert sum(requested) == 2*count
    assert rng.stream == scalar.stream == ((stream+1) & ((1 << 64)-1))
    np.testing.assert_array_equal(rng.negative_binomial(10., .7, size=12), scalar.negative_binomial(10., .7, size=12))


@pytest.mark.parametrize('steps', [0, 1, 31, 200_000, 400_034])
def test_pcg_affine_jump_matches_scalar_state(steps):
    rng = m._PCG32(2026111404, 19)
    expected = m._pcg32_advance_state(rng.state, rng.increment, steps)
    for _ in range(steps):
        rng._next_uint32()
    assert expected == rng.state


@pytest.mark.parametrize('shape', [(0, 17), (3, 0), ()])
def test_chunked_bank_empty_and_scalar_shapes(shape):
    rng, scalar = m.PythonSampler(123), m.PythonSampler(123)
    actual = m.gaussian_multiplier_bank(rng, shape, normal_mode='pcg32-numpy-box-muller-v1')
    expected = scalar.standard_normal(shape)
    assert actual.shape == shape
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)
    assert rng.stream == scalar.stream == 1


@pytest.mark.parametrize('shape', [(-1, 2), (1.5, 2), (True, 2), None])
def test_chunked_bank_rejects_shape_before_stream_advance(shape):
    rng = m.PythonSampler(123)
    with pytest.raises(ValueError, match='shape'):
        m.gaussian_multiplier_bank(rng, shape, normal_mode='pcg32-numpy-box-muller-v1')
    assert rng.stream == 0


def test_independent_reference_crossing_probability():
    # Independent Gaussian coordinates exercise the adaptive deletion, not just
    # strict-crossing fractions of paths used to choose their own boundary.
    rng = np.random.default_rng(208)
    outcomes = []
    for _ in range(2500):
        bank, fresh = rng.normal(size=(99, 3)), rng.normal(size=3)
        state = m.GaussianSpendingBoundary(99, 3, .1)
        outcomes.append(any(fresh[j] > state.advance(bank[:, j]).boundary for j in range(3)))
    assert abs(np.mean(outcomes) - .1) < .025


def test_online_uses_shared_state_and_explicit_legacy(monkeypatch):
    from types import SimpleNamespace
    def scan(cases, deaths, delay, **kwargs):
        available = len(cases) != 14
        return SimpleNamespace(candidates=np.array([1]) if available else np.array([]),
            statistic=0., influence=np.ones((len(cases), 1))/len(cases), tau_hat=1, direction=1,
            null_fit=SimpleNamespace(success=available, status='success' if available else 'failed', pi=.02, phi=10.))
    monkeypatch.setattr(m, 'score_scan', scan)
    for policy in (m.GAUSSIAN_SPENDING_POLICY, m.LEGACY_GAUSSIAN_POLICY):
        detector = m.OnlineGaussianProcessScore([1.], horizon=21, planned_looks=(7, 14, 21),
            n_multiplier=399, rng=m.PythonSampler(14), backend='python', boundary_policy=policy)
        reference = m.GaussianSpendingBoundary(399, 3, .05)
        for day in range(1, 22):
            result = detector.update(100., 2.)
            if day not in (7, 14, 21):
                continue
            if day == 14:
                reference.advance(available=False)
                assert np.isinf(result.threshold)
                continue
            maxima = m.multiplier_statistics_from_draws(detector._multipliers[:, :day],
                np.ones((day, 1))/day, backend='python')
            expected = reference.advance(maxima).boundary if policy == m.GAUSSIAN_SPENDING_POLICY else m.monte_carlo_critical_value(maxima, .05/3)
            assert result.threshold == expected
        assert detector.boundary_method == policy
