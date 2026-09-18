"""Self-contained OCFR methods, reference samplers, and native bindings.

This file is mechanically consolidated from the validated OCFR package modules.
It depends only on NumPy/SciPy and optionally ``csrc/ocfr_kernels.c``.  The
public API defaults to ``backend="c"``. The Python implementation remains
available as an explicit reference backend, while ``backend="auto"`` offers
an optional C-first fallback for systems without a native build.
"""
from __future__ import annotations
import shutil
import subprocess
import sys
import tempfile
# Source section: fast.py
"""Optional C99 acceleration for the current score and exposure kernels."""
import ctypes
import os
from pathlib import Path
import numpy as np
_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)
_UCHAR_PTR = ctypes.POINTER(ctypes.c_ubyte)
_INT64_PTR = ctypes.POINTER(ctypes.c_int64)
_UINT64_PTR = ctypes.POINTER(ctypes.c_uint64)

def _library_path() -> Path | None:
    root = Path(__file__).resolve().parent
    names = ('ocfr_fast.dll', 'libocfr_fast.so', 'libocfr_fast.dylib')
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None

class FastKernels:

    def __init__(self) -> None:
        path = _library_path()
        if path is None:
            raise RuntimeError('compiled OCFR C kernel is not available')
        if os.name == 'nt':
            self.lib = ctypes.CDLL(str(path), winmode=0)
        else:
            self.lib = ctypes.CDLL(str(path))
        self.lib.ocfr_exposure.argtypes = [ctypes.c_int, ctypes.c_int, _DOUBLE_PTR, _DOUBLE_PTR, ctypes.c_int, ctypes.c_int, _DOUBLE_PTR]
        self.lib.ocfr_exposure.restype = ctypes.c_int
        self.lib.ocfr_nb_loglik.argtypes = [ctypes.c_int, _DOUBLE_PTR, _DOUBLE_PTR, ctypes.c_double]
        self.lib.ocfr_nb_loglik.restype = ctypes.c_double
        self.lib.ocfr_score_from_matrix.argtypes = [ctypes.c_int, ctypes.c_int, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR, ctypes.c_double, _DOUBLE_PTR, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.lib.ocfr_score_from_matrix.restype = ctypes.c_double
        self.lib.ocfr_candidate_matrix.argtypes = [ctypes.c_int, ctypes.c_int, _DOUBLE_PTR, _DOUBLE_PTR, ctypes.c_int, ctypes.POINTER(ctypes.c_int), _DOUBLE_PTR]
        self.lib.ocfr_candidate_matrix.restype = ctypes.c_int
        self.lib.ocfr_finite_support_score.argtypes = [ctypes.c_int, ctypes.c_int, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR, ctypes.c_double, ctypes.c_int, _DOUBLE_PTR, _DOUBLE_PTR, _UCHAR_PTR]
        self.lib.ocfr_finite_support_score.restype = ctypes.c_int
        self.lib.ocfr_gp_exposure_components.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR]
        self.lib.ocfr_gp_exposure_components.restype = ctypes.c_int
        self.lib.ocfr_gp_score_fields.argtypes = [ctypes.c_int, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR, ctypes.c_int, ctypes.c_int, ctypes.c_int, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR, _UCHAR_PTR, ctypes.POINTER(ctypes.c_int)]
        self.lib.ocfr_gp_score_fields.restype = ctypes.c_int
        self.lib.ocfr_multiplier_maxima.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR]
        self.lib.ocfr_multiplier_maxima.restype = ctypes.c_int
        self.lib.ocfr_rng_uniform.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, _DOUBLE_PTR]
        self.lib.ocfr_rng_uniform.restype = ctypes.c_int
        self.lib.ocfr_rng_normal.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, _DOUBLE_PTR]
        self.lib.ocfr_rng_normal.restype = ctypes.c_int
        self.lib.ocfr_rng_integers.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, _UINT64_PTR]
        self.lib.ocfr_rng_integers.restype = ctypes.c_int
        self.lib.ocfr_rng_poisson.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, _DOUBLE_PTR, _INT64_PTR]
        self.lib.ocfr_rng_poisson.restype = ctypes.c_int
        self.lib.ocfr_rng_negative_binomial.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, _DOUBLE_PTR, _DOUBLE_PTR, _INT64_PTR]
        self.lib.ocfr_rng_negative_binomial.restype = ctypes.c_int
        self.lib.ocfr_rng_binomial.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, _INT64_PTR, _DOUBLE_PTR, _INT64_PTR]
        self.lib.ocfr_rng_binomial.restype = ctypes.c_int

    @staticmethod
    def _as_double(x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(x, dtype=np.float64)

    def exposure(self, cases: np.ndarray, delay: np.ndarray, start: int, end: int) -> np.ndarray:
        cases = self._as_double(cases)
        delay = self._as_double(delay)
        out = np.zeros(cases.size, dtype=np.float64)
        code = self.lib.ocfr_exposure(cases.size, delay.size, cases.ctypes.data_as(_DOUBLE_PTR), delay.ctypes.data_as(_DOUBLE_PTR), int(start), int(end), out.ctypes.data_as(_DOUBLE_PTR))
        if code != 0:
            raise RuntimeError(f'ocfr_exposure failed with code {code}')
        return out

    def nb_loglik(self, y: np.ndarray, mu: np.ndarray, phi: float) -> float:
        y = self._as_double(y)
        mu = self._as_double(mu)
        return float(self.lib.ocfr_nb_loglik(y.size, y.ctypes.data_as(_DOUBLE_PTR), mu.ctypes.data_as(_DOUBLE_PTR), float(phi)))

    def score_from_matrix(self, bmat: np.ndarray, residual_score: np.ndarray, inv_variance: np.ndarray, x: np.ndarray, ipp: float) -> tuple[float, int, float, int]:
        bmat = self._as_double(bmat)
        residual_score = self._as_double(residual_score)
        inv_variance = self._as_double(inv_variance)
        x = self._as_double(x)
        best_idx = ctypes.c_int(-1)
        signed = ctypes.c_double(0.0)
        eligible = ctypes.c_int(0)
        best = self.lib.ocfr_score_from_matrix(bmat.shape[0], bmat.shape[1], bmat.ctypes.data_as(_DOUBLE_PTR), residual_score.ctypes.data_as(_DOUBLE_PTR), inv_variance.ctypes.data_as(_DOUBLE_PTR), x.ctypes.data_as(_DOUBLE_PTR), float(ipp), ctypes.byref(signed), ctypes.byref(best_idx), ctypes.byref(eligible))
        return (float(best), int(best_idx.value), float(signed.value), int(eligible.value))

    def candidate_matrix(self, cases: np.ndarray, delay: np.ndarray, taus: np.ndarray) -> np.ndarray:
        cases = self._as_double(cases)
        delay = self._as_double(delay)
        taus = np.ascontiguousarray(taus, dtype=np.int32)
        out = np.zeros((taus.size, cases.size), dtype=np.float64)
        code = self.lib.ocfr_candidate_matrix(cases.size, delay.size, cases.ctypes.data_as(_DOUBLE_PTR), delay.ctypes.data_as(_DOUBLE_PTR), taus.size, taus.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), out.ctypes.data_as(_DOUBLE_PTR))
        if code != 0:
            raise RuntimeError(f'ocfr_candidate_matrix failed with code {code}')
        return out

    def finite_support_score(self, cases: np.ndarray, delay: np.ndarray, x: np.ndarray, residual_score: np.ndarray, inv_variance: np.ndarray, ipp: float, tau_start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cases = self._as_double(cases)
        delay = self._as_double(delay)
        x = self._as_double(x)
        residual_score = self._as_double(residual_score)
        inv_variance = self._as_double(inv_variance)
        count = cases.size - int(tau_start)
        if count < 0:
            raise ValueError('tau_start exceeds the observed prefix')
        q_values = np.empty(count, dtype=np.float64)
        u_values = np.empty(count, dtype=np.float64)
        eligibility = np.empty(count, dtype=np.uint8)
        code = self.lib.ocfr_finite_support_score(cases.size, delay.size, cases.ctypes.data_as(_DOUBLE_PTR), delay.ctypes.data_as(_DOUBLE_PTR), x.ctypes.data_as(_DOUBLE_PTR), residual_score.ctypes.data_as(_DOUBLE_PTR), inv_variance.ctypes.data_as(_DOUBLE_PTR), float(ipp), int(tau_start), q_values.ctypes.data_as(_DOUBLE_PTR), u_values.ctypes.data_as(_DOUBLE_PTR), eligibility.ctypes.data_as(_UCHAR_PTR))
        if code != 0:
            raise RuntimeError(f'ocfr_finite_support_score failed with code {code}')
        return (q_values, u_values, eligibility.astype(bool))

    def gp_exposure_components(self, cases: np.ndarray, delay: np.ndarray, *, lag_offset: int=1) -> tuple[np.ndarray, np.ndarray]:
        """C99 exposure builder for the one-change Gaussian score detector."""
        cases = self._as_double(cases)
        delay = self._as_double(delay)
        if cases.ndim != 1 or cases.size < 3:
            raise ValueError('cases must be a vector with at least three days')
        if lag_offset not in (0, 1):
            raise ValueError('lag_offset must be 0 or 1')
        total = np.zeros(cases.size, dtype=np.float64)
        post = np.zeros((cases.size, cases.size - 1), dtype=np.float64)
        code = self.lib.ocfr_gp_exposure_components(cases.size, delay.size, lag_offset, cases.ctypes.data_as(_DOUBLE_PTR), delay.ctypes.data_as(_DOUBLE_PTR), total.ctypes.data_as(_DOUBLE_PTR), post.ctypes.data_as(_DOUBLE_PTR))
        if code != 0:
            raise RuntimeError(f'ocfr_gp_exposure_components failed with code {code}')
        return (total, post)

    def gp_score_fields(self, post: np.ndarray, exposure: np.ndarray, deaths: np.ndarray, mean: np.ndarray, variance: np.ndarray, *, observation_start: int, candidate_start: int, use_lindeberg_gate: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return all current OCFR score fields and the valid-candidate mask."""
        post = self._as_double(post)
        exposure = self._as_double(exposure)
        deaths = self._as_double(deaths)
        mean = self._as_double(mean)
        variance = self._as_double(variance)
        n = exposure.size
        if post.shape != (n, n - 1):
            raise ValueError('post must have shape (n, n - 1)')
        for name, value in (('deaths', deaths), ('mean', mean), ('variance', variance)):
            if value.shape != (n,):
                raise ValueError(f'{name} must have shape (n,)')
        information = np.empty(n - 1, dtype=np.float64)
        max_leverage = np.empty(n - 1, dtype=np.float64)
        z_values = np.empty(n - 1, dtype=np.float64)
        q_values = np.empty(n - 1, dtype=np.float64)
        influence = np.zeros((n, n - 1), dtype=np.float64)
        valid = np.zeros(n - 1, dtype=np.uint8)
        eligible = ctypes.c_int(0)
        code = self.lib.ocfr_gp_score_fields(n, post.ctypes.data_as(_DOUBLE_PTR), exposure.ctypes.data_as(_DOUBLE_PTR), deaths.ctypes.data_as(_DOUBLE_PTR), mean.ctypes.data_as(_DOUBLE_PTR), variance.ctypes.data_as(_DOUBLE_PTR), int(observation_start), int(candidate_start), int(bool(use_lindeberg_gate)), information.ctypes.data_as(_DOUBLE_PTR), max_leverage.ctypes.data_as(_DOUBLE_PTR), z_values.ctypes.data_as(_DOUBLE_PTR), q_values.ctypes.data_as(_DOUBLE_PTR), influence.ctypes.data_as(_DOUBLE_PTR), valid.ctypes.data_as(_UCHAR_PTR), ctypes.byref(eligible))
        if code != 0:
            raise RuntimeError(f'ocfr_gp_score_fields failed with code {code}')
        mask = valid.astype(bool)
        if int(np.sum(mask)) != int(eligible.value):
            raise RuntimeError('C kernel returned an inconsistent eligible count')
        return (information, max_leverage, z_values, q_values, influence, mask)

    def multiplier_maxima(self, multipliers: np.ndarray, influence: np.ndarray) -> np.ndarray:
        """Return row-wise squared Gaussian-process suprema using C99.

        ``multipliers`` is a finite ``(B, n)`` standard-normal draw matrix and
        ``influence`` is a finite ``(n, k)`` influence matrix.  The result is
        the maximum over candidate columns for each multiplier row.  Inputs
        are copied to contiguous ``float64`` storage before crossing the C
        ABI, so the row-major layout is explicit and reproducible.
        """
        try:
            multipliers = np.asarray(multipliers, dtype=np.float64)
            influence = np.asarray(influence, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError('multiplier inputs must be numeric arrays') from exc
        if multipliers.ndim != 2:
            raise ValueError('multipliers must have shape (B, n)')
        if influence.ndim != 2:
            raise ValueError('influence must have shape (n, k)')
        if multipliers.shape[0] < 1 or multipliers.shape[1] < 1:
            raise ValueError('multipliers must have positive dimensions')
        if influence.shape[1] < 1:
            raise ValueError('influence must contain at least one candidate')
        if influence.shape[0] != multipliers.shape[1]:
            raise ValueError('multipliers and influence have incompatible observation dimensions')
        if not np.all(np.isfinite(multipliers)):
            raise ValueError('multipliers must contain only finite values')
        if not np.all(np.isfinite(influence)):
            raise ValueError('influence must contain only finite values')
        multipliers = np.ascontiguousarray(multipliers, dtype=np.float64)
        influence = np.ascontiguousarray(influence, dtype=np.float64)
        out = np.empty(multipliers.shape[0], dtype=np.float64)
        code = self.lib.ocfr_multiplier_maxima(multipliers.shape[0], multipliers.shape[1], influence.shape[1], multipliers.ctypes.data_as(_DOUBLE_PTR), influence.ctypes.data_as(_DOUBLE_PTR), out.ctypes.data_as(_DOUBLE_PTR))
        if code != 0:
            raise RuntimeError(f'ocfr_multiplier_maxima failed with code {code}')
        return out

    @staticmethod
    def _as_int64(x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(x, dtype=np.int64)

    @staticmethod
    def _as_uint64(x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(x, dtype=np.uint64)

    def rng_uniform(self, seed: int, stream: int, n: int) -> np.ndarray:
        """Return OCFR's reproducible open-unit uniforms from C99."""
        if n < 0:
            raise ValueError('n must be non-negative')
        out = np.empty(int(n), dtype=np.float64)
        code = self.lib.ocfr_rng_uniform(ctypes.c_uint64(int(seed) & (1 << 64) - 1), ctypes.c_uint64(int(stream) & (1 << 64) - 1), int(n), out.ctypes.data_as(_DOUBLE_PTR))
        if code != 0:
            raise RuntimeError(f'ocfr_rng_uniform failed with code {code}')
        return out

    def rng_normal(self, seed: int, stream: int, n: int) -> np.ndarray:
        """Return OCFR's reproducible standard normals from C99."""
        if n < 0:
            raise ValueError('n must be non-negative')
        out = np.empty(int(n), dtype=np.float64)
        code = self.lib.ocfr_rng_normal(ctypes.c_uint64(int(seed) & (1 << 64) - 1), ctypes.c_uint64(int(stream) & (1 << 64) - 1), int(n), out.ctypes.data_as(_DOUBLE_PTR))
        if code != 0:
            raise RuntimeError(f'ocfr_rng_normal failed with code {code}')
        return out

    def rng_integers(self, seed: int, stream: int, n: int, low: int, high: int) -> np.ndarray:
        """Return unbiased uint64 integers on the half-open interval."""
        if n < 0:
            raise ValueError('n must be non-negative')
        low_u = int(low) & (1 << 64) - 1
        high_u = int(high) & (1 << 64) - 1
        if high_u <= low_u:
            raise ValueError('high must exceed low')
        out = np.empty(int(n), dtype=np.uint64)
        code = self.lib.ocfr_rng_integers(ctypes.c_uint64(int(seed) & (1 << 64) - 1), ctypes.c_uint64(int(stream) & (1 << 64) - 1), int(n), ctypes.c_uint64(low_u), ctypes.c_uint64(high_u), out.ctypes.data_as(_UINT64_PTR))
        if code != 0:
            raise RuntimeError(f'ocfr_rng_integers failed with code {code}')
        return out

    def rng_poisson(self, seed: int, stream: int, mean: np.ndarray) -> np.ndarray:
        mean = self._as_double(mean)
        out = np.empty(mean.size, dtype=np.int64)
        code = self.lib.ocfr_rng_poisson(ctypes.c_uint64(int(seed) & (1 << 64) - 1), ctypes.c_uint64(int(stream) & (1 << 64) - 1), int(mean.size), mean.ctypes.data_as(_DOUBLE_PTR), out.ctypes.data_as(_INT64_PTR))
        if code != 0:
            raise RuntimeError(f'ocfr_rng_poisson failed with code {code}')
        return out

    def rng_negative_binomial(self, seed: int, stream: int, shape: np.ndarray, probability: np.ndarray) -> np.ndarray:
        shape = self._as_double(shape)
        probability = self._as_double(probability)
        if shape.shape != probability.shape:
            raise ValueError('shape and probability must have equal shapes')
        out = np.empty(shape.size, dtype=np.int64)
        code = self.lib.ocfr_rng_negative_binomial(ctypes.c_uint64(int(seed) & (1 << 64) - 1), ctypes.c_uint64(int(stream) & (1 << 64) - 1), int(shape.size), shape.ctypes.data_as(_DOUBLE_PTR), probability.ctypes.data_as(_DOUBLE_PTR), out.ctypes.data_as(_INT64_PTR))
        if code != 0:
            raise RuntimeError(f'ocfr_rng_negative_binomial failed with code {code}')
        return out

    def rng_binomial(self, seed: int, stream: int, trials: np.ndarray, probability: np.ndarray) -> np.ndarray:
        trials = self._as_int64(trials)
        probability = self._as_double(probability)
        if trials.shape != probability.shape:
            raise ValueError('trials and probability must have equal shapes')
        out = np.empty(trials.size, dtype=np.int64)
        code = self.lib.ocfr_rng_binomial(ctypes.c_uint64(int(seed) & (1 << 64) - 1), ctypes.c_uint64(int(stream) & (1 << 64) - 1), int(trials.size), trials.ctypes.data_as(_INT64_PTR), probability.ctypes.data_as(_DOUBLE_PTR), out.ctypes.data_as(_INT64_PTR))
        if code != 0:
            raise RuntimeError(f'ocfr_rng_binomial failed with code {code}')
        return out

def fast_available() -> bool:
    try:
        FastKernels()
    except (OSError, RuntimeError):
        return False
    return True

# Source section: samplers.py
"""Reproducible Python/C random samplers used by the OCFR experiments.

The project uses a small PCG32-based stream rather than relying on the host C
runtime or on NumPy's implementation-specific bit stream.  ``make_sampler``
returns the C-backed implementation when the optional native library is
available and otherwise returns the numerically identical Python reference.

The two implementations use the same ``(seed, stream)`` convention and the
same distribution algorithms.  Each method call consumes one stream number;
thus two samplers constructed with the same seed and making the same sequence
of method calls produce the same deterministic values on a tested
IEEE-754/libm platform.  This is a reproducible software contract, not a claim
that the bit stream is identical to NumPy or that the finite-precision
algorithms are exact for every possible parameter.
"""
import math
from typing import Any
import numpy as np
_NATIVE_KERNELS: Any | None = None
_MASK64 = (1 << 64) - 1
_MASK32 = (1 << 32) - 1
_PCG_MULT = 6364136223846793005
_UINT32_SCALE = float(1 << 32)
_PI = math.pi

def _as_seed(value: int) -> int:
    try:
        return int(value) & _MASK64
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('seed and stream must be integer-like') from exc

def _shape_from_size(size: Any) -> tuple[int, ...]:
    if size is None:
        return ()
    if isinstance(size, tuple):
        shape = tuple((int(value) for value in size))
    elif isinstance(size, list):
        shape = tuple((int(value) for value in size))
    else:
        shape = (int(size),)
    if any((value < 0 for value in shape)):
        raise ValueError('size entries must be non-negative')
    return shape

def _flat_parameters(first: Any, second: Any | None=None, *, size: Any=None) -> tuple[np.ndarray, np.ndarray | None, tuple[int, ...]]:
    """Broadcast scalar/vector parameters to one flat C-compatible shape."""
    a = np.asarray(first, dtype=np.float64)
    b = None if second is None else np.asarray(second, dtype=np.float64)
    if size is None:
        arrays = (a,) if b is None else (a, b)
        broadcasted = np.broadcast_arrays(*arrays)
        shape = broadcasted[0].shape
    else:
        shape = _shape_from_size(size)
        arrays = (np.broadcast_to(a, shape),)
        if b is not None:
            arrays = arrays + (np.broadcast_to(b, shape),)
        broadcasted = arrays
    first_flat = np.ascontiguousarray(broadcasted[0], dtype=np.float64).ravel()
    second_flat = None if b is None else np.ascontiguousarray(broadcasted[1], dtype=np.float64).ravel()
    return (first_flat, second_flat, shape)

def _restore(values: np.ndarray, shape: tuple[int, ...]) -> Any:
    restored = np.asarray(values).reshape(shape)
    if shape == ():
        return restored.reshape(-1)[0].item()
    return restored

class _PCG32:
    """Pure-Python mirror of the C99 PCG32 state transition."""

    def __init__(self, seed: int, stream: int) -> None:
        self.state = 0
        self.increment = (_as_seed(stream) << 1 | 1) & _MASK64
        self._next_uint32()
        self.state = self.state + _as_seed(seed) & _MASK64
        self._next_uint32()

    def _next_uint32(self) -> int:
        oldstate = self.state
        xorshifted = (oldstate >> 18 ^ oldstate) >> 27 & _MASK32
        rotation = oldstate >> 59
        self.state = oldstate * _PCG_MULT + self.increment & _MASK64
        return (xorshifted >> rotation | xorshifted << (-rotation & 31) & _MASK32) & _MASK32

    def uniform_open(self) -> float:
        return (self._next_uint32() + 0.5) / _UINT32_SCALE

    def uint64(self) -> int:
        return self._next_uint32() << 32 | self._next_uint32()

    def normal_standard(self) -> float:
        u1 = self.uniform_open()
        u2 = self.uniform_open()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * _PI * u2)

    def poisson_one(self, mean: float) -> int:
        if not math.isfinite(mean) or mean < 0.0 or mean > 1000000000000.0:
            raise ValueError('Poisson means must be finite in [0, 1e12]')
        if mean == 0.0:
            return 0
        if mean < 30.0:
            limit = math.exp(-mean)
            product = 1.0
            count = 0
            while True:
                count += 1
                product *= self.uniform_open()
                if product <= limit:
                    return count - 1
        sqrt_mean = math.sqrt(mean)
        b = 0.931 + 2.53 * sqrt_mean
        a = -0.059 + 0.02483 * b
        inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
        vr = 0.9277 - 3.6224 / (b - 2.0)
        log_mean = math.log(mean)
        while True:
            u = self.uniform_open() - 0.5
            v = self.uniform_open()
            us = 0.5 - abs(u)
            candidate = math.floor((2.0 * a / us + b) * u + mean + 0.43)
            if candidate < 0:
                continue
            if us >= 0.07 and v <= vr:
                return int(candidate)
            if us < 0.013 and v > us:
                continue
            if math.log(v * inv_alpha / (a / (us * us) + b)) <= -mean + candidate * log_mean - math.lgamma(candidate + 1.0):
                return int(candidate)

    def gamma_one(self, shape: float, scale: float) -> float:
        if not math.isfinite(shape) or not math.isfinite(scale) or shape <= 0.0 or (scale <= 0.0):
            raise ValueError('gamma shape and scale must be finite and positive')
        if shape < 1.0:
            boosted = self.gamma_one(shape + 1.0, 1.0)
            return scale * boosted * self.uniform_open() ** (1.0 / shape)
        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = self.normal_standard()
            one_plus = 1.0 + c * x
            if one_plus <= 0.0:
                continue
            v = one_plus * one_plus * one_plus
            u = self.uniform_open()
            x4 = x * x * x * x
            if u < 1.0 - 0.0331 * x4 or math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
                return scale * d * v

    def negative_binomial_one(self, shape: float, probability: float) -> int:
        if math.isnan(shape) or shape <= 0.0 or math.isnan(probability) or (probability < 0.0) or (probability > 1.0):
            raise ValueError('invalid negative-binomial parameters')
        if probability == 1.0:
            return 0
        if probability == 0.0:
            raise ValueError('negative-binomial probability must be positive')
        if math.isinf(shape):
            return self.poisson_one(shape * (1.0 - probability) / probability)
        if shape >= 10000000.0:
            return self.poisson_one(shape * (1.0 - probability) / probability)
        mean_gamma = shape * (1.0 - probability) / probability
        intensity = self.gamma_one(shape, mean_gamma / shape)
        return self.poisson_one(intensity)

class _SamplerBase:
    """Shared shape handling and stream accounting for both backends."""
    backend: str

    def __init__(self, seed: int, stream: int=0) -> None:
        self.seed = _as_seed(seed)
        self.stream = _as_seed(stream)

    def _next_stream(self) -> int:
        stream = self.stream
        self.stream = self.stream + 1 & _MASK64
        return stream

    def standard_normal(self, size: Any=None) -> Any:
        shape = _shape_from_size(size)
        values = self._standard_normal_flat(int(np.prod(shape, dtype=np.int64)) if shape else 1)
        return _restore(values, shape)

    def normal(self, loc: Any=0.0, scale: Any=1.0, size: Any=None) -> Any:
        shape = _shape_from_size(size)
        loc_flat, scale_flat, out_shape = _flat_parameters(loc, scale, size=size)
        if size is None:
            shape = out_shape
        if not np.all(np.isfinite(loc_flat)) or not np.all(np.isfinite(scale_flat)):
            raise ValueError('normal parameters must be finite')
        if np.any(scale_flat < 0.0):
            raise ValueError('normal scale must be non-negative')
        values = self._standard_normal_flat(loc_flat.size)
        values = loc_flat + scale_flat * values
        return _restore(values, shape)

    def uniform(self, low: Any=0.0, high: Any=1.0, size: Any=None) -> Any:
        low_flat, high_flat, shape = _flat_parameters(low, high, size=size)
        assert high_flat is not None
        if not np.all(np.isfinite(low_flat)) or not np.all(np.isfinite(high_flat)):
            raise ValueError('uniform bounds must be finite')
        if np.any(high_flat < low_flat):
            raise ValueError('uniform high must be at least low')
        values = self._uniform_flat(low_flat.size)
        values = low_flat + (high_flat - low_flat) * values
        return _restore(values, shape)

    def random(self, size: Any=None) -> Any:
        return self.uniform(0.0, 1.0, size=size)
    random_sample = random

    def poisson(self, lam: Any=1.0, size: Any=None) -> Any:
        means, _, shape = _flat_parameters(lam, size=size)
        if not np.all(np.isfinite(means)) or np.any(means < 0.0):
            raise ValueError('Poisson means must be finite and non-negative')
        values = self._poisson_flat(means)
        return _restore(values, shape)

    def negative_binomial(self, n: Any, p: Any, size: Any=None) -> Any:
        shape_flat, probability_flat, shape = _flat_parameters(n, p, size=size)
        assert probability_flat is not None
        if not np.all(np.isfinite(shape_flat)) or np.any(shape_flat <= 0.0) or np.any(np.isnan(probability_flat)) or np.any(probability_flat < 0.0) or np.any(probability_flat > 1.0):
            raise ValueError('invalid negative-binomial parameters')
        values = self._negative_binomial_flat(shape_flat, probability_flat)
        return _restore(values, shape)

    def binomial(self, n: Any, p: Any, size: Any=None) -> Any:
        trials_raw = np.asarray(n)
        if size is None:
            probability_raw = np.asarray(p, dtype=np.float64)
            trials, probability = np.broadcast_arrays(trials_raw, probability_raw)
            shape = trials.shape
        else:
            shape = _shape_from_size(size)
            trials = np.broadcast_to(trials_raw, shape)
            probability = np.broadcast_to(np.asarray(p, dtype=np.float64), shape)
        if np.any(np.asarray(trials) < 0) or np.any(np.asarray(trials) != np.asarray(trials, dtype=np.int64)):
            raise ValueError('binomial trial counts must be non-negative integers')
        trials_flat = np.ascontiguousarray(trials, dtype=np.int64).ravel()
        probability_flat = np.ascontiguousarray(probability, dtype=np.float64).ravel()
        if not np.all(np.isfinite(probability_flat)) or np.any(probability_flat < 0.0) or np.any(probability_flat > 1.0):
            raise ValueError('binomial probabilities must lie in [0, 1]')
        values = self._binomial_flat(trials_flat, probability_flat)
        return _restore(values, shape)

    def integers(self, low: int, high: int | None=None, size: Any=None, dtype: Any=np.int64, endpoint: bool=False) -> Any:
        if endpoint:
            if high is None:
                high = int(low) + 1
                low = 0
            else:
                high = int(high) + 1
        elif high is None:
            high = int(low)
            low = 0
        low = int(low)
        high = int(high)
        if low < 0 or high <= low or high > (1 << 64) - 1:
            raise ValueError('integers currently requires 0 <= low < high < 2**64')
        shape = _shape_from_size(size)
        n = int(np.prod(shape, dtype=np.int64)) if shape else 1
        values = self._integers_flat(n, low, high)
        restored = _restore(values, shape)
        if shape == ():
            return np.asarray(restored, dtype=dtype).item()
        return np.asarray(restored, dtype=dtype)

    def _uniform_flat(self, n: int) -> np.ndarray:
        raise NotImplementedError

    def _standard_normal_flat(self, n: int) -> np.ndarray:
        raise NotImplementedError

    def _integers_flat(self, n: int, low: int, high: int) -> np.ndarray:
        raise NotImplementedError

    def _poisson_flat(self, means: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _negative_binomial_flat(self, shape: np.ndarray, probability: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _binomial_flat(self, trials: np.ndarray, probability: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class PythonSampler(_SamplerBase):
    """Transparent PCG32 reference implementation used for fallback/tests."""
    backend = 'python'

    def _generator(self) -> _PCG32:
        return _PCG32(self.seed, self._next_stream())

    def _uniform_flat(self, n: int) -> np.ndarray:
        rng = self._generator()
        return np.asarray([rng.uniform_open() for _ in range(n)], dtype=np.float64)

    def _standard_normal_flat(self, n: int) -> np.ndarray:
        rng = self._generator()
        return np.asarray([rng.normal_standard() for _ in range(n)], dtype=np.float64)

    def _integers_flat(self, n: int, low: int, high: int) -> np.ndarray:
        rng = self._generator()
        range_width = high - low
        limit = _MASK64 - _MASK64 % range_width
        values: list[int] = []
        for _ in range(n):
            draw = rng.uint64()
            while draw >= limit:
                draw = rng.uint64()
            values.append(low + draw % range_width)
        return np.asarray(values, dtype=np.uint64)

    def _poisson_flat(self, means: np.ndarray) -> np.ndarray:
        rng = self._generator()
        return np.asarray([rng.poisson_one(float(mean)) for mean in means], dtype=np.int64)

    def _negative_binomial_flat(self, shape: np.ndarray, probability: np.ndarray) -> np.ndarray:
        rng = self._generator()
        return np.asarray([rng.negative_binomial_one(float(a), float(b)) for a, b in zip(shape, probability)], dtype=np.int64)

    def _binomial_flat(self, trials: np.ndarray, probability: np.ndarray) -> np.ndarray:
        rng = self._generator()
        values: list[int] = []
        for count, p in zip(trials, probability):
            count_int = int(count)
            if p == 0.0:
                values.append(0)
            elif p == 1.0:
                values.append(count_int)
            else:
                values.append(sum((rng.uniform_open() < float(p) for _ in range(count_int))))
        return np.asarray(values, dtype=np.int64)

class CSampler(_SamplerBase):
    """PCG32 sampler that dispatches distribution draws to the C99 library."""
    backend = 'c'

    def __init__(self, seed: int, stream: int=0) -> None:
        super().__init__(seed, stream)
        global _NATIVE_KERNELS
        if _NATIVE_KERNELS is None:
            _NATIVE_KERNELS = FastKernels()
        self._kernels = _NATIVE_KERNELS

    def _call_stream(self) -> int:
        return self._next_stream()

    def _uniform_flat(self, n: int) -> np.ndarray:
        return self._kernels.rng_uniform(self.seed, self._call_stream(), n)

    def _standard_normal_flat(self, n: int) -> np.ndarray:
        return self._kernels.rng_normal(self.seed, self._call_stream(), n)

    def _integers_flat(self, n: int, low: int, high: int) -> np.ndarray:
        return self._kernels.rng_integers(self.seed, self._call_stream(), n, low, high)

    def _poisson_flat(self, means: np.ndarray) -> np.ndarray:
        return self._kernels.rng_poisson(self.seed, self._call_stream(), means)

    def _negative_binomial_flat(self, shape: np.ndarray, probability: np.ndarray) -> np.ndarray:
        return self._kernels.rng_negative_binomial(self.seed, self._call_stream(), shape, probability)

    def _binomial_flat(self, trials: np.ndarray, probability: np.ndarray) -> np.ndarray:
        return self._kernels.rng_binomial(self.seed, self._call_stream(), trials, probability)

def make_sampler(seed: int, stream: int=0, *, backend: str='c') -> PythonSampler | CSampler:
    """Construct the project sampler, preferring the compiled C99 backend.

    ``backend='c'`` is the strict default used by formal experiment drivers.
    ``backend='python'`` forces the reference implementation, and
    ``backend='auto'`` permits a C-first fallback for portability.
    """
    if backend not in {'auto', 'python', 'c'}:
        raise ValueError("backend must be 'auto', 'python', or 'c'")
    if backend == 'python':
        return PythonSampler(seed, stream)
    try:
        return CSampler(seed, stream)
    except (ImportError, OSError, RuntimeError, AttributeError):
        if backend == 'c':
            raise
        return PythonSampler(seed, stream)

def sampler_backend_available() -> bool:
    """Return whether the optional C sampler library can be loaded."""
    try:
        CSampler(0)
    except (ImportError, OSError, RuntimeError, AttributeError):
        return False
    return True
__all__ = ['CSampler', 'PythonSampler', 'make_sampler', 'sampler_backend_available']

# Source section: model.py
"""Delay-convolution mean model and death simulation used by the paper."""
from typing import Sequence
import numpy as np

def validate_delay_pmf(delay_pmf: Sequence[float]) -> np.ndarray:
    """Validate and normalize a prespecified positive-delay probability mass."""
    f = np.asarray(delay_pmf, dtype=np.float64)
    if f.ndim != 1 or f.size == 0:
        raise ValueError('delay_pmf must be a non-empty one-dimensional sequence')
    if not np.all(np.isfinite(f)) or np.any(f < 0.0):
        raise ValueError('delay_pmf must contain finite non-negative values')
    total = float(f.sum())
    if not np.isclose(total, 1.0, rtol=1e-08, atol=1e-12):
        raise ValueError(f'delay_pmf must sum to one; got {total:.12g}')
    return np.ascontiguousarray(f / total)

def convolved_exposure(cases: Sequence[float], delay_pmf: Sequence[float], start: int=0, end: int | None=None) -> np.ndarray:
    """Expected death exposure from cohorts in ``[start, end)``.

    ``delay_pmf[0]`` corresponds to a one-day confirmation-to-death delay, so
    cohort ``u`` first contributes to death day ``u + 1``.
    """
    c = np.asarray(cases, dtype=np.float64)
    f = np.asarray(delay_pmf, dtype=np.float64)
    if c.ndim != 1 or f.ndim != 1:
        raise ValueError('cases and delay_pmf must be one-dimensional')
    n = c.size
    stop = n if end is None else int(end)
    begin = int(start)
    if not 0 <= begin <= stop <= n:
        raise ValueError('cohort interval must satisfy 0 <= start <= end <= len(cases)')
    selected = np.zeros(n, dtype=np.float64)
    selected[begin:stop] = c[begin:stop]
    out = np.zeros(n, dtype=np.float64)
    if n > 1 and stop > begin:
        raw = np.convolve(selected, f, mode='full')
        out[1:] = raw[:n - 1]
    return out

def means_from_cfr_path(cases: Sequence[float], delay_pmf: Sequence[float], cfr: Sequence[float]) -> np.ndarray:
    """Map cohort-specific risks to expected report-day deaths."""
    c = np.asarray(cases, dtype=np.float64)
    p = np.asarray(cfr, dtype=np.float64)
    if c.shape != p.shape:
        raise ValueError('cases and cfr path must have the same shape')
    return convolved_exposure(c * p, delay_pmf)

def simulate_nb_deaths(cases: Sequence[float], delay_pmf: Sequence[float], cfr: Sequence[float], phi: float, rng: np.random.Generator) -> np.ndarray:
    """Generate NB2 or Poisson deaths from a cohort-risk path.

    ``rng`` may be a NumPy ``Generator`` or an object returned by
    :func:`make_sampler`; both provide the small distribution
    interface used here.
    """
    mu = means_from_cfr_path(cases, delay_pmf, cfr)
    if np.isinf(phi):
        return rng.poisson(mu).astype(np.int64)
    if not np.isfinite(phi) or phi <= 0:
        raise ValueError('phi must be positive or infinity')
    prob = phi / (phi + mu)
    return rng.negative_binomial(phi, prob).astype(np.int64)

# Source section: likelihood.py
"""Negative-binomial likelihood fits used by OCFR and its benchmarks."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln, logit
from scipy.stats import norm
MU_FLOOR = 1e-12
P_BOUNDS = (1e-10, 1.0 - 1e-10)
LOG_PHI_BOUNDS = (np.log(0.001), np.log(100000000.0))
LL_EQUIVALENCE_PER_OBSERVATION = 5e-08

@dataclass(frozen=True)
class NBFit:
    rates: np.ndarray
    phi: float
    loglik: float
    success: bool
    status: str
    nit: int

@dataclass(frozen=True)
class FixedTauWaldIntervals:
    """First-order fixed-change-day intervals for a two-rate NB2 fit.

    The covariance matrix is based on the expected information for the two
    identity-scale CFR coefficients, ``X.T @ diag(1 / V) @ X``.  CFR
    intervals are formed on the logit scale and then transformed back to the
    probability scale.  The risk-ratio interval is formed by applying the
    delta method to ``log(post/pre)`` and is then exponentiated.  These are
    descriptive intervals conditional on the supplied, prespecified ``tau``;
    they do not adjust for selecting ``tau`` from the data.
    """
    rates: np.ndarray
    cfr_logit: np.ndarray
    cfr_logit_lower: np.ndarray
    cfr_logit_upper: np.ndarray
    cfr_lower: np.ndarray
    cfr_upper: np.ndarray
    risk_ratio: float
    log_risk_ratio: float
    log_risk_ratio_lower: float
    log_risk_ratio_upper: float
    risk_ratio_lower: float
    risk_ratio_upper: float
    information: np.ndarray
    covariance: np.ndarray
    cfr_logit_standard_errors: np.ndarray
    log_risk_ratio_standard_error: float

def nb_loglik(y: np.ndarray, mu: np.ndarray, phi: float) -> float:
    y = np.asarray(y, dtype=np.float64)
    mu = np.maximum(np.asarray(mu, dtype=np.float64), MU_FLOOR)
    if np.isinf(phi) or phi >= 10000000.0:
        return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1.0)))
    value = gammaln(y + phi) - gammaln(phi) - gammaln(y + 1.0) + phi * np.log(phi) + y * np.log(mu) - (y + phi) * np.log(phi + mu)
    return float(np.sum(value))

def _nll_and_grad(theta: np.ndarray, y: np.ndarray, offset: np.ndarray, design: np.ndarray) -> tuple[float, np.ndarray]:
    rates = theta[:-1]
    phi = float(np.exp(theta[-1]))
    mu = np.maximum(offset + design @ rates, MU_FLOOR)
    ll = nb_loglik(y, mu, phi)
    poisson_limit = phi >= 10000000.0
    dldmu = y / mu - 1.0 if poisson_limit else y / mu - (y + phi) / (mu + phi)
    grad_rates = -(design.T @ dldmu)
    if poisson_limit:
        grad_logphi = 0.0
    else:
        dldphi = digamma(y + phi) - digamma(phi) + np.log(phi) + 1.0 - np.log(mu + phi) - (y + phi) / (mu + phi)
        grad_logphi = -phi * float(np.sum(dldphi))
    grad = np.concatenate([np.asarray(grad_rates), [grad_logphi]])
    return (-ll, grad)

def _nll_logit_and_grad(theta: np.ndarray, y: np.ndarray, offset: np.ndarray, design: np.ndarray) -> tuple[float, np.ndarray]:
    """Joint NB2 objective with CFR coefficients represented as logits."""
    rates = expit(theta[:-1])
    phi = float(np.exp(theta[-1]))
    mu = np.maximum(offset + design @ rates, MU_FLOOR)
    ll = nb_loglik(y, mu, phi)
    poisson_limit = phi >= 10000000.0
    dldmu = y / mu - 1.0 if poisson_limit else y / mu - (y + phi) / (mu + phi)
    grad_rates = -(design.T @ dldmu) * rates * (1.0 - rates)
    if poisson_limit:
        grad_logphi = 0.0
    else:
        dldphi = digamma(y + phi) - digamma(phi) + np.log(phi) + 1.0 - np.log(mu + phi) - (y + phi) / (mu + phi)
        grad_logphi = -phi * float(np.sum(dldphi))
    return (-ll, np.concatenate([np.asarray(grad_rates), [grad_logphi]]))

def _fixed_phi_nll_and_grad(eta: np.ndarray, y: np.ndarray, offset: np.ndarray, design: np.ndarray, phi: float) -> tuple[float, np.ndarray]:
    """NB2 objective for CFR logits with dispersion fixed at the null fit."""
    rates = expit(eta)
    mu = np.maximum(offset + design @ rates, MU_FLOOR)
    ll = nb_loglik(y, mu, phi)
    dldmu = y / mu - 1.0 if phi >= 10000000.0 else y / mu - (y + phi) / (mu + phi)
    grad_eta = -(design.T @ dldmu) * rates * (1.0 - rates)
    return (-ll, np.asarray(grad_eta, dtype=np.float64))

def _initial_phi(y: np.ndarray, mu: np.ndarray) -> float:
    resid2 = np.square(y - mu)
    denom = float(np.sum(np.maximum(resid2 - mu, 0.0)))
    numer = float(np.sum(np.square(mu)))
    if denom <= MU_FLOOR or numer <= MU_FLOOR:
        return 1000000.0
    return float(np.clip(numer / denom, 0.1, 1000000.0))

def fit_nb_identity(y: np.ndarray, offset: np.ndarray, design: np.ndarray, initial_rates: np.ndarray | None=None, initial_phi: float | None=None, maxiter: int=150) -> NBFit:
    """Profile an NB2 model with mean ``offset + design @ rates``."""
    y = np.asarray(y, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design[:, None]
    if y.ndim != 1 or offset.shape != y.shape or design.shape[0] != y.size:
        raise ValueError('incompatible y, offset, and design shapes')
    k = design.shape[1]
    if initial_rates is None:
        target = np.maximum(y - offset, 0.0)
        denom = np.maximum(design.sum(axis=0), MU_FLOOR)
        common = float(target.sum() / max(float(design.sum()), MU_FLOOR))
        rates0 = np.full(k, common, dtype=np.float64)
        active = denom > MU_FLOOR
        rates0[active] = np.clip(rates0[active], *P_BOUNDS)
    else:
        rates0 = np.asarray(initial_rates, dtype=np.float64).copy()
        if rates0.shape != (k,):
            raise ValueError('initial_rates has the wrong shape')
    rates0 = np.clip(rates0, *P_BOUNDS)
    mu0 = np.maximum(offset + design @ rates0, MU_FLOOR)
    moment_phi = _initial_phi(y, mu0)
    if initial_phi is None:
        phi0 = moment_phi
    else:
        previous_phi = float(initial_phi)
        phi0 = moment_phi if moment_phi < 100000.0 else previous_phi
    if not np.isfinite(phi0):
        phi0 = 100000000.0
    eta_bound = abs(float(logit(P_BOUNDS[0])))
    theta0 = np.concatenate([logit(rates0), [np.log(np.clip(phi0, 0.001, 100000000.0))]])
    bounds = [(-eta_bound, eta_bound)] * k + [LOG_PHI_BOUNDS]
    result = minimize(_nll_logit_and_grad, theta0, args=(y, offset, design), method='L-BFGS-B', jac=True, bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-11, 'gtol': 1e-07, 'maxls': 40})
    used_fallback = False
    if not result.success:
        fallback = minimize(lambda theta: _nll_logit_and_grad(theta, y, offset, design)[0], result.x, method='Powell', bounds=bounds, options={'maxiter': 400, 'xtol': 1e-08, 'ftol': 1e-11})
        if fallback.success and fallback.fun <= result.fun + 1e-09:
            result = fallback
            used_fallback = True
    if not result.success:
        candidate_rates = np.clip(expit(result.x[:-1]), *P_BOUNDS)
        candidate_phi = float(np.exp(result.x[-1]))
        candidate_mu = np.maximum(offset + design @ candidate_rates, MU_FLOOR)
        candidate_loglik = nb_loglik(y, candidate_mu, candidate_phi)
        poisson = fit_nb_rates_fixed_phi(y, offset, design, 100000000.0, candidate_rates)
        tolerance = LL_EQUIVALENCE_PER_OBSERVATION * max(1, y.size)
        if poisson.success and poisson.loglik >= candidate_loglik - tolerance:
            return NBFit(rates=poisson.rates, phi=poisson.phi, loglik=poisson.loglik, success=True, status='success:poisson_equivalent', nit=int(getattr(result, 'nit', -1)) + poisson.nit)
    rates = np.clip(expit(result.x[:-1]), *P_BOUNDS)
    phi = float(np.exp(result.x[-1]))
    mu = np.maximum(offset + design @ rates, MU_FLOOR)
    status = 'success:powell_fallback' if result.success and used_fallback else 'success' if result.success else f'nonconvergence:{result.message}'
    return NBFit(rates=rates, phi=phi, loglik=nb_loglik(y, mu, phi), success=bool(result.success), status=status, nit=int(getattr(result, 'nit', -1)))

def fit_nb_rates_fixed_phi(y: np.ndarray, offset: np.ndarray, design: np.ndarray, phi: float, initial_rates: np.ndarray, maxiter: int=100) -> NBFit:
    """Fit only CFR coefficients while holding NB dispersion fixed.

    The logit parameterization avoids the boundary line-search failures seen
    when low-count candidate models jointly profile CFRs and dispersion.  This
    fit defines the score-peak plug-in likelihood statistic; it is not a
    numerical approximation to the full three-parameter alternative.
    """
    y = np.asarray(y, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design[:, None]
    initial_rates = np.asarray(initial_rates, dtype=np.float64)
    if y.ndim != 1 or offset.shape != y.shape or design.shape[0] != y.size or (initial_rates.shape != (design.shape[1],)):
        raise ValueError('incompatible fixed-dispersion fit inputs')
    if not np.isfinite(phi) or phi <= 0.0:
        raise ValueError('phi must be finite and positive')
    eta0 = logit(np.clip(initial_rates, *P_BOUNDS))
    eta_bound = abs(float(logit(P_BOUNDS[0])))
    result = minimize(_fixed_phi_nll_and_grad, eta0, args=(y, offset, design, float(phi)), method='L-BFGS-B', jac=True, bounds=[(-eta_bound, eta_bound)] * design.shape[1], options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-07, 'maxls': 50})
    used_fallback = False
    if not result.success:
        fallback = minimize(lambda eta: _fixed_phi_nll_and_grad(eta, y, offset, design, float(phi))[0], result.x, method='Powell', bounds=[(-eta_bound, eta_bound)] * design.shape[1], options={'maxiter': 300, 'xtol': 1e-08, 'ftol': 1e-10})
        if fallback.success and fallback.fun <= result.fun + 1e-09:
            result = fallback
            used_fallback = True
    if not result.success:
        simplex = minimize(lambda eta: _fixed_phi_nll_and_grad(eta, y, offset, design, float(phi))[0], result.x, method='Nelder-Mead', bounds=[(-eta_bound, eta_bound)] * design.shape[1], options={'maxiter': 1000, 'xatol': 1e-09, 'fatol': 1e-11})
        if simplex.success and simplex.fun <= result.fun + 1e-09:
            result = simplex
            used_fallback = True
    rates = np.clip(expit(result.x), *P_BOUNDS)
    mu = np.maximum(offset + design @ rates, MU_FLOOR)
    status = 'success:deterministic_fallback' if result.success and used_fallback else 'success' if result.success else f'nonconvergence:{result.message}'
    return NBFit(rates=np.asarray(rates, dtype=np.float64), phi=float(phi), loglik=nb_loglik(y, mu, float(phi)), success=bool(result.success), status=status, nit=int(getattr(result, 'nit', -1)))

def nb_variance(mu: np.ndarray, phi: float) -> np.ndarray:
    mu = np.maximum(np.asarray(mu, dtype=np.float64), MU_FLOOR)
    if np.isinf(phi) or phi >= 10000000.0:
        return mu
    return mu + np.square(mu) / phi

def fixed_tau_wald_intervals(fit: NBFit, offset: np.ndarray, design: np.ndarray, *, confidence: float=0.95) -> FixedTauWaldIntervals:
    """Compute first-order Wald intervals conditional on a fixed change day.

    Parameters
    ----------
    fit:
        A successful :class:`NBFit` from the NB2 identity-mean model.  The
        fitted dispersion is used when constructing the expected variance.
    offset, design:
        The same model components used for the fit, with mean
        ``offset + design @ fit.rates``.  ``design`` must contain exactly two
        columns, in pre-change and post-change CFR order.
    confidence:
        Nominal confidence level.  The default is 0.95.

    Returns
    -------
    FixedTauWaldIntervals
        Intervals for the two CFRs are calculated on the logit scale and
        returned on both the logit and probability scales.  The log-risk-ratio
        interval uses the delta method and is also returned after exponentiating
        its endpoints.

    Notes
    -----
    These are fixed-``tau`` descriptive intervals.  They condition on the
    supplied change day and do not account for the data-driven maximization
    over candidate change days used by the online detector.

    Raises
    ------
    ValueError
        If the fit is unsuccessful or model inputs are malformed.
    numpy.linalg.LinAlgError
        If the expected information is singular or not positive definite.
    """
    if not isinstance(fit, NBFit):
        raise TypeError('fit must be an NBFit instance')
    if not fit.success:
        raise ValueError('fixed-tau confidence intervals require a successfully fitted NB2 model')
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError('confidence must lie strictly between zero and one')
    offset = np.asarray(offset, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    rates = np.asarray(fit.rates, dtype=np.float64)
    if offset.ndim != 1:
        raise ValueError('offset must be a one-dimensional array')
    if design.ndim != 2:
        raise ValueError('design must be a two-dimensional array')
    if design.shape[0] != offset.size:
        raise ValueError('offset and design have incompatible shapes')
    if design.shape[1] != 2:
        raise ValueError('design must have exactly two CFR columns')
    if rates.ndim != 1 or rates.shape != (design.shape[1],):
        raise ValueError('fit.rates must have one entry for each CFR column')
    if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(design)):
        raise ValueError('offset and design must contain only finite values')
    if not np.all(np.isfinite(rates)) or np.any((rates <= 0.0) | (rates >= 1.0)):
        raise ValueError('fit.rates must be finite and strictly between zero and one')
    phi = float(fit.phi)
    if not np.isfinite(phi) and (not np.isinf(phi)) or phi <= 0.0:
        raise ValueError('fit.phi must be positive (or infinite at the Poisson limit)')
    mean = offset + design @ rates
    if not np.all(np.isfinite(mean)) or np.any(mean <= 0.0):
        raise ValueError('the fitted identity-mean model must have positive means')
    variance = nb_variance(mean, phi)
    if variance.shape != mean.shape or not np.all(np.isfinite(variance)):
        raise ValueError('NB2 expected variances must be finite')
    if np.any(variance <= 0.0):
        raise ValueError('NB2 expected variances must be strictly positive')
    information = design.T @ (design / variance[:, None])
    information = 0.5 * (information + information.T)
    if information.shape != (2, 2) or not np.all(np.isfinite(information)):
        raise ValueError('expected information must be a finite 2 by 2 matrix')
    eigenvalues = np.linalg.eigvalsh(information)
    rank_tolerance = np.finfo(np.float64).eps * max(information.shape) * max(1.0, float(eigenvalues[-1]))
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[-1] <= 0.0 or eigenvalues[0] <= rank_tolerance:
        raise np.linalg.LinAlgError('expected information is singular or not positive definite; fixed-tau CFR intervals are unavailable')
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError('expected information is singular; fixed-tau CFR intervals are unavailable') from exc
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        raise np.linalg.LinAlgError('inverse expected information is not finite; fixed-tau CFR intervals are unavailable')
    covariance_diagonal = np.diag(covariance)
    if np.any(covariance_diagonal < 0.0) or not np.all(np.isfinite(covariance_diagonal)):
        raise np.linalg.LinAlgError('inverse expected information has invalid variances')
    quantile = float(norm.ppf(0.5 + float(confidence) / 2.0))
    cfr_logit = logit(rates)
    logit_scale = rates * (1.0 - rates)
    cfr_logit_standard_errors = np.sqrt(covariance_diagonal) / logit_scale
    cfr_logit_lower = cfr_logit - quantile * cfr_logit_standard_errors
    cfr_logit_upper = cfr_logit + quantile * cfr_logit_standard_errors
    cfr_lower = expit(cfr_logit_lower)
    cfr_upper = expit(cfr_logit_upper)
    log_risk_ratio = float(np.log(rates[1]) - np.log(rates[0]))
    risk_ratio = float(np.exp(log_risk_ratio))
    log_risk_ratio_gradient = np.array([-1.0 / rates[0], 1.0 / rates[1]])
    log_risk_ratio_variance = float(log_risk_ratio_gradient @ covariance @ log_risk_ratio_gradient)
    if not np.isfinite(log_risk_ratio_variance) or log_risk_ratio_variance < 0.0:
        raise np.linalg.LinAlgError('delta-method log-risk-ratio variance is invalid; fixed-tau risk-ratio interval is unavailable')
    log_risk_ratio_standard_error = float(np.sqrt(log_risk_ratio_variance))
    log_risk_ratio_lower = log_risk_ratio - quantile * log_risk_ratio_standard_error
    log_risk_ratio_upper = log_risk_ratio + quantile * log_risk_ratio_standard_error
    risk_ratio_lower = float(np.exp(log_risk_ratio_lower))
    risk_ratio_upper = float(np.exp(log_risk_ratio_upper))
    return FixedTauWaldIntervals(rates=rates.copy(), cfr_logit=np.asarray(cfr_logit, dtype=np.float64), cfr_logit_lower=np.asarray(cfr_logit_lower, dtype=np.float64), cfr_logit_upper=np.asarray(cfr_logit_upper, dtype=np.float64), cfr_lower=np.asarray(cfr_lower, dtype=np.float64), cfr_upper=np.asarray(cfr_upper, dtype=np.float64), risk_ratio=risk_ratio, log_risk_ratio=log_risk_ratio, log_risk_ratio_lower=log_risk_ratio_lower, log_risk_ratio_upper=log_risk_ratio_upper, risk_ratio_lower=risk_ratio_lower, risk_ratio_upper=risk_ratio_upper, information=information, covariance=covariance, cfr_logit_standard_errors=np.asarray(cfr_logit_standard_errors, dtype=np.float64), log_risk_ratio_standard_error=log_risk_ratio_standard_error)
fixed_tau_parameter_intervals = fixed_tau_wald_intervals

# Source section: single_change_gp.py
"""Single-change negative-binomial CFR score scan.

This module is deliberately independent of the earlier ARL, restart, EM, and
e-process implementations.  It implements the statistic

    M_t = max_tau Z_t(tau)^2

and two calibrations:

* a fixed-look Gaussian multiplier calibration for the squared Gaussian-process
  supremum null law;
* a simple finite-horizon Bonferroni boundary for repeated looks without
  future-data access.

All public inference functions estimate CFR and dispersion from observed data.
No truth parameter is accepted by the detector API.
"""
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.stats import chi2

@dataclass(frozen=True)
class NullScoreFit:
    pi: float
    phi: float
    exposure: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    success: bool
    status: str

@dataclass(frozen=True)
class ScoreScan:
    candidates: np.ndarray
    z: np.ndarray
    q: np.ndarray
    efficient_information: np.ndarray
    max_leverage: np.ndarray
    influence: np.ndarray
    statistic: float
    tau_hat: int | None
    direction: int
    null_fit: NullScoreFit

@dataclass(frozen=True)
class FixedLookResult:
    statistic: float
    critical_value: float
    pvalue: float
    reject: bool
    tau_hat: int | None
    direction: int
    pi_hat: float
    phi_hat: float
    fixed_tau: int | None
    fixed_tau_z: float
    fixed_tau_q: float
    naive_chisq_reject: bool
    n_candidates: int
    fit_success: bool
    fit_status: str

@dataclass(frozen=True)
class GaussianOnlineResult:
    """One update from the numerical Gaussian-process OCFR detector."""
    day: int
    is_look: bool
    alarm: bool
    statistic: float
    threshold: float
    estimated_change_day: int | None
    direction: int
    pi_hat: float
    phi_hat: float
    n_candidates: int
    cumulative_alpha: float
    numerical_crossing_probability: float
    status: str

def _validate_stream(cases: Sequence[float], deaths: Sequence[int | float]) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(cases, dtype=np.float64)
    d = np.asarray(deaths, dtype=np.float64)
    if c.ndim != 1 or d.ndim != 1 or c.shape != d.shape or (c.size < 3):
        raise ValueError('cases and deaths must be one-dimensional equal-length arrays')
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(d)) or np.any(c < 0.0) or np.any(d < 0.0):
        raise ValueError('cases and deaths must be finite and non-negative')
    return (np.ascontiguousarray(c), np.ascontiguousarray(d))

def exposure_components(cases: Sequence[float], delay_pmf: Sequence[float], *, lag_offset: int=1) -> tuple[np.ndarray, np.ndarray]:
    """Return total exposure and all post-candidate exposure columns.

    The returned matrix has shape ``(n, n-1)``.  Column ``tau-1`` is
    ``B_s(tau)`` for zero-based cohort candidate ``tau``.  A one-day delay is
    represented by ``delay_pmf[0]``.
    """
    c = np.asarray(cases, dtype=np.float64)
    f = validate_delay_pmf(delay_pmf)
    if c.ndim != 1 or c.size < 3 or np.any(c < 0.0) or (not np.all(np.isfinite(c))):
        raise ValueError('cases must be a finite non-negative one-dimensional array')
    if lag_offset not in (0, 1):
        raise ValueError('lag_offset must be 0 or 1')
    n = c.size
    cohort_contribution = np.zeros((n, n), dtype=np.float64)
    for lag_index, probability in enumerate(f):
        lag = lag_index + lag_offset
        if lag >= n or probability == 0.0:
            continue
        cohort = np.arange(0, n - lag)
        cohort_contribution[cohort + lag, cohort] = c[cohort] * probability
    reverse_sum = np.cumsum(cohort_contribution[:, ::-1], axis=1)[:, ::-1]
    total = reverse_sum[:, 0]
    post = reverse_sum[:, 1:n]
    return (np.ascontiguousarray(total), np.ascontiguousarray(post))

def _fit_score_null_from_exposure(deaths: np.ndarray, exposure: np.ndarray, *, observation_start: int) -> NullScoreFit:
    """Fit the NB2 null from a validated, precomputed exposure vector."""
    positive = (exposure > MU_FLOOR) & (np.arange(exposure.size, dtype=np.int64) >= int(observation_start))
    if int(np.sum(positive)) < 3:
        zeros = np.maximum(exposure * 0.0, MU_FLOOR)
        return NullScoreFit(pi=np.nan, phi=np.nan, exposure=exposure, mean=zeros, variance=zeros, success=False, status='insufficient_positive_exposure')
    fit = fit_nb_identity(deaths[positive], np.zeros(int(np.sum(positive)), dtype=np.float64), exposure[positive, None])
    mean = np.maximum(float(fit.rates[0]) * exposure, MU_FLOOR)
    variance = nb_variance(mean, float(fit.phi))
    return NullScoreFit(pi=float(fit.rates[0]), phi=float(fit.phi), exposure=exposure, mean=mean, variance=variance, success=bool(fit.success), status=str(fit.status))

def fit_score_null(cases: Sequence[float], deaths: Sequence[int | float], delay_pmf: Sequence[float], *, observation_start: int=0, lag_offset: int=1) -> NullScoreFit:
    """Fit the constant-CFR NB2 null using only the supplied prefix."""
    c, d = _validate_stream(cases, deaths)
    if not isinstance(observation_start, (int, np.integer)) or observation_start < 0 or observation_start >= c.size:
        raise ValueError('observation_start must index the supplied prefix')
    exposure, _ = exposure_components(c, delay_pmf, lag_offset=lag_offset)
    return _fit_score_null_from_exposure(d, exposure, observation_start=observation_start)

def _score_scan_python(cases: Sequence[float], deaths: Sequence[int | float], delay_pmf: Sequence[float], *, use_lindeberg_gate: bool=True, observation_start: int=0, candidate_start: int=1, lag_offset: int=1) -> ScoreScan:
    """Compute the maximal efficient-score scan at one observed prefix."""
    c, d = _validate_stream(cases, deaths)
    if not isinstance(candidate_start, (int, np.integer)) or candidate_start < 1 or candidate_start >= c.size:
        raise ValueError('candidate_start must lie inside the supplied prefix')
    exposure, post = exposure_components(c, delay_pmf, lag_offset=lag_offset)
    fit = _fit_score_null_from_exposure(d, exposure, observation_start=observation_start)
    empty = np.empty(0, dtype=np.float64)
    if not fit.success:
        return ScoreScan(candidates=np.empty(0, dtype=np.int64), z=empty, q=empty, efficient_information=empty, max_leverage=empty, influence=np.empty((c.size, 0), dtype=np.float64), statistic=np.nan, tau_hat=None, direction=0, null_fit=fit)
    inv_variance = 1.0 / np.maximum(fit.variance, MU_FLOOR)
    active = np.arange(c.size, dtype=np.int64) >= int(observation_start)
    i_pi_pi = float(np.sum(np.square(exposure[active]) * inv_variance[active]))
    if not np.isfinite(i_pi_pi) or i_pi_pi <= MU_FLOOR:
        failed = NullScoreFit(pi=fit.pi, phi=fit.phi, exposure=fit.exposure, mean=fit.mean, variance=fit.variance, success=False, status='degenerate_baseline_information')
        return ScoreScan(candidates=np.empty(0, dtype=np.int64), z=empty, q=empty, efficient_information=empty, max_leverage=empty, influence=np.empty((c.size, 0), dtype=np.float64), statistic=np.nan, tau_hat=None, direction=0, null_fit=failed)
    cross = np.sum(post[active] * (exposure[active] * inv_variance[active])[:, None], axis=0)
    projection = cross / i_pi_pi
    residualized = post - exposure[:, None] * projection[None, :]
    residualized[~active] = 0.0
    variance_share_numer = np.square(residualized) * inv_variance[:, None]
    information = np.sum(variance_share_numer, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        max_leverage = np.max(variance_share_numer, axis=0) / information
    scale = max(1.0, float(np.nanmax(information, initial=0.0)))
    valid = np.isfinite(information) & (information > 1e-12 * scale) & np.isfinite(max_leverage)
    candidates_all = np.arange(1, c.size, dtype=np.int64)
    valid &= candidates_all >= int(candidate_start)
    if use_lindeberg_gate:
        n_positive = max(1, int(np.sum((exposure > MU_FLOOR) & active)))
        valid &= max_leverage <= 1.0 / np.sqrt(float(n_positive))
    candidates = candidates_all[valid]
    if candidates.size == 0:
        return ScoreScan(candidates=candidates, z=empty, q=empty, efficient_information=empty, max_leverage=empty, influence=np.empty((c.size, 0), dtype=np.float64), statistic=np.nan, tau_hat=None, direction=0, null_fit=fit)
    g = residualized[:, valid]
    j = information[valid]
    efficient_score = np.sum(g * ((d - fit.mean) * inv_variance)[:, None], axis=0)
    z = efficient_score / np.sqrt(j)
    q = np.square(z)
    winner = int(np.argmax(q))
    influence = g / np.sqrt(fit.variance)[:, None] / np.sqrt(j)[None, :]
    direction = int(np.sign(z[winner]))
    return ScoreScan(candidates=candidates, z=np.ascontiguousarray(z), q=np.ascontiguousarray(q), efficient_information=np.ascontiguousarray(j), max_leverage=np.ascontiguousarray(max_leverage[valid]), influence=np.ascontiguousarray(influence), statistic=float(q[winner]), tau_hat=int(candidates[winner]), direction=direction, null_fit=fit)

def _score_scan_c(cases: Sequence[float], deaths: Sequence[int | float], delay_pmf: Sequence[float], *, use_lindeberg_gate: bool, observation_start: int, candidate_start: int, lag_offset: int=1) -> ScoreScan:
    """Compute the current score fields with the verified C99 kernel."""
    c, d = _validate_stream(cases, deaths)
    if not isinstance(candidate_start, (int, np.integer)) or candidate_start < 1 or candidate_start >= c.size:
        raise ValueError('candidate_start must lie inside the supplied prefix')
    kernels = FastKernels()
    exposure, post = kernels.gp_exposure_components(c, validate_delay_pmf(delay_pmf), lag_offset=lag_offset)
    fit = _fit_score_null_from_exposure(d, exposure, observation_start=observation_start)
    empty = np.empty(0, dtype=np.float64)
    if not fit.success:
        return ScoreScan(candidates=np.empty(0, dtype=np.int64), z=empty, q=empty, efficient_information=empty, max_leverage=empty, influence=np.empty((c.size, 0), dtype=np.float64), statistic=np.nan, tau_hat=None, direction=0, null_fit=fit)
    information, max_leverage, z_all, q_all, influence_all, valid = kernels.gp_score_fields(post, fit.exposure, d, fit.mean, fit.variance, observation_start=observation_start, candidate_start=candidate_start, use_lindeberg_gate=use_lindeberg_gate)
    candidates_all = np.arange(1, c.size, dtype=np.int64)
    candidates = candidates_all[valid]
    if candidates.size == 0:
        return ScoreScan(candidates=candidates, z=empty, q=empty, efficient_information=empty, max_leverage=empty, influence=np.empty((c.size, 0), dtype=np.float64), statistic=np.nan, tau_hat=None, direction=0, null_fit=fit)
    z = np.ascontiguousarray(z_all[valid])
    q = np.ascontiguousarray(q_all[valid])
    winner = int(np.argmax(q))
    return ScoreScan(candidates=np.ascontiguousarray(candidates), z=z, q=q, efficient_information=np.ascontiguousarray(information[valid]), max_leverage=np.ascontiguousarray(max_leverage[valid]), influence=np.ascontiguousarray(influence_all[:, valid]), statistic=float(q[winner]), tau_hat=int(candidates[winner]), direction=int(np.sign(z[winner])), null_fit=fit)

def score_scan(cases: Sequence[float], deaths: Sequence[int | float], delay_pmf: Sequence[float], *, use_lindeberg_gate: bool=True, observation_start: int=0, candidate_start: int=1, backend: str='c', lag_offset: int=1) -> ScoreScan:
    """Compute the maximal efficient-score scan on Python or C99.

    ``backend='auto'`` prefers the compiled C99 kernels when they are available
    and otherwise falls back to the transparent NumPy reference. Requesting
    ``backend='c'`` is strict and raises if the native library cannot be used.
    """
    if backend not in {'auto', 'python', 'c'}:
        raise ValueError("backend must be 'auto', 'python', or 'c'")
    if backend in {'auto', 'c'}:
        try:
            return _score_scan_c(cases, deaths, delay_pmf, use_lindeberg_gate=use_lindeberg_gate, observation_start=observation_start, candidate_start=candidate_start, lag_offset=lag_offset)
        except (OSError, RuntimeError, AttributeError):
            if backend == 'c':
                raise
    return _score_scan_python(cases, deaths, delay_pmf, use_lindeberg_gate=use_lindeberg_gate, observation_start=observation_start, candidate_start=candidate_start, lag_offset=lag_offset)

def report_day_score_scan(
    aligned_scan: ScoreScan,
    deaths: Sequence[int | float],
    *,
    observation_start: int = 0,
    use_lindeberg_gate: bool = True,
) -> ScoreScan:
    """Matched report-day score ablation for cohort alignment.

    The fitted null mean, dispersion, observation prefix, nuisance direction,
    and candidate grid are inherited from ``aligned_scan``.  Only the change
    direction differs: candidate ``tau`` scales report-day exposure from
    calendar day ``tau`` onward instead of aligning deaths to originating case
    cohorts.  This makes the contrast interpretable as an alignment ablation,
    rather than as a comparison of unrelated fitted models.
    """
    fit = aligned_scan.null_fit
    d = np.asarray(deaths, dtype=np.float64)
    candidates = np.asarray(aligned_scan.candidates, dtype=np.int64)
    empty = np.empty(0, dtype=np.float64)
    if not fit.success or candidates.size == 0:
        return ScoreScan(candidates=np.empty(0, dtype=np.int64), z=empty, q=empty,
                         efficient_information=empty, max_leverage=empty,
                         influence=np.empty((d.size, 0), dtype=np.float64),
                         statistic=np.nan, tau_hat=None, direction=0, null_fit=fit)
    if d.ndim != 1 or d.size != fit.exposure.size:
        raise ValueError("deaths must match the fitted prefix")
    active = np.arange(d.size, dtype=np.int64) >= int(observation_start)
    inv_variance = 1.0 / np.maximum(fit.variance, MU_FLOOR)
    baseline = fit.exposure
    i_pi_pi = float(np.sum(np.square(baseline[active]) * inv_variance[active]))
    calendar = np.arange(d.size, dtype=np.int64)[:, None]
    directions = baseline[:, None] * (calendar >= candidates[None, :])
    cross = np.sum(
        directions[active] * (baseline[active] * inv_variance[active])[:, None],
        axis=0,
    )
    residualized = directions - baseline[:, None] * (cross / i_pi_pi)[None, :]
    residualized[~active] = 0.0
    shares = np.square(residualized) * inv_variance[:, None]
    information = np.sum(shares, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        leverage = np.max(shares, axis=0) / information
    scale = max(1.0, float(np.nanmax(information, initial=0.0)))
    valid = np.isfinite(information) & (information > 1e-12 * scale) & np.isfinite(leverage)
    if use_lindeberg_gate:
        n_positive = max(1, int(np.sum((baseline > MU_FLOOR) & active)))
        valid &= leverage <= 1.0 / np.sqrt(float(n_positive))
    candidates = candidates[valid]
    if candidates.size == 0:
        return ScoreScan(candidates=candidates, z=empty, q=empty,
                         efficient_information=empty, max_leverage=empty,
                         influence=np.empty((d.size, 0), dtype=np.float64),
                         statistic=np.nan, tau_hat=None, direction=0, null_fit=fit)
    g = residualized[:, valid]
    j = information[valid]
    score = np.sum(g * ((d - fit.mean) * inv_variance)[:, None], axis=0)
    z = score / np.sqrt(j)
    q = np.square(z)
    winner = int(np.argmax(q))
    influence = g / np.sqrt(fit.variance)[:, None] / np.sqrt(j)[None, :]
    return ScoreScan(
        candidates=np.ascontiguousarray(candidates), z=np.ascontiguousarray(z),
        q=np.ascontiguousarray(q), efficient_information=np.ascontiguousarray(j),
        max_leverage=np.ascontiguousarray(leverage[valid]),
        influence=np.ascontiguousarray(influence), statistic=float(q[winner]),
        tau_hat=int(candidates[winner]), direction=int(np.sign(z[winner])), null_fit=fit,
    )

def _validate_multiplier_inputs(multipliers: Sequence[Sequence[float]], influence: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize the shared multiplier-transform inputs."""
    try:
        draws = np.asarray(multipliers, dtype=np.float64)
        weights = np.asarray(influence, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('multiplier inputs must be numeric arrays') from exc
    if draws.ndim != 2:
        raise ValueError('multipliers must have shape (B, n)')
    if weights.ndim != 2:
        raise ValueError('influence must have shape (n, k)')
    if draws.shape[0] < 1 or draws.shape[1] < 1:
        raise ValueError('multipliers must have positive dimensions')
    if weights.shape[0] != draws.shape[1]:
        raise ValueError('multipliers and influence have incompatible observation dimensions')
    if not np.all(np.isfinite(draws)):
        raise ValueError('multipliers must contain only finite values')
    if not np.all(np.isfinite(weights)):
        raise ValueError('influence must contain only finite values')
    return (np.ascontiguousarray(draws, dtype=np.float64), np.ascontiguousarray(weights, dtype=np.float64))

def multiplier_statistics_from_draws(multipliers: Sequence[Sequence[float]], influence: Sequence[Sequence[float]], *, backend: str='c') -> np.ndarray:
    """Transform supplied Gaussian draws into squared GP suprema.

    ``multipliers`` is a shared, row-major ``(B, n)`` matrix of NumPy-generated
    standard-normal draws and ``influence`` is the corresponding ``(n, k)``
    influence matrix.  ``backend='auto'`` uses the compiled C99 transform when
    available and falls back to the NumPy reference; ``backend='c'`` requests
    the native transform strictly.  Keeping draw generation in Python makes
    the two implementations directly reproducible and numerically comparable.

    If ``influence`` has zero candidate columns, the supremum is undefined and
    a length-``B`` vector of ``NaN`` values is returned, matching the scan API.
    """
    if backend not in {'auto', 'python', 'c'}:
        raise ValueError("backend must be 'auto', 'python', or 'c'")
    draws, weights = _validate_multiplier_inputs(multipliers, influence)
    if weights.shape[1] == 0:
        return np.full(draws.shape[0], np.nan, dtype=np.float64)
    if backend in {'auto', 'c'}:
        try:
            return FastKernels().multiplier_maxima(draws, weights)
        except (OSError, RuntimeError, AttributeError):
            if backend == 'c':
                raise
    gaussian_process = draws @ weights
    return np.max(np.square(gaussian_process), axis=1)

def multiplier_statistics(scan: ScoreScan, n_multiplier: int, rng: np.random.Generator, *, backend: str='c') -> np.ndarray:
    """Generate NumPy multipliers and transform them into squared GP suprema."""
    if n_multiplier < 19:
        raise ValueError('n_multiplier must be at least 19')
    if scan.candidates.size == 0:
        return np.full(n_multiplier, np.nan, dtype=np.float64)
    multipliers = rng.standard_normal((n_multiplier, scan.influence.shape[0]))
    return multiplier_statistics_from_draws(multipliers, scan.influence, backend=backend)

def monte_carlo_critical_value(values: Sequence[float], alpha: float) -> float:
    """Conservative ``(B+1)`` order-statistic critical value."""
    if not 0.0 < alpha < 1.0:
        raise ValueError('alpha must lie in (0, 1)')
    x = np.sort(np.asarray(values, dtype=np.float64))
    if x.ndim != 1 or x.size == 0 or (not np.all(np.isfinite(x))):
        raise ValueError('values must be a non-empty finite vector')
    order = int(np.ceil((1.0 - alpha) * (x.size + 1)))
    if order > x.size:
        return np.inf
    return float(x[order - 1])

def fixed_look_test(cases: Sequence[float], deaths: Sequence[int | float], delay_pmf: Sequence[float], *, alpha: float=0.05, n_multiplier: int=499, rng: np.random.Generator, fixed_tau: int | None=None, use_lindeberg_gate: bool=True, backend: str='c') -> FixedLookResult:
    """Test one supplied prefix using a Gaussian multiplier critical value."""
    scan = score_scan(cases, deaths, delay_pmf, use_lindeberg_gate=use_lindeberg_gate, backend=backend)
    if not scan.null_fit.success or scan.candidates.size == 0:
        return FixedLookResult(statistic=np.nan, critical_value=np.inf, pvalue=1.0, reject=False, tau_hat=None, direction=0, pi_hat=scan.null_fit.pi, phi_hat=scan.null_fit.phi, fixed_tau=fixed_tau, fixed_tau_z=np.nan, fixed_tau_q=np.nan, naive_chisq_reject=False, n_candidates=int(scan.candidates.size), fit_success=False, fit_status=scan.null_fit.status)
    simulated = multiplier_statistics(scan, n_multiplier, rng, backend=backend)
    critical = monte_carlo_critical_value(simulated, alpha)
    exceedances = int(np.sum(simulated >= scan.statistic))
    pvalue = (1.0 + exceedances) / (n_multiplier + 1.0)
    fixed_z = np.nan
    fixed_q = np.nan
    if fixed_tau is not None:
        location = np.flatnonzero(scan.candidates == int(fixed_tau))
        if location.size:
            fixed_z = float(scan.z[int(location[0])])
            fixed_q = float(scan.q[int(location[0])])
    return FixedLookResult(statistic=scan.statistic, critical_value=critical, pvalue=float(pvalue), reject=bool(scan.statistic > critical), tau_hat=scan.tau_hat, direction=scan.direction, pi_hat=scan.null_fit.pi, phi_hat=scan.null_fit.phi, fixed_tau=fixed_tau, fixed_tau_z=fixed_z, fixed_tau_q=fixed_q, naive_chisq_reject=bool(scan.statistic > chi2.ppf(1.0 - alpha, df=1)), n_candidates=int(scan.candidates.size), fit_success=True, fit_status=scan.null_fit.status)

def bonferroni_boundary(horizon: int, alpha: float=0.05, look_every: int=7) -> tuple[float, tuple[int, ...], int]:
    """Return an analytic familywise boundary for a fixed look schedule."""
    if horizon < 3:
        raise ValueError('horizon must be at least 3')
    if look_every < 1:
        raise ValueError('look_every must be positive')
    if not 0.0 < alpha < 1.0:
        raise ValueError('alpha must lie in (0, 1)')
    looks = tuple(range(look_every, horizon + 1, look_every))
    if not looks or looks[-1] != horizon:
        looks = looks + (horizon,)
    n_tests = int(sum((max(0, prefix - 1) for prefix in looks)))
    threshold = float(chi2.ppf(1.0 - alpha / n_tests, df=1))
    return (threshold, looks, n_tests)

class OnlineGaussianProcessScore:
    """Online one-change detector with a numerical GP boundary.

    Each scheduled look uses a current-prefix Gaussian multiplier quantile at
    level ``alpha / J``. If the look-specific working approximation attains
    that local level, Bonferroni's inequality gives the episode-level bound
    without a cross-look dependence model or spending-shape parameter. The
    detector never constructs a future score or exposure.
    """

    def __init__(self, delay_pmf: Sequence[float], *, horizon: int, alpha: float=0.05, look_every: int=7, n_multiplier: int=10000, rng: np.random.Generator, use_lindeberg_gate: bool=True, backend: str='c') -> None:
        if n_multiplier < 399:
            raise ValueError('n_multiplier must be at least 399')
        _, looks, _ = bonferroni_boundary(horizon, alpha, look_every)
        self._delay = validate_delay_pmf(delay_pmf)
        self._horizon = int(horizon)
        self._alpha = float(alpha)
        self._looks_ordered = looks
        self._look_to_index = {look: index + 1 for index, look in enumerate(looks)}
        self._gate = bool(use_lindeberg_gate)
        if backend not in {'auto', 'python', 'c'}:
            raise ValueError("backend must be 'auto', 'python', or 'c'")
        self._backend = backend
        self._multipliers = np.ascontiguousarray(rng.standard_normal((int(n_multiplier), int(horizon))), dtype=np.float64)
        self._crossed = np.zeros(int(n_multiplier), dtype=bool)
        self._cases: list[float] = []
        self._deaths: list[float] = []
        self._stopped = False

    @property
    def n_multiplier(self) -> int:
        return int(self._multipliers.shape[0])

    @property
    def boundary_method(self) -> str:
        return 'online_gaussian_process_bonferroni_looks'

    def update(self, cases_today: float, deaths_today: int | float) -> GaussianOnlineResult:
        if self._stopped:
            raise RuntimeError('the one-change detector has already stopped')
        if len(self._cases) >= self._horizon:
            raise RuntimeError('the declared monitoring horizon has ended')
        case = float(cases_today)
        death = float(deaths_today)
        if not np.isfinite(case) or not np.isfinite(death) or case < 0.0 or (death < 0.0):
            raise ValueError('daily cases and deaths must be finite and non-negative')
        self._cases.append(case)
        self._deaths.append(death)
        prefix = len(self._cases)
        look_index = self._look_to_index.get(prefix)
        if look_index is None or prefix < 3:
            return GaussianOnlineResult(day=prefix - 1, is_look=False, alarm=False, statistic=np.nan, threshold=np.inf, estimated_change_day=None, direction=0, pi_hat=np.nan, phi_hat=np.nan, n_candidates=0, cumulative_alpha=0.0, numerical_crossing_probability=float(np.mean(self._crossed)), status='not_a_scheduled_look')
        per_look_alpha = self._alpha / len(self._looks_ordered)
        cumulative_alpha = look_index * per_look_alpha
        scan = score_scan(self._cases, self._deaths, self._delay, use_lindeberg_gate=self._gate, backend=self._backend)
        if not scan.null_fit.success or scan.candidates.size == 0:
            return GaussianOnlineResult(day=prefix - 1, is_look=True, alarm=False, statistic=scan.statistic, threshold=np.inf, estimated_change_day=None, direction=0, pi_hat=scan.null_fit.pi, phi_hat=scan.null_fit.phi, n_candidates=int(scan.candidates.size), cumulative_alpha=cumulative_alpha, numerical_crossing_probability=float(np.mean(self._crossed)), status=scan.null_fit.status)
        maxima = multiplier_statistics_from_draws(self._multipliers[:, :prefix], scan.influence, backend=self._backend)
        threshold = monte_carlo_critical_value(maxima, per_look_alpha)
        self._crossed |= maxima > threshold
        alarm = bool(np.isfinite(scan.statistic) and scan.statistic > threshold)
        if alarm:
            self._stopped = True
        return GaussianOnlineResult(day=prefix - 1, is_look=True, alarm=alarm, statistic=scan.statistic, threshold=threshold, estimated_change_day=scan.tau_hat if alarm else None, direction=scan.direction if alarm else 0, pi_hat=scan.null_fit.pi, phi_hat=scan.null_fit.phi, n_candidates=int(scan.candidates.size), cumulative_alpha=cumulative_alpha, numerical_crossing_probability=float(np.mean(self._crossed)), status=scan.null_fit.status)

# Source section: published_benchmarks.py
"""Online published-framework comparators for the formal OCFR study.

The implementations expose statistics only. Thresholds are deliberately
absent: every comparator must be calibrated on independent null streams.
"""
from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    day: int
    statistic: float
    estimated_calendar_change_day: int | None
    estimated_cohort_change_day: int | None
    direction: int

def _delay_median(delay_pmf: Sequence[float], lag_offset: int=1) -> int:
    probabilities = np.asarray(delay_pmf, dtype=np.float64)
    return int(np.searchsorted(np.cumsum(probabilities), 0.5) + lag_offset)

def _baseline_residuals(cases: np.ndarray, deaths: np.ndarray, delay_pmf: np.ndarray, reference_end: int, observation_start: int=0, lag_offset: int=1) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit a reference null while retaining the complete case history.

    ``observation_start`` marks the first outcome used for inference.  Cases
    before that index remain in the convolution history, while the reference
    fit uses only outcomes in ``[observation_start, reference_end)``.  This is
    the same filtration used by the OCFR detector and avoids treating the
    history rows as if they had an observed death exposure.
    """
    c = np.asarray(cases, dtype=np.float64)
    d = np.asarray(deaths, dtype=np.float64)
    if c.ndim != 1 or d.ndim != 1 or c.shape != d.shape:
        raise ValueError('cases and deaths must be one-dimensional equal-length arrays')
    if not isinstance(reference_end, (int, np.integer)) or not isinstance(observation_start, (int, np.integer)) or observation_start < 0 or (observation_start >= reference_end) or (reference_end >= c.size):
        raise ValueError('require 0 <= observation_start < reference_end < stream length')
    fit = fit_score_null(c[:reference_end], d[:reference_end], delay_pmf, observation_start=int(observation_start), lag_offset=lag_offset)
    if not fit.success:
        raise RuntimeError(f'benchmark reference fit failed: {fit.status}')
    exposure, _ = exposure_components(c, delay_pmf, lag_offset=lag_offset)
    mean = np.maximum(fit.pi * exposure, MU_FLOOR)
    variance = nb_variance(mean, fit.phi)
    residuals = (d - mean) / np.sqrt(np.maximum(variance, MU_FLOOR))
    return (residuals, mean, float(fit.pi), float(fit.phi))

def _profile_nb_scale(y: np.ndarray, baseline_mean: np.ndarray, phi: float) -> tuple[float, float]:
    """Profile a multiplicative NB mean shift with safeguarded Newton steps."""
    if y.size == 0:
        return (1.0, 0.0)
    mu = np.maximum(baseline_mean, MU_FLOOR)
    gamma = float(np.clip(np.sum(y) / max(np.sum(mu), MU_FLOOR), 0.0001, 10000.0))
    if np.isinf(phi) or phi >= 10000000.0:
        fitted = gamma * mu
        return (gamma, max(0.0, 2.0 * (nb_loglik(y, fitted, phi) - nb_loglik(y, mu, phi))))
    for _ in range(40):
        denominator = phi + gamma * mu
        score = float(np.sum(y / gamma - (y + phi) * mu / denominator))
        hessian = float(np.sum(-y / (gamma * gamma) + (y + phi) * mu * mu / denominator ** 2))
        if not np.isfinite(score) or not np.isfinite(hessian) or hessian >= 0.0:
            break
        proposal = gamma - score / hessian
        if proposal <= 1e-06 or proposal >= 1000000.0:
            proposal = np.sqrt(gamma * np.clip(proposal, 1e-06, 1000000.0))
        if abs(proposal - gamma) <= 1e-09 * (1.0 + gamma):
            gamma = float(proposal)
            break
        gamma = float(proposal)
    gamma = float(np.clip(gamma, 1e-06, 1000000.0))
    lr = 2.0 * (nb_loglik(y, gamma * mu, phi) - nb_loglik(y, mu, phi))
    return (gamma, max(0.0, float(lr)))

def nb_surveillance_glr(cases: Sequence[float], deaths: Sequence[int | float], delay_pmf: Sequence[float], *, reference_end: int=28, observation_start: int=0, lag_offset: int=1) -> BenchmarkResult:
    """Window-unlimited two-sided NB intercept GLR.

    This is the count-regression-chart statistic of Höhle and Paul applied to
    the daily death mean. Its change location is a calendar-death location;
    subtraction of the prespecified delay median gives a transparent cohort
    proxy for localization comparisons.
    """
    c = np.asarray(cases, dtype=np.float64)
    d = np.asarray(deaths, dtype=np.float64)
    f = np.asarray(delay_pmf, dtype=np.float64)
    if c.shape != d.shape or c.ndim != 1 or c.size <= reference_end:
        raise ValueError('aligned stream longer than reference_end required')
    _, mean, _, phi = _baseline_residuals(c, d, f, reference_end, observation_start=observation_start, lag_offset=lag_offset)
    best_statistic = -np.inf
    best_calendar: int | None = None
    best_scale = 1.0
    for change in range(reference_end, c.size):
        scale, statistic = _profile_nb_scale(d[change:], mean[change:], phi)
        if statistic > best_statistic:
            best_statistic = statistic
            best_calendar = change
            best_scale = scale
    lag = _delay_median(f, lag_offset)
    cohort = None if best_calendar is None else max(0, best_calendar - lag)
    return BenchmarkResult(method='NB-surveillance-GLR', day=c.size - 1, statistic=float(best_statistic), estimated_calendar_change_day=best_calendar, estimated_cohort_change_day=cohort, direction=int(np.sign(best_scale - 1.0)))

def focus_working_model(cases: Sequence[float], deaths: Sequence[int | float], delay_pmf: Sequence[float], *, reference_end: int=28, observation_start: int=0, lag_offset: int=1) -> BenchmarkResult:
    """Exact Gaussian mean-change GLR statistic on NB Pearson residuals.

    The statistic is the one computed by Gaussian FOCuS; this transparent
    reference implementation scans all candidates rather than using functional
    pruning. It is therefore numerically exact but not a runtime claim for the
    published pruning algorithm.
    """
    c = np.asarray(cases, dtype=np.float64)
    d = np.asarray(deaths, dtype=np.float64)
    f = np.asarray(delay_pmf, dtype=np.float64)
    residuals, _, _, _ = _baseline_residuals(c, d, f, reference_end, observation_start=observation_start, lag_offset=lag_offset)
    monitor = residuals[reference_end:]
    reversed_sum = np.cumsum(monitor[::-1])[::-1]
    lengths = np.arange(monitor.size, 0, -1, dtype=np.float64)
    statistics = np.square(reversed_sum) / lengths
    index = int(np.argmax(statistics))
    calendar = reference_end + index
    lag = _delay_median(f, lag_offset)
    return BenchmarkResult(method='FOCuS-working-model', day=c.size - 1, statistic=float(statistics[index]), estimated_calendar_change_day=calendar, estimated_cohort_change_day=max(0, calendar - lag), direction=int(np.sign(reversed_sum[index])))

def page_cusum(cases: Sequence[float], deaths: Sequence[int | float], delay_pmf: Sequence[float], *, reference_end: int=28, allowance: float=0.5, observation_start: int=0, lag_offset: int=1) -> BenchmarkResult:
    """Two-sided Page-CUSUM based on fitted NB Pearson-score increments."""
    if allowance < 0.0:
        raise ValueError('allowance must be nonnegative')
    c = np.asarray(cases, dtype=np.float64)
    d = np.asarray(deaths, dtype=np.float64)
    f = np.asarray(delay_pmf, dtype=np.float64)
    residuals, _, _, _ = _baseline_residuals(c, d, f, reference_end, observation_start=observation_start, lag_offset=lag_offset)
    positive = 0.0
    negative = 0.0
    positive_start = reference_end
    negative_start = reference_end
    for day in range(reference_end, c.size):
        value = float(residuals[day])
        if positive + value - allowance <= 0.0:
            positive = 0.0
            positive_start = day + 1
        else:
            positive += value - allowance
        if negative - value - allowance <= 0.0:
            negative = 0.0
            negative_start = day + 1
        else:
            negative += -value - allowance
    # Report the recursion at the current planned look.  Episode-level maxima
    # are formed by the caller over the common monitoring schedule.
    if positive <= 0.0 and negative <= 0.0:
        statistic = 0.0
        best_calendar = c.size - 1
        best_direction = 0
    elif positive >= negative:
        statistic = positive
        best_calendar = positive_start
        best_direction = 1
    else:
        statistic = negative
        best_calendar = negative_start
        best_direction = -1
    lag = _delay_median(f, lag_offset)
    return BenchmarkResult(method='Page-CUSUM', day=c.size - 1, statistic=float(statistic), estimated_calendar_change_day=best_calendar, estimated_cohort_change_day=max(0, best_calendar - lag), direction=best_direction)

def published_benchmark_statistics(cases: Sequence[float], deaths: Sequence[int | float], delay_pmf: Sequence[float], *, reference_end: int=28, observation_start: int=0, lag_offset: int=1) -> dict[str, BenchmarkResult]:
    """Evaluate all locked comparator statistics on one observed prefix.

    The arrays retain the full calendar coordinate system.  ``observation_start``
    excludes pre-observation deaths from the reference fit and monitoring, but
    does not discard pre-observation cases needed by the delay convolution.
    """
    results = (nb_surveillance_glr(cases, deaths, delay_pmf, reference_end=reference_end, observation_start=observation_start, lag_offset=lag_offset), page_cusum(cases, deaths, delay_pmf, reference_end=reference_end, observation_start=observation_start, lag_offset=lag_offset), focus_working_model(cases, deaths, delay_pmf, reference_end=reference_end, observation_start=observation_start, lag_offset=lag_offset))
    return {result.method: result for result in results}

# Source section: evaluation.py
"""Epidemiologically interpretable operating characteristics for one episode."""
from typing import Sequence
import numpy as np
from scipy.stats import beta, norm

def binomial_interval(successes: int, total: int, *, confidence: float=0.95) -> tuple[float, float]:
    """Return the two-sided Clopper--Pearson interval."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError('successes and total are inconsistent')
    if not 0.0 < confidence < 1.0:
        raise ValueError('confidence must lie in (0, 1)')
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - tail, successes + 1, total - successes))
    return (lower, upper)

def episode_metrics(alarm_days: Sequence[float], *, true_change: int, horizon: int, estimated_changes: Sequence[float] | None=None, timely_window: int=28) -> dict[str, float | int]:
    """Summarize an alternative episode without conditioning away failures.

    A missing alarm, or an alarm before the true change, receives the maximum
    restricted delay ``horizon - true_change + 1``. Localization is summarized
    only for timely post-change alarms and is therefore explicitly secondary.
    """
    alarms = np.asarray(alarm_days, dtype=float)
    if alarms.ndim != 1 or alarms.size == 0:
        raise ValueError('alarm_days must be a non-empty vector')
    if not 0 <= true_change < horizon or timely_window < 0:
        raise ValueError('invalid change, horizon, or timely window')
    finite = np.isfinite(alarms)
    premature = finite & (alarms < true_change)
    post = finite & (alarms >= true_change)
    timely = post & (alarms <= true_change + timely_window)
    restriction = float(horizon - true_change + 1)
    restricted_delay = np.full(alarms.size, restriction, dtype=float)
    restricted_delay[post] = np.minimum(alarms[post] - true_change, restriction)
    out: dict[str, float | int] = {'n': int(alarms.size), 'premature_probability': float(np.mean(premature)), 'post_change_detection_probability': float(np.mean(post)), 'timely_detection_probability': float(np.mean(timely)), 'restricted_mean_delay': float(np.mean(restricted_delay)), 'median_post_change_delay': float(np.median(alarms[post] - true_change)) if np.any(post) else np.nan, 'timely_alarms': int(np.sum(timely))}
    lower, upper = binomial_interval(int(np.sum(timely)), alarms.size)
    out['timely_detection_ci_lower'] = lower
    out['timely_detection_ci_upper'] = upper
    if estimated_changes is not None:
        locations = np.asarray(estimated_changes, dtype=float)
        if locations.shape != alarms.shape:
            raise ValueError('estimated_changes must align with alarm_days')
        valid_location = timely & np.isfinite(locations)
        absolute_error = np.abs(locations[valid_location] - true_change)
        signed_error = locations[valid_location] - true_change
        out.update({'localization_median_absolute_error': float(np.median(absolute_error)) if absolute_error.size else np.nan, 'localization_p90_absolute_error': float(np.quantile(absolute_error, 0.9)) if absolute_error.size else np.nan, 'localization_signed_bias': float(np.mean(signed_error)) if signed_error.size else np.nan})
    return out

def paired_binary_difference_interval(first: Sequence[bool], second: Sequence[bool], *, confidence: float=0.95) -> tuple[float, float, float]:
    """Normal-score interval for a paired difference in probabilities."""
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError('paired inputs must be aligned non-empty vectors')
    paired = a.astype(float) - b.astype(float)
    estimate = float(np.mean(paired))
    standard_error = float(np.std(paired, ddof=1) / np.sqrt(paired.size))
    quantile = float(norm.ppf(0.5 + confidence / 2.0))
    return (estimate, estimate - quantile * standard_error, estimate + quantile * standard_error)

# Source section: robustness.py
"""Epidemiological data-generating perturbations for OCFR robustness studies."""
from typing import Sequence
import numpy as np
from scipy.stats import gamma, lognorm

def gamma_delay_pmf(mean_days: float, *, shape: float=4.0, maximum_lag: int=45) -> np.ndarray:
    if mean_days <= 0.0 or shape <= 0.0 or maximum_lag < 2:
        raise ValueError('invalid gamma delay parameters')
    edges = np.arange(maximum_lag + 1, dtype=np.float64)
    probabilities = np.diff(gamma.cdf(edges, a=shape, scale=mean_days / shape))
    probabilities /= probabilities.sum()
    return validate_delay_pmf(probabilities)

def lognormal_delay_pmf(mean_days: float, *, coefficient_of_variation: float=0.6, maximum_lag: int=45) -> np.ndarray:
    if mean_days <= 0.0 or coefficient_of_variation <= 0.0:
        raise ValueError('invalid log-normal delay parameters')
    sigma2 = np.log1p(coefficient_of_variation ** 2)
    sigma = np.sqrt(sigma2)
    scale = mean_days / np.exp(0.5 * sigma2)
    edges = np.arange(maximum_lag + 1, dtype=np.float64)
    probabilities = np.diff(lognorm.cdf(edges, s=sigma, scale=scale))
    probabilities /= probabilities.sum()
    return validate_delay_pmf(probabilities)

def bimodal_delay_pmf(mean_days: float=14.0, *, maximum_lag: int=45) -> np.ndarray:
    """A prespecified short/long-delay mixture with approximately fixed mean."""
    short = gamma_delay_pmf(0.55 * mean_days, shape=5.0, maximum_lag=maximum_lag)
    long = gamma_delay_pmf(1.45 * mean_days, shape=7.0, maximum_lag=maximum_lag)
    return validate_delay_pmf(0.5 * short + 0.5 * long)

def gradual_cfr_path(horizon: int, left: float, right: float, midpoint: int, transition_days: float) -> np.ndarray:
    if horizon < 3 or not 0 < left < 1 or (not 0 < right < 1):
        raise ValueError('invalid CFR path parameters')
    if not 0 <= midpoint < horizon or transition_days <= 0.0:
        raise ValueError('invalid transition location or duration')
    days = np.arange(horizon, dtype=np.float64)
    slope = 2.0 * np.log(9.0) / transition_days
    weight = 1.0 / (1.0 + np.exp(-slope * (days - midpoint)))
    return left + (right - left) * weight

def apply_fixed_reporting_lag(deaths: Sequence[int | float], lag_days: int) -> np.ndarray:
    """Return the real-time vintage after a fixed reporting delay."""
    values = np.asarray(deaths, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0) or lag_days < 0:
        raise ValueError('invalid deaths or reporting lag')
    out = np.zeros_like(values)
    if lag_days == 0:
        return values.copy()
    if lag_days < values.size:
        out[lag_days:] = values[:-lag_days]
    return out

def apply_weekend_transfer(deaths: Sequence[int | float], transfer_probability: float, rng: np.random.Generator) -> np.ndarray:
    """Move a fraction of Saturday/Sunday reports to the next Monday.

    The sampler can be NumPy's generator or :func:`make_sampler`.
    """
    values = np.asarray(deaths, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0) or (not 0.0 <= transfer_probability <= 1.0):
        raise ValueError('invalid deaths or transfer probability')
    out = values.copy()
    for day in range(values.size):
        weekday = day % 7
        if weekday not in (5, 6):
            continue
        moved = int(rng.binomial(int(out[day]), transfer_probability))
        out[day] -= moved
        monday = day + (7 - weekday)
        if monday < out.size:
            out[monday] += moved
    return out

def simulate_deaths_with_dispersion_path(cases: Sequence[float], delay_pmf: Sequence[float], cfr: Sequence[float], phi_path: Sequence[float], rng: np.random.Generator) -> np.ndarray:
    """Generate NB2 deaths with time-varying dispersion.

    The supplied sampler can be NumPy's generator or the C-backed OCFR
    sampler returned by :func:`make_sampler`.
    """
    mean = means_from_cfr_path(cases, delay_pmf, cfr)
    phi = np.asarray(phi_path, dtype=np.float64)
    if phi.shape != mean.shape or np.any(phi <= 0.0):
        raise ValueError('phi_path must align with cases and be positive')
    out = np.empty(mean.size, dtype=np.int64)
    poisson = np.isinf(phi)
    if np.any(poisson):
        out[poisson] = rng.poisson(mean[poisson])
    finite = ~poisson
    if np.any(finite):
        probability = phi[finite] / (phi[finite] + mean[finite])
        out[finite] = rng.negative_binomial(phi[finite], probability)
    return out

def apply_ascertainment_shift(latent_cases: Sequence[float], change_day: int, multiplier: float) -> np.ndarray:
    """Construct an observed-case path after a case-ascertainment change."""
    cases = np.asarray(latent_cases, dtype=np.float64).copy()
    if not 0 <= change_day < cases.size or multiplier <= 0.0:
        raise ValueError('invalid ascertainment shift')
    cases[change_day:] = np.rint(cases[change_day:] * multiplier)
    return cases


def _case_profile_mean(horizon: int, profile: str, level: float) -> np.ndarray:
    """Return the deterministic mean underlying a simulated case profile."""

    t = np.arange(horizon, dtype=float)
    if profile == 'constant':
        mean = np.full(horizon, level)
    elif profile == 'wave':
        mean = level * (
            0.25 + 1.5 * np.exp(-0.5 * ((t - 0.55 * horizon) / (0.18 * horizon)) ** 2)
        )
    elif profile == 'two_wave':
        mean = level * (
            0.2
            + 1.1 * np.exp(-0.5 * ((t - 0.30 * horizon) / (0.12 * horizon)) ** 2)
            + 1.4 * np.exp(-0.5 * ((t - 0.72 * horizon) / (0.10 * horizon)) ** 2)
        )
    elif profile == 'weekly':
        mean = level * (1.0 + 0.25 * np.sin(2.0 * np.pi * t / 7.0))
    elif profile == 'rising':
        mean = level * np.linspace(0.25, 1.75, horizon)
    elif profile == 'falling':
        mean = level * np.linspace(1.75, 0.25, horizon)
    elif profile == 'stochastic_nb':
        mean = level * (
            0.25 + 1.25 * np.exp(-0.5 * ((t - 0.55 * horizon) / (0.18 * horizon)) ** 2)
        )
    else:
        raise ValueError(f'unknown case profile: {profile}')
    return np.maximum(mean, 0.0)


def make_cases(
    horizon: int, profile: str, level: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate independent case counts for a configured profile."""

    mean = _case_profile_mean(horizon, profile, level)
    if profile == 'stochastic_nb':
        case_shape = 10.0
        probability = case_shape / (case_shape + mean)
        return rng.negative_binomial(case_shape, probability).astype(float)
    return rng.poisson(mean).astype(float)


def make_correlated_epidemic_cases(
    horizon: int,
    profile: str,
    level: float,
    rng: np.random.Generator,
    *,
    autoregression: float = 0.90,
    process_sd: float = 0.18,
    case_shape: float = 30.0,
    weekday_amplitude: float = 0.15,
) -> np.ndarray:
    """Generate cases around an AR(1), weekday-modulated epidemic path."""

    if horizon < 3 or level <= 0.0:
        raise ValueError('invalid horizon or level')
    if not 0.0 <= autoregression < 1.0 or process_sd < 0.0:
        raise ValueError('invalid latent-process parameters')
    if case_shape <= 0.0 or not 0.0 <= weekday_amplitude < 1.0:
        raise ValueError('invalid observation parameters')
    target = np.maximum(
        _case_profile_mean(horizon, profile, level), max(1.0, 0.01 * level)
    )
    innovation_sd = process_sd * np.sqrt(max(1.0 - autoregression**2, 0.0))
    latent = np.zeros(horizon, dtype=float)
    latent[0] = rng.normal(0.0, process_sd)
    for day in range(1, horizon):
        latent[day] = autoregression * latent[day - 1] + rng.normal(0.0, innovation_sd)
    weekday = 1.0 + weekday_amplitude * np.sin(
        2.0 * np.pi * np.arange(horizon, dtype=float) / 7.0
    )
    mean = target * np.exp(latent - 0.5 * process_sd**2) * weekday
    probability = case_shape / (case_shape + np.maximum(mean, 0.0))
    return rng.negative_binomial(case_shape, probability).astype(float)


METHOD_REGISTRY = {
    'OCFR': score_scan,
    'NB-GLR': nb_surveillance_glr,
    'Page-CUSUM': page_cusum,
    'FOCuS': focus_working_model,
}


def build_native(*, force: bool = False) -> Path:
    """Build the bundled C99 backend next to this module."""

    root = Path(__file__).resolve().parent
    source = root / 'csrc' / 'ocfr_kernels.c'
    output = root / (
        'ocfr_fast.dll' if sys.platform == 'win32'
        else ('libocfr_fast.dylib' if sys.platform == 'darwin' else 'libocfr_fast.so')
    )
    if output.exists() and not force:
        return output
    if not source.exists():
        raise FileNotFoundError(f'missing bundled native source: {source}')

    if shutil.which('zig'):
        compiler = ['zig', 'cc', '-O3', '-shared']
    elif shutil.which('gcc'):
        compiler = ['gcc', '-O3', '-shared']
    elif shutil.which('clang'):
        compiler = ['clang', '-O3', '-shared']
    elif shutil.which('cl'):
        compiler = ['cl', '/O2', '/LD']
    else:
        try:
            import ziglang  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                'No C compiler found; install Zig/GCC/Clang/MSVC or `pip install ziglang`.'
            ) from exc
        compiler = [sys.executable, '-m', 'ziglang', 'cc', '-O3', '-shared']

    with tempfile.TemporaryDirectory(prefix='ocfr-core-native-') as temporary:
        stage = Path(temporary)
        staged_source = stage / source.name
        staged_output = stage / output.name
        shutil.copy2(source, staged_source)
        if compiler[0].lower() == 'cl':
            command = [*compiler, str(staged_source), f'/Fe:{staged_output}']
        else:
            command = [*compiler, str(staged_source), '-o', str(staged_output)]
            if sys.platform != 'win32':
                command.append('-lm')
        environment = os.environ.copy()
        cache_root = root / '.build_cache'
        environment['ZIG_GLOBAL_CACHE_DIR'] = str(cache_root / 'zig-global')
        environment['ZIG_LOCAL_CACHE_DIR'] = str(cache_root / 'zig-local')
        Path(environment['ZIG_GLOBAL_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
        Path(environment['ZIG_LOCAL_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=root, env=environment, check=True)
        shutil.copy2(staged_output, output)
    return output


def backend_parity_report(seed: int = 20260914) -> dict[str, object]:
    """Compare Python and C samplers and score fields on matched inputs."""

    if not fast_available():
        return {
            'native_available': False,
            'passed': False,
            'reason': 'build the native backend with methods.build_native()',
        }

    python_rng = PythonSampler(seed)
    c_rng = CSampler(seed)
    sampler_checks: dict[str, bool] = {}
    sampler_checks['uniform'] = bool(
        np.array_equal(python_rng.uniform(size=256), c_rng.uniform(size=256))
    )
    sampler_checks['normal'] = bool(
        np.array_equal(python_rng.standard_normal(256), c_rng.standard_normal(256))
    )
    poisson_mean = np.linspace(0.0, 100.0, 256)
    sampler_checks['poisson'] = bool(
        np.array_equal(python_rng.poisson(poisson_mean), c_rng.poisson(poisson_mean))
    )
    shape = np.linspace(0.5, 20.0, 256)
    probability = np.linspace(0.15, 0.95, 256)
    sampler_checks['negative_binomial'] = bool(
        np.array_equal(
            python_rng.negative_binomial(shape, probability),
            c_rng.negative_binomial(shape, probability),
        )
    )
    trials = np.arange(256, dtype=np.int64) % 40
    sampler_checks['binomial'] = bool(
        np.array_equal(
            python_rng.binomial(trials, 0.3), c_rng.binomial(trials, 0.3)
        )
    )
    sampler_checks['integers'] = bool(
        np.array_equal(
            python_rng.integers(0, 10_000, size=256),
            c_rng.integers(0, 10_000, size=256),
        )
    )

    data_rng = PythonSampler(seed + 1)
    delay = gamma_delay_pmf(14.0, shape=4.0, maximum_lag=45)
    cases = make_cases(120, 'wave', 800.0, data_rng)
    cfr = np.full(120, 0.02)
    cfr[60:] = 0.03
    deaths = simulate_nb_deaths(cases, delay, cfr, 10.0, data_rng)
    python_scan = score_scan(cases, deaths, delay, backend='python')
    c_scan = score_scan(cases, deaths, delay, backend='c')
    candidates_equal = bool(np.array_equal(python_scan.candidates, c_scan.candidates))
    max_z_error = (
        float(np.max(np.abs(python_scan.z - c_scan.z)))
        if candidates_equal and python_scan.z.size else float('inf')
    )
    statistic_error = abs(float(python_scan.statistic) - float(c_scan.statistic))
    score_passed = bool(
        candidates_equal
        and python_scan.tau_hat == c_scan.tau_hat
        and max_z_error <= 1e-10
        and statistic_error <= 1e-10
    )
    passed = all(sampler_checks.values()) and score_passed
    return {
        'native_available': True,
        'samplers': sampler_checks,
        'score': {
            'candidates_equal': candidates_equal,
            'tau_equal': python_scan.tau_hat == c_scan.tau_hat,
            'max_abs_z_error': max_z_error,
            'statistic_abs_error': statistic_error,
            'passed': score_passed,
        },
        'passed': bool(passed),
    }


def assert_backend_parity(seed: int = 20260914) -> dict[str, object]:
    """Run :func:`backend_parity_report` and raise on any mismatch."""

    report = backend_parity_report(seed)
    if not report.get('passed', False):
        raise AssertionError(f'Python/C backend parity failed: {report}')
    return report


__all__ = [
    'BenchmarkResult', 'CSampler', 'FixedLookResult', 'FixedTauWaldIntervals',
    'GaussianOnlineResult', 'METHOD_REGISTRY', 'NBFit', 'NullScoreFit',
    'OnlineGaussianProcessScore', 'PythonSampler', 'ScoreScan',
    'apply_ascertainment_shift', 'apply_fixed_reporting_lag',
    'apply_weekend_transfer', 'assert_backend_parity', 'backend_parity_report',
    'bimodal_delay_pmf', 'binomial_interval', 'bonferroni_boundary',
    'build_native', 'convolved_exposure', 'episode_metrics',
    'exposure_components', 'fast_available', 'fit_nb_identity',
    'fit_nb_rates_fixed_phi', 'fit_score_null', 'fixed_look_test',
    'fixed_tau_parameter_intervals', 'fixed_tau_wald_intervals',
    'focus_working_model', 'gamma_delay_pmf', 'gradual_cfr_path',
    'lognormal_delay_pmf', 'make_cases', 'make_correlated_epidemic_cases',
    'make_sampler', 'means_from_cfr_path', 'monte_carlo_critical_value',
    'multiplier_statistics', 'multiplier_statistics_from_draws', 'nb_loglik',
    'nb_surveillance_glr', 'nb_variance', 'page_cusum',
    'paired_binary_difference_interval', 'published_benchmark_statistics',
    'sampler_backend_available', 'score_scan',
    'simulate_deaths_with_dispersion_path', 'simulate_nb_deaths',
    'validate_delay_pmf',
]
