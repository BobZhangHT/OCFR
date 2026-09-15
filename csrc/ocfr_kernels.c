#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef _WIN32
#define OCFR_API __declspec(dllexport)
#else
#define OCFR_API
#endif

OCFR_API int ocfr_exposure(
    int n, int delay_len, const double *cases, const double *delay,
    int start, int end, double *out
) {
    int s, u;
    if (!cases || !delay || !out || n < 0 || delay_len <= 0 ||
        start < 0 || end < start || end > n) return 1;
    for (s = 0; s < n; ++s) out[s] = 0.0;
    for (u = start; u < end; ++u) {
        int max_lag = delay_len;
        if (u + max_lag >= n) max_lag = n - u - 1;
        for (int lag = 1; lag <= max_lag; ++lag) {
            out[u + lag] += cases[u] * delay[lag - 1];
        }
    }
    return 0;
}

OCFR_API double ocfr_nb_loglik(
    int n, const double *y, const double *mu, double phi
) {
    double total = 0.0;
    if (!y || !mu || n < 0 || !(phi > 0.0)) return NAN;
    for (int i = 0; i < n; ++i) {
        const double m = mu[i] > 1e-12 ? mu[i] : 1e-12;
        if (phi >= 1e7) {
            total += y[i] * log(m) - m - lgamma(y[i] + 1.0);
        } else {
            total += lgamma(y[i] + phi) - lgamma(phi) - lgamma(y[i] + 1.0)
                + phi * log(phi) + y[i] * log(m)
                - (y[i] + phi) * log(phi + m);
        }
    }
    return total;
}

OCFR_API double ocfr_score_from_matrix(
    int n_candidates, int n_obs, const double *bmat,
    const double *residual_score, const double *inv_variance,
    const double *x, double ipp, double *signed_score, int *best_index,
    int *eligible_count
) {
    double best_q = 0.0;
    double best_u = 0.0;
    int best_k = -1;
    int eligible = 0;
    if (!bmat || !residual_score || !inv_variance || !x || !best_index || !eligible_count ||
        n_candidates < 0 || n_obs < 0 || !(ipp > 0.0)) return NAN;
    for (int k = 0; k < n_candidates; ++k) {
        const double *b = bmat + (size_t)k * (size_t)n_obs;
        double u = 0.0, ibb = 0.0, ibp = 0.0;
        for (int s = 0; s < n_obs; ++s) {
            u += b[s] * residual_score[s];
            ibb += b[s] * b[s] * inv_variance[s];
            ibp += b[s] * x[s] * inv_variance[s];
        }
        const double j = ibb - ibp * ibp / ipp;
        const double tol = 64.0 * 2.2204460492503131e-16 * (ibb > 1.0 ? ibb : 1.0);
        if (j <= tol) continue;
        ++eligible;
        const double q = u * u / j;
        if (q > best_q + 1e-12 || (fabs(q - best_q) <= 1e-12 && (best_k < 0 || k < best_k))) {
            best_q = q;
            best_u = u;
            best_k = k;
        }
    }
    *best_index = best_k;
    *eligible_count = eligible;
    if (signed_score) *signed_score = best_u;
    return best_q;
}

OCFR_API int ocfr_candidate_matrix(
    int n, int delay_len, const double *cases, const double *delay,
    int n_candidates, const int *taus, double *out
) {
    if (!cases || !delay || !taus || !out || n < 0 || delay_len <= 0 || n_candidates < 0)
        return 1;
    for (int k = 0; k < n_candidates; ++k) {
        const int tau = taus[k];
        double *row = out + (size_t)k * (size_t)n;
        if (tau < 0 || tau > n) return 2;
        for (int s = 0; s < n; ++s) row[s] = 0.0;
        for (int u = tau; u < n; ++u) {
            int max_lag = delay_len;
            if (u + max_lag >= n) max_lag = n - u - 1;
            for (int lag = 1; lag <= max_lag; ++lag)
                row[u + lag] += cases[u] * delay[lag - 1];
        }
    }
    return 0;
}

OCFR_API int ocfr_finite_support_score(
    int n, int delay_len, const double *cases, const double *delay,
    const double *x, const double *residual_score,
    const double *inv_variance, double ipp, int tau_start,
    double *q_out, double *u_out, unsigned char *eligible_out
) {
    double *suffix_u;
    double *suffix_i;
    if (!cases || !delay || !x || !residual_score || !inv_variance ||
        !q_out || !u_out || !eligible_out || n < 0 || delay_len <= 0 ||
        tau_start < 0 || tau_start > n || !(ipp > 0.0)) return 1;
    suffix_u = (double *)calloc((size_t)n + 1U, sizeof(double));
    suffix_i = (double *)calloc((size_t)n + 1U, sizeof(double));
    if (!suffix_u || !suffix_i) {
        free(suffix_u);
        free(suffix_i);
        return 2;
    }
    for (int s = n - 1; s >= 0; --s) {
        suffix_u[s] = suffix_u[s + 1] + x[s] * residual_score[s];
        suffix_i[s] = suffix_i[s + 1] + x[s] * x[s] * inv_variance[s];
    }
    for (int tau = tau_start; tau < n; ++tau) {
        const int index = tau - tau_start;
        int suffix_start = tau + delay_len;
        double u, ibb, ibp;
        if (suffix_start > n) suffix_start = n;
        u = suffix_u[suffix_start];
        ibb = suffix_i[suffix_start];
        ibp = suffix_i[suffix_start];
        for (int s = tau + 1; s < suffix_start; ++s) {
            double b = 0.0;
            for (int cohort = tau; cohort < s; ++cohort)
                b += cases[cohort] * delay[s - cohort - 1];
            u += b * residual_score[s];
            ibb += b * b * inv_variance[s];
            ibp += b * x[s] * inv_variance[s];
        }
        const double iaa = ipp - 2.0 * ibp + ibb;
        const double cross = ibp - ibb;
        const double determinant = iaa * ibb - cross * cross;
        const double scale = fmax(fmax(iaa, ibb), 1.0);
        const double j = ibb - ibp * ibp / ipp;
        const double tolerance =
            64.0 * 2.2204460492503131e-16 * fmax(ibb, 1.0);
        q_out[index] = -INFINITY;
        u_out[index] = 0.0;
        eligible_out[index] = 0U;
        if (!(iaa > 0.0 && ibb > 0.0 &&
              determinant > 2.2204460492503131e-16 * scale * scale &&
              j > tolerance))
            continue;
        eligible_out[index] = 1U;
        u_out[index] = u;
        q_out[index] = u * u / j;
    }
    free(suffix_u);
    free(suffix_i);
    return 0;
}

/*
 * Build the total convolution exposure and all post-change exposure columns
 * used by the one-change Gaussian-process score detector.
 *
 * `post` is row-major with shape n x (n - 1). Column k corresponds to the
 * zero-based cohort candidate tau = k + 1. The reverse recurrence deliberately
 * follows the same cohort summation order as the NumPy reference.
 */
OCFR_API int ocfr_gp_exposure_components(
    int n, int delay_len, const double *cases, const double *delay,
    double *total, double *post
) {
    const int n_candidates = n - 1;
    if (!cases || !delay || !total || !post || n < 3 || delay_len <= 0)
        return 1;
    for (int s = 0; s < n; ++s) {
        double running = 0.0;
        double *row = post + (size_t)s * (size_t)n_candidates;
        for (int tau = n - 1; tau >= 1; --tau) {
            const int lag = s - tau;
            if (lag >= 1 && lag <= delay_len)
                running += cases[tau] * delay[lag - 1];
            row[tau - 1] = running;
        }
        if (s >= 1 && s <= delay_len)
            running += cases[0] * delay[s - 1];
        total[s] = running;
    }
    return 0;
}

/*
 * Compute every field needed by the current maximal efficient-score scan.
 * Arrays `post` and `influence` are row-major n x (n - 1). Invalid candidate
 * columns are returned with NaN scalar diagnostics and zero influence.
 * Candidate selection exactly mirrors ocfr.single_change_gp.score_scan.
 */
OCFR_API int ocfr_gp_score_fields(
    int n, const double *post, const double *exposure, const double *deaths,
    const double *mean, const double *variance, int observation_start,
    int candidate_start, int use_lindeberg_gate, double *information,
    double *max_leverage, double *z_out, double *q_out, double *influence,
    unsigned char *valid_out, int *eligible_count
) {
    const int n_candidates = n - 1;
    double ipp = 0.0;
    double max_information = 0.0;
    int n_positive = 0;
    int eligible = 0;
    if (!post || !exposure || !deaths || !mean || !variance ||
        !information || !max_leverage || !z_out || !q_out || !influence ||
        !valid_out || !eligible_count || n < 3 || observation_start < 0 ||
        observation_start >= n || candidate_start < 1 || candidate_start >= n)
        return 1;

    for (int s = observation_start; s < n; ++s) {
        const double v = variance[s];
        if (!(v > 0.0) || !isfinite(v)) return 2;
        ipp += exposure[s] * exposure[s] / v;
        if (exposure[s] > 1e-12) ++n_positive;
    }
    if (!(ipp > 1e-12) || !isfinite(ipp)) return 3;
    if (n_positive < 1) n_positive = 1;

    for (int k = 0; k < n_candidates; ++k) {
        double cross = 0.0;
        double j = 0.0;
        double leverage_numer = 0.0;
        double u = 0.0;
        for (int s = observation_start; s < n; ++s) {
            const double b =
                post[(size_t)s * (size_t)n_candidates + (size_t)k];
            cross += b * exposure[s] / variance[s];
        }
        const double projection = cross / ipp;
        for (int s = observation_start; s < n; ++s) {
            const double b =
                post[(size_t)s * (size_t)n_candidates + (size_t)k];
            const double g = b - exposure[s] * projection;
            const double share = g * g / variance[s];
            j += share;
            if (share > leverage_numer) leverage_numer = share;
            u += g * (deaths[s] - mean[s]) / variance[s];
        }
        information[k] = j;
        max_leverage[k] =
            (j > 0.0 && isfinite(j)) ? leverage_numer / j : NAN;
        z_out[k] = (j > 0.0 && isfinite(j)) ? u / sqrt(j) : NAN;
        q_out[k] =
            (j > 0.0 && isfinite(j)) ? z_out[k] * z_out[k] : NAN;
        valid_out[k] = 0U;
        if (isfinite(j) && j > max_information) max_information = j;
    }

    if (max_information < 1.0) max_information = 1.0;
    for (int k = 0; k < n_candidates; ++k) {
        const int tau = k + 1;
        int valid =
            isfinite(information[k]) &&
            information[k] > 1e-12 * max_information &&
            isfinite(max_leverage[k]) &&
            tau >= candidate_start;
        if (valid && use_lindeberg_gate) {
            const double limit = 1.0 / sqrt((double)n_positive);
            valid = max_leverage[k] <= limit;
        }
        if (!valid) {
            z_out[k] = NAN;
            q_out[k] = NAN;
            for (int s = 0; s < n; ++s)
                influence[(size_t)s * (size_t)n_candidates + (size_t)k] = 0.0;
            continue;
        }

        const double root_j = sqrt(information[k]);
        double cross = 0.0;
        for (int s = observation_start; s < n; ++s) {
            const double b =
                post[(size_t)s * (size_t)n_candidates + (size_t)k];
            cross += b * exposure[s] / variance[s];
        }
        const double projection = cross / ipp;
        for (int s = 0; s < observation_start; ++s)
            influence[(size_t)s * (size_t)n_candidates + (size_t)k] = 0.0;
        for (int s = observation_start; s < n; ++s) {
            const double b =
                post[(size_t)s * (size_t)n_candidates + (size_t)k];
            const double g = b - exposure[s] * projection;
            influence[(size_t)s * (size_t)n_candidates + (size_t)k] =
                g / sqrt(variance[s]) / root_j;
        }
        valid_out[k] = 1U;
        ++eligible;
    }
    *eligible_count = eligible;
    return 0;
}

/*
 * Transform supplied standard-normal multiplier draws into the conditional
 * Gaussian-process supremum paths used for calibration.  Both inputs are
 * row-major: `multipliers` has shape B x n and `influence` has shape n x k.
 * Keeping the random draws outside this kernel makes the Python and C
 * implementations directly comparable for a fixed seed.
 */
OCFR_API int ocfr_multiplier_maxima(
    int n_multiplier, int n_obs, int n_candidates,
    const double *multipliers, const double *influence, double *out
) {
    if (!multipliers || !influence || !out || n_multiplier <= 0 ||
        n_obs <= 0 || n_candidates <= 0)
        return 1;
    for (int b = 0; b < n_multiplier; ++b) {
        double maximum = 0.0;
        const double *xi = multipliers + (size_t)b * (size_t)n_obs;
        for (int k = 0; k < n_candidates; ++k) {
            double process = 0.0;
            for (int s = 0; s < n_obs; ++s) {
                process +=
                    xi[s] * influence[(size_t)s * (size_t)n_candidates +
                                      (size_t)k];
            }
            const double squared = process * process;
            if (k == 0 || squared > maximum) maximum = squared;
        }
        out[b] = maximum;
    }
    return 0;
}

/*
 * Reproducible random-number primitives
 * --------------------------------------
 *
 * The simulation code uses a small, self-contained PCG32 generator instead
 * of the C library ``rand`` function.  The latter is implementation-defined
 * and would make a release depend on the host C runtime.  A sampler call is
 * identified by ``(seed, stream)``; callers can advance ``stream`` between
 * calls to obtain independent deterministic substreams.  The Python
 * reference in ``ocfr/samplers.py`` implements the same state transition and
 * rejection algorithms.  The API intentionally exposes draws as arrays so
 * the two implementations can be compared byte-for-byte for fixed inputs.
 *
 * These routines are not intended to reproduce NumPy's Generator bit stream.
 * They define OCFR's own reproducible reference stream, while the generated
 * variables have the requested target distributions up to floating-point
 * arithmetic and the documented finite-domain checks.
 */

#define OCFR_PCG_MULT 6364136223846793005ULL
#define OCFR_UINT32_SCALE 4294967296.0
#define OCFR_PI 3.141592653589793238462643383279502884

typedef struct {
    uint64_t state;
    uint64_t increment;
} ocfr_pcg32_state;

static uint32_t ocfr_pcg32_next(ocfr_pcg32_state *rng) {
    const uint64_t oldstate = rng->state;
    const uint32_t xorshifted =
        (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    const uint32_t rot = (uint32_t)(oldstate >> 59u);
    rng->state = oldstate * OCFR_PCG_MULT + rng->increment;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

static void ocfr_pcg32_seed(
    ocfr_pcg32_state *rng, uint64_t seed, uint64_t stream
) {
    rng->state = 0ULL;
    rng->increment = (stream << 1u) | 1ULL;
    (void)ocfr_pcg32_next(rng);
    rng->state += seed;
    (void)ocfr_pcg32_next(rng);
}

static double ocfr_uniform_open(ocfr_pcg32_state *rng) {
    /* The half-unit offset prevents log(0) in the Box--Muller transform. */
    return ((double)ocfr_pcg32_next(rng) + 0.5) / OCFR_UINT32_SCALE;
}

static uint64_t ocfr_uint64_next(ocfr_pcg32_state *rng) {
    const uint64_t high = (uint64_t)ocfr_pcg32_next(rng);
    const uint64_t low = (uint64_t)ocfr_pcg32_next(rng);
    return (high << 32u) | low;
}

static double ocfr_normal_standard(ocfr_pcg32_state *rng) {
    const double u1 = ocfr_uniform_open(rng);
    const double u2 = ocfr_uniform_open(rng);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * OCFR_PI * u2);
}

static int ocfr_valid_length(int n) {
    return n >= 0;
}

static int64_t ocfr_poisson_one(
    ocfr_pcg32_state *rng, double lambda
) {
    if (!isfinite(lambda) || lambda < 0.0 || lambda > 1.0e12)
        return INT64_MIN;
    if (lambda == 0.0)
        return 0;

    /* Knuth inversion is stable and inexpensive for the small means that
       occur in sparse-death scenarios. */
    if (lambda < 30.0) {
        const double limit = exp(-lambda);
        double product = 1.0;
        int64_t count = 0;
        do {
            ++count;
            product *= ocfr_uniform_open(rng);
        } while (product > limit);
        return count - 1;
    }

    /* PTRS transformed rejection (Hoermann, 1993), avoiding a normal
       approximation while remaining fast for the large epidemic means. */
    {
        const double sqrt_lambda = sqrt(lambda);
        const double b = 0.931 + 2.53 * sqrt_lambda;
        const double a = -0.059 + 0.02483 * b;
        const double inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        const double vr = 0.9277 - 3.6224 / (b - 2.0);
        const double log_lambda = log(lambda);
        for (;;) {
            const double u = ocfr_uniform_open(rng) - 0.5;
            const double v = ocfr_uniform_open(rng);
            const double us = 0.5 - fabs(u);
            const double candidate_real =
                floor((2.0 * a / us + b) * u + lambda + 0.43);
            const int64_t candidate = (int64_t)candidate_real;
            if (candidate < 0)
                continue;
            if (us >= 0.07 && v <= vr)
                return candidate;
            if (us < 0.013 && v > us)
                continue;
            if (log(v * inv_alpha / (a / (us * us) + b)) <=
                -lambda + (double)candidate * log_lambda -
                    lgamma((double)candidate + 1.0))
                return candidate;
        }
    }
}

static double ocfr_gamma_one(
    ocfr_pcg32_state *rng, double shape, double scale
) {
    if (!isfinite(shape) || !isfinite(scale) || shape <= 0.0 || scale <= 0.0)
        return NAN;
    if (shape < 1.0) {
        const double boosted = ocfr_gamma_one(rng, shape + 1.0, 1.0);
        return scale * boosted * pow(ocfr_uniform_open(rng), 1.0 / shape);
    }
    {
        const double d = shape - 1.0 / 3.0;
        const double c = 1.0 / sqrt(9.0 * d);
        for (;;) {
            const double x = ocfr_normal_standard(rng);
            const double one_plus = 1.0 + c * x;
            if (one_plus <= 0.0)
                continue;
            {
                const double v = one_plus * one_plus * one_plus;
                const double u = ocfr_uniform_open(rng);
                if (u < 1.0 - 0.0331 * x * x * x * x ||
                    log(u) < 0.5 * x * x + d * (1.0 - v + log(v)))
                    return scale * d * v;
            }
        }
    }
}

static int64_t ocfr_negative_binomial_one(
    ocfr_pcg32_state *rng, double shape, double probability
) {
    if (!isfinite(shape) || shape <= 0.0 ||
        !isfinite(probability) || probability < 0.0 || probability > 1.0)
        return INT64_MIN;
    if (probability == 1.0)
        return 0;
    if (probability == 0.0)
        return INT64_MIN;
    /* At this boundary the likelihood and the sampler both use the Poisson
       interpretation, avoiding an ill-conditioned nearly-degenerate gamma. */
    if (shape >= 1.0e7)
        return ocfr_poisson_one(rng, shape * (1.0 - probability) / probability);
    {
        const double mean_gamma =
            shape * (1.0 - probability) / probability;
        const double intensity = ocfr_gamma_one(rng, shape, mean_gamma / shape);
        if (!isfinite(intensity))
            return INT64_MIN;
        return ocfr_poisson_one(rng, intensity);
    }
}

OCFR_API int ocfr_rng_uniform(
    uint64_t seed, uint64_t stream, int n, double *out
) {
    ocfr_pcg32_state rng;
    if (!out || !ocfr_valid_length(n)) return 1;
    ocfr_pcg32_seed(&rng, seed, stream);
    for (int i = 0; i < n; ++i) out[i] = ocfr_uniform_open(&rng);
    return 0;
}

OCFR_API int ocfr_rng_normal(
    uint64_t seed, uint64_t stream, int n, double *out
) {
    ocfr_pcg32_state rng;
    if (!out || !ocfr_valid_length(n)) return 1;
    ocfr_pcg32_seed(&rng, seed, stream);
    for (int i = 0; i < n; ++i) out[i] = ocfr_normal_standard(&rng);
    return 0;
}

OCFR_API int ocfr_rng_integers(
    uint64_t seed, uint64_t stream, int n,
    uint64_t low, uint64_t high, uint64_t *out
) {
    ocfr_pcg32_state rng;
    uint64_t range;
    uint64_t limit;
    if (!out || !ocfr_valid_length(n) || high <= low) return 1;
    range = high - low;
    /* Reject the incomplete top interval so bounded integers are unbiased. */
    limit = UINT64_MAX - (UINT64_MAX % range);
    ocfr_pcg32_seed(&rng, seed, stream);
    for (int i = 0; i < n; ++i) {
        uint64_t draw;
        do { draw = ocfr_uint64_next(&rng); } while (draw >= limit);
        out[i] = low + draw % range;
    }
    return 0;
}

OCFR_API int ocfr_rng_poisson(
    uint64_t seed, uint64_t stream, int n, const double *mean,
    int64_t *out
) {
    ocfr_pcg32_state rng;
    if (!mean || !out || !ocfr_valid_length(n)) return 1;
    ocfr_pcg32_seed(&rng, seed, stream);
    for (int i = 0; i < n; ++i) {
        out[i] = ocfr_poisson_one(&rng, mean[i]);
        if (out[i] == INT64_MIN) return 2;
    }
    return 0;
}

OCFR_API int ocfr_rng_negative_binomial(
    uint64_t seed, uint64_t stream, int n, const double *shape,
    const double *probability, int64_t *out
) {
    ocfr_pcg32_state rng;
    if (!shape || !probability || !out || !ocfr_valid_length(n)) return 1;
    ocfr_pcg32_seed(&rng, seed, stream);
    for (int i = 0; i < n; ++i) {
        out[i] = ocfr_negative_binomial_one(
            &rng, shape[i], probability[i]
        );
        if (out[i] == INT64_MIN) return 2;
    }
    return 0;
}

OCFR_API int ocfr_rng_binomial(
    uint64_t seed, uint64_t stream, int n, const int64_t *trials,
    const double *probability, int64_t *out
) {
    ocfr_pcg32_state rng;
    if (!trials || !probability || !out || !ocfr_valid_length(n)) return 1;
    ocfr_pcg32_seed(&rng, seed, stream);
    for (int i = 0; i < n; ++i) {
        const int64_t size = trials[i];
        const double p = probability[i];
        int64_t successes = 0;
        if (size < 0 || !isfinite(p) || p < 0.0 || p > 1.0) return 2;
        if (p == 1.0) {
            out[i] = size;
            continue;
        }
        if (p == 0.0 || size == 0) {
            out[i] = 0;
            continue;
        }
        if (size > 100000000) return 3;
        for (int64_t trial = 0; trial < size; ++trial)
            if (ocfr_uniform_open(&rng) < p) ++successes;
        out[i] = successes;
    }
    return 0;
}
