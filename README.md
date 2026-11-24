# OCFR

OCFR detects changes in delay-adjusted epidemic case-fatality risk. It fits a
negative-binomial convolution model and scans possible cohort change dates.
One Gaussian multiplier bank calibrates cumulative first-crossing boundaries
across the planned analyses for the practical Section 2 procedure. The
simulation comparison also applies an oracle calibration to the same OCFR
scan and compares both calibrations with NB-GLR and Page CUSUM. The detector
uses a supplied delay distribution that links each case cohort to later deaths.

The 22-look simulation design uses 1,999 Gaussian paths. Real applications
and the portable online API use 10,000 paths to reduce variation from the
realized numerical bank. Both use the same boundary construction.

## Contents

| File | Purpose |
| --- | --- |
| `config.py` | Data-generating scenarios, method settings, seeds and study plans. |
| `methods.py` | Portable detectors, estimators, intervals, samplers and C bindings. |
| `simulation.py` | Parallel experiments, atomic caches, resume and scoped reset. |
| `analysis.py` | Simulation summaries, interval calibration and publication figures. |
| `real_data.py` | Public-data download, preprocessing, monitoring and application figures. |
| `csrc/ocfr_kernels.c` | C acceleration and matched random sampling. |

The repository contains code, tests and essential documentation. Downloaded
data, results, figures, caches, compiled libraries and manuscripts are ignored.

## Install

Use Python 3.10 or newer, preferably in a virtual environment:

```bash
python -m pip install -e ".[test]"
```

C is the default backend. With GCC, Clang, MSVC or Zig available:

```bash
python -c "import methods; print(methods.build_native(force=True))"
python -c "import methods; print(methods.assert_backend_parity())"
```

The Python implementation needs no compiler. Select it explicitly with
`--backend python` when a C compiler or a loadable native library is unavailable.
The v2 Python multiplier bank uses vectorized PCG32/Box--Muller generation;
temporary arrays are bounded by chunks, and count draws retain the scalar
reference stream. Tests check integer draws,
stream progression and numerical agreement of the score and boundary paths.

## Reproduce the manuscript

Run the simulation study and historical surveillance applications from a
fresh checkout after installation. Use separate output directories for
full studies and short workflow checks. The programs keep downloaded data,
results, figures, and caches outside version control.

### Simulation study

```bash
ocfr-simulate --stage paper --mode full --jobs 8 --output outputs/paper
```

This command trains the fixed thresholds, audits false alarms on independent
null replications, compares detection and localization, runs the sensitivity
and boundary studies, and evaluates post-alarm estimation. The paper's
simulation figures and their tables are written under `outputs/paper`.
Figure 1 uses illustrative values. Figs. 2?6 and the simulation tables use the
newly computed results.

The study uses 10,000 null-training and 5,000 null-audit replications for each
of three incidence families. Each of 41 reported evaluation conditions uses
1,000 replications. Interval cutoffs are fitted separately from detection
thresholds. Their validation uses 1,000 new replications for each of the 16
reference alternatives. The workflow saves cell-level and replication-level
evidence alongside each stage manifest.

Practical OCFR uses data-prefix Gaussian boundaries. Oracle OCFR uses the
same fitted scan and a fixed cutoff trained under the known null generator
for each incidence family. The report-day NB-GLR and Page CUSUM benchmarks
receive their own cutoffs from the same training design. First alarms on or
after the true change contribute to detection probability. All requested
replications stay in its denominator. Localization accuracy is summarized
separately, and the study applies both estimators in Fig. 5 to the same
first-alarm data and selected date.

The paper command reproduces the recorded backend choices. Practical
simulation scores and benchmark training use Python. Oracle cutoff training
and the designated raw-score replays use the C kernels. The simulation uses
1,999 shared Gaussian paths. Repeat the command with the same output
directory to verify completed stages and resume. Use a fresh directory for
an independent recomputation or after changing the code. The workload is
suitable for multiple workers. `--jobs -1` uses all available CPUs.

To check the command-line workflow with ten replications per condition, run
the smaller protocol below. Its empirical thresholds can be infinite
because ten training replications cannot resolve the 5% Monte Carlo rank.

```bash
ocfr-simulate --stage calibration --mode demo --jobs 2 --output outputs/demo
ocfr-simulate --stage audit --mode demo --jobs 2 --output outputs/demo
ocfr-simulate --stage evaluation --mode demo --jobs 2 --output outputs/demo
ocfr-analyze --protocol compact-v2 --input outputs/demo
```

Use `ocfr-simulate --help` for running an individual stage, selecting scenarios,
and resetting its scoped cache. The earlier study protocols remain accessible
through their explicit names.

### Historical surveillance applications

The current protocol replays WHO COVID-19 surveillance for Japan, South Korea,
and the United Kingdom and SINAN pandemic influenza surveillance from Brazil.
The two data sources are the [WHO dashboard][who] and [Brazil's SINAN
Influenza archive][sinan]. The WHO data use case and death report dates.
The SINAN data use symptom-onset and death-occurrence dates for hospitalized,
laboratory-confirmed cases. The replay indexes finalized or reconciled records
by these dates.

Rebuild the daily counts from current public downloads, check them against
the paper inputs, and generate Figs. 7–8:

```bash
ocfr-real-data --stage all --data data/paper --output outputs/surveillance --require-manuscript-inputs
```

The public releases reproduced the archived counts when verified for this
revision. Data providers can revise their historical files. If the check
fails, extract the manuscript's frozen aggregate input archive into a fresh
data directory, with both CSV files under `processed/`, and rerun the
command with that directory. The command uses these files offline. Use
`--stage fetch` to generate the upstream provenance and record-flow counts
separately. The two files required for an offline replay are
`processed/who_covid_processed.csv` and
`processed/brazil_influenza_daily.csv`.

```bash
ocfr-real-data --stage benchmarks --data data/paper --output outputs/surveillance --workers 8 --require-manuscript-inputs
ocfr-real-data --stage audit --data data/paper --output outputs/surveillance --workers 8 --require-manuscript-inputs
```

The main replay writes the four OCFR series, delay and calendar
sensitivity analyses, residual summaries, and Figs. 7?8. The benchmark command
runs NB-GLR and Page CUSUM with look-specific calibration from each observed
prefix, retaining its first-alarm dates and the independent null-path
diagnostics. The audit command runs the separate OCFR null audits conditional
on the archived case series. Defaults retain 10,000 paths for application
boundaries, 10,000 training paths and 10,000 independent validation paths
for benchmark cutoffs, and 1,000 replications per OCFR null audit.

Risk ratios in the applications use nominal Wald intervals calculated with
the selected change date and first alarm date held fixed. A later statistic
below its boundary leaves the earlier alarm in the trace. See the CLI help
for backend choices, short-run counts and the earlier dengue example.

[who]: https://data.who.int/dashboards/covid19/data
[sinan]: https://dadosabertos.saude.gov.br/dataset/srag-2009-2012

## Use the detector independently

Copy `methods.py` and `csrc/` into another project, or import the installed
module. Supply a delay probability vector and a declared look schedule:

```python
from methods import OnlineGaussianProcessScore, gamma_delay_pmf, make_sampler

detector = OnlineGaussianProcessScore(
    gamma_delay_pmf(14, shape=4, maximum_lag=45),
    horizon=182, planned_looks=tuple(range(35, 183, 7)),
    n_multiplier=10_000, rng=make_sampler(2026111404, backend="c"),
    backend="c",
)
for cases_today, deaths_today in observations:
    result = detector.update(cases_today, deaths_today)
    if result.alarm:
        break
```

The detector stops at its first strict boundary crossing. Its numerical
reference-crossing fraction is a diagnostic of the calibration bank; actual
episode error is evaluated with independent null episodes.

## Tests

```bash
python -m pytest
```

Tests cover boundary ranks and support removal, prefix-only decisions,
matched C/Python calculations, sampler state, protocol contracts, cache
integrity and deterministic parallel/resume behavior. Native tests require
a working C backend and can build one; do not run those tests on a host that
blocks native compilation or loading. For a portable-only check without a
native build, run:

```bash
python -m pytest --ignore=tests/test_compact_native_multiplier.py --ignore=tests/test_compact_surveillance_native.py -k "not numerically_matched and not zero_lag_weekly_exposure_has_native_parity and not sensitivity_generator_native_reference_parity"
```

This deliberately deselects native parity checks and does not validate the C
backend. CI retains the separate native build and parity check on supported
runners. See `--help` for each of the three installed command entry points.

Project: <https://github.com/BobZhangHT/OCFR>.
