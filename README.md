# OCFR

OCFR detects changes in delay-adjusted epidemic case-fatality risk. It fits a
negative-binomial convolution model and scans possible cohort change dates.
One Gaussian multiplier bank calibrates cumulative first-crossing boundaries
across the planned analyses for the practical Section 2 procedure. The
simulation comparison also applies an oracle calibration to the same OCFR
scan and compares both calibrations with NB-GLR and Page CUSUM. The detector
uses a supplied confirmation-to-death delay law.

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

## Run the simulations

The default `compact-v2` protocol uses steady, sparse and epidemic-wave case
series. Demo and full have identical scientific settings and seeds; their
replication counts are 10 and 1,000. Run the three stages in order:

```bash
ocfr-simulate --stage calibration --mode demo --jobs 4 --output outputs/demo
ocfr-simulate --stage audit --mode demo --jobs 4 --output outputs/demo
ocfr-simulate --stage evaluation --mode demo --jobs 4 --output outputs/demo
ocfr-analyze --protocol compact-v2 --input outputs/demo
```

Use `--mode full` and a separate output directory for full runs. Ten training
episodes cannot resolve a 5% empirical cutoff under the conservative
Monte Carlo rank rule, so demo empirical calibrations can have infinite
cutoffs. Demo checks the workflow. Full and release-study runs assess
statistical performance.

The named paper replication plan uses 10,000 independent null-training
episodes and 5,000 independent null-audit episodes per incidence family, and
1,000 episodes per evaluation condition. The base analysis is an intermediate
report. The sensitivity and paired-calibration stages build on the completed
base study:

```bash
ocfr-simulate --protocol compact-v2-release --stage calibration --mode full --jobs 8 --output outputs/paper
ocfr-simulate --protocol compact-v2-release --stage audit --mode full --jobs 8 --output outputs/paper
ocfr-simulate --protocol compact-v2-release --stage evaluation --mode full --jobs 8 --output outputs/paper
ocfr-analyze --protocol compact-v2 --input outputs/paper
```

Add `--backend python` to the base simulation commands for the Python backend.
The practical OCFR boundaries use the available data prefix and the common
Gaussian bank. `OCFR-oracle`, NB-GLR and Page CUSUM each use a fixed
cutoff derived separately for steady, sparse and wave incidence from 10,000
null-training episodes per family in the full paper study. Each cutoff is the
9,501st smallest episode maximum across the planned looks. Evaluation uses
strict crossings and independent audit and alternative episodes. Reports
show achieved false-alarm rates as well as power. Useful detection counts a
first alarm on or after the true change. Premature and absent alarms remain
in the denominator.

On each episode, practical OCFR and `OCFR-oracle` use the same
observations, planned looks, fitted score path and candidate cohort dates.
Their first alarms can differ because one uses prefix-specific Gaussian
boundaries and the other uses the family's fixed cutoff. The audit, detection,
localization and sensitivity comparisons use both calibrations. Post-alarm
estimation in Figure 5 uses the practical OCFR first alarm.

Repeating a command resumes verified per-episode caches. `--jobs N` selects
parallel workers. `--reset selected` recomputes the selected stage/scenarios;
`--reset derived` rebuilds tables from intact caches. `--reset all` resets the
selected scope within its scientific namespace. Other source/configuration
namespaces remain available. Use `--scenarios NAME ...` for selected detection
conditions. Manifests bind settings, source hashes, seeds and artifact hashes.

Explicit `--protocol compact-v1` retains historical reproduction. Its older
method labels and boundary rules belong to that archived protocol.

### Denser delay grid and legacy sensitivity output

After the base study completes, add the ten intermediate sensitivity conditions
without rerunning detection and estimation for the other figures:

```bash
ocfr-simulate --stage sensitivity-extension --base-study outputs/paper --output outputs/sensitivity --jobs 8
```

The extension fills delay means 12, 13, 15 and 16 days under constant risk and
RR=1.5. The frozen extension also retains two historical ascertainment cells.
Its standalone analysis command, `ocfr-analyze --protocol compact-v2 --input
outputs/paper --sensitivity-extension outputs/sensitivity`, reproduces the
**legacy sensitivity figure only**, including the old ascertainment panel.
The paired-calibration workflow below uses the retained delay conditions
and the separate dispersion study for the paper's Figure 6.

Replication count and backend inherit the parent; full mode runs 1,000 episodes
per new condition. `--mode demo` runs ten, with the same numerical settings.
The extension has its own verified caches and scoped `--reset` handling and
keeps the original comparator cutoffs. Parent result identities are retained.
For a parent produced with an earlier source version, supply
`--base-code PATH_TO_FROZEN_CORE`; numerical compatibility is checked against
those retained producer files before using the cutoffs.

### Dispersion sensitivity in Figure 6D

Run the separate eight-cell study after the completed parent and ten-cell
extension. Figure 6D uses
useful power at RR=1.5 over constant NB dispersions 5, 10, 20 and 40 on a
log2 axis. Null alarms at all four values remain mandatory reproducibility
diagnostics; the paper reports their interpretation inline and contains no
separate null-audit figure or appendix section.

```bash
ocfr-simulate --stage dispersion-sensitivity --base-study outputs/paper --output outputs/dispersion --jobs 8
ocfr-analyze --protocol compact-v2 --input outputs/paper --sensitivity-extension outputs/sensitivity --dispersion-study outputs/dispersion
```

Supply `--base-code PATH_TO_FROZEN_CORE` to both commands when the parent has
a different retained source version. Mode and backend inherit the parent;
`--mode demo` uses 10 episodes per cell and full uses 1,000, with identical
scientific settings. The independent master seed is 2026111604. Wave incidence,
H=182, change day 92, RR=1 or 1.5, and correct true/assumed Gamma delay mean
14 are fixed. Both RR levels deliberately use the same computational
alternative split and shared stream groups to pair cases. Reporting classifies
RR=1 as null; this allocation does not turn null episodes into power observations.
All methods retain their original fitting algorithms. The empirical calibrations
use their frozen parent cutoffs trained at dispersion 10 across all four
dispersion settings. The separate null results show achieved error at each
dispersion. The paper reports 41 evaluation conditions and
41,000 episodes: 33 retained base/extension conditions plus these eight cells,
each with 1,000 episodes. Benchmark training and independent null-audit banks
are separate. The frozen historical runners additionally retain four
ascertainment conditions, excluded from the current paper reports.

`dispersion-sensitivity/` has its own source/config/parent-bound cache namespace,
manifest and scoped `--reset` behavior. It cannot overwrite the parent or reuse
historical private validation caches. Analysis validates both producer identities,
exports `dispersion_null_audit.csv`, dispersion summary/contrast/failure tables,
Figure 6 and the null-audit plot under `outputs/sensitivity/analysis/`. The
historically named `CompactFigS2_dispersion_null` output is a reproducibility
diagnostic, not a manuscript figure, and must not be promoted to the paper.
The historical base and ten-cell extension retain
their original ascertainment cells and legacy plotting behavior for compatibility.
With `--dispersion-study`, those cells are not plotted in Figure 6 or promoted
to a supplementary ascertainment figure. Numerical source compatibility is
checked against the retained parent; historical manifests and scientific digests
are not relabeled as products of the new reproduction code.

### Paired detection comparisons in Figures 2–4 and 6

After the full `compact-v2-release` base study and both sensitivity stages
complete, run the paired stage and analyze its output separately:

```bash
ocfr-simulate --stage paired-detection --base-study outputs/paper --extension-study outputs/sensitivity --dispersion-study outputs/dispersion --output outputs/paired --jobs 8
ocfr-analyze --paired-detection --input outputs/paired
```

The paired stage requires the native C backend for replaying raw OCFR scores
in the reference training bank. It verifies the frozen benchmark cutoffs and
uses those same training episodes to calibrate `OCFR-oracle`. It pairs the
rules on the retained audit, evaluation and sensitivity episodes. Training is
independent of these episodes. The paired output directory holds method-level
results and the cutoff table at `calibration/thresholds.csv`. Analysis writes
tables and figures under `outputs/paired/analysis/`.

## Post-alarm estimation

Postprocess a completed full study to fit pre-change CFR, post-change CFR
and their risk ratio at each newly selected first-alarm prefix:

```bash
ocfr-simulate --protocol compact-v2-release --stage estimation --mode full --jobs 8 --output outputs/paper
ocfr-simulate --protocol compact-v2-release --stage estimation-reference --mode full --jobs 8 --output outputs/paper
ocfr-simulate --protocol compact-v2-release --stage estimation-stabilized --mode full --jobs 8 --output outputs/paper
```

These stages inherit the source study's backend and selected change dates.
The stabilized fit uses a fixed penalty on the log risk ratio. Ordinary
post-alarm Wald intervals are descriptive, conditional on the selected dates.
The separately calibrated stopped-root intervals require calibration under
the same detector before independent validation. No old-rule cutoff can be
transferred to a new stopping rule.

Freeze the numerical interval calibration before generating the independent
validation bank, then apply it using the printed cutoff hash:

```bash
ocfr-analyze --protocol compact-v2 --input outputs/paper --calibrate-intervals
ocfr-simulate --protocol compact-v2-release --stage estimation-validation --mode full --jobs 8 --output outputs/paper
ocfr-analyze --protocol compact-v2 --input outputs/paper --apply-intervals --expected-cutoff-sha256 PRINTED_HASH
ocfr-simulate --protocol compact-v2-release --stage estimation-ls --estimation-bank both --mode full --jobs 8 --output outputs/paper
ocfr-analyze --protocol compact-v2 --input outputs/paper
```

Interval calibration uses training errors only and freezes one asymmetric
root pair per target across all 16 declared design conditions. The independent
bank checks coverage at newly selected first alarms. This finite-design
empirical assessment does not establish a general selection-adjusted coverage
guarantee or transfer calibrated cutoffs to a real-data series. LS and OCFR point estimates use
identical selected dates and observations. Figures and tables report actual
alarm and availability counts for their respective denominators.

## Real-data applications

```bash
ocfr-real-data --stage all --backend c
```

This downloads public aggregate COVID-19 and Bangladesh dengue data, processes
the series, runs monitoring and generates summaries and figures. Individual
stages are `fetch`, `run` and `plot`. Use `--data PATH`, `--output PATH` and
`--backend python` as needed. `--refresh` downloads the sources again.
COVID-19 monitoring uses Japan, South Korea and the United Kingdom; dengue
reports use the Bangladesh 2023 weekly series. The declared delay distributions
are supplied to each application. Source addresses and preprocessing settings
are fixed in `config.py`; downloaded bytes and processed inputs are hashed.
Application intervals are unadjusted nominal 95% Wald summaries of the selected
first-alarm fit, not selection-adjusted confidence statements. The simulation
interval calibration is not applied to these applications. A final statistic
below its boundary does not reverse an earlier alarm.

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
