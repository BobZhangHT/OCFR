# OCFR core

This directory is the standalone, upload-ready implementation of **OCFR**
(online change detection for delay-adjusted epidemic case-fatality risk).  It
contains the complete simulation, analysis, native/reference methods, and
real-data workflow without depending on the parent manuscript repository.

## Files

- `config.py`: the single source of truth for data-generating processes,
  scenarios, methods, seeds, and real-data settings.
- `methods.py`: portable OCFR, benchmark, interval, sampler, and C-binding
  implementations. `backend="c"` is the strict default; `backend="python"`
  selects the numerically matched reference implementation.
- `simulation.py`: deterministic parallel simulation runner with atomic
  per-replication caches, automatic resume, and scoped reset.
- `analysis.py`: type-I-error, power, delay, and localization summaries plus
  publication-ready PDF/PNG figures and CSV tables.
- `real_data.py`: data download, provenance-preserving preprocessing,
  surveillance experiments, interval estimation, benchmark analysis, and
  figures for COVID-19 and Bangladesh dengue.
- `csrc/ocfr_kernels.c`: optional C99 acceleration and matched random-number
  generation used by `methods.py`.

The five Python files are intentionally flat so that copying them and `csrc/`
into another project is sufficient. Generated data, caches, binaries, and
figures are excluded by `.gitignore`.

## Repository boundary

The repository contains source code, tests, configuration, citation metadata,
and documentation only. Generated outputs, downloaded data, caches, compiled
libraries, figures, and manuscript files are excluded by `.gitignore` and by
an automated repository-boundary test.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

The Python backend requires no compiler. To build the default C backend, use
Zig, GCC, Clang, MSVC, or install the Python `ziglang` package, then run:

```bash
python -c "import methods; print(methods.build_native())"
python -c "import methods; print(methods.assert_backend_parity())"
```

The parity check requires exact equality for all matched random samplers and a
maximum score-field difference of at most `1e-10`. A platform's application
control policy may prevent a newly built library from loading; in that case,
use `--backend python` until the locally built library is approved.

## Simulations

Run the 10-replication smoke workflow:

```bash
python simulation.py --mode demo --jobs -1
```

Run the full 1000-replication workflow:

```bash
python simulation.py --mode full --jobs -1
```

For the high-precision Figure 5 size audit, use the separately named frozen
protocol:

```bash
python simulation.py --mode full --protocol size-control --backend c --jobs -1
```

This protocol uses 19,999 threshold-training episodes, 10,000 independent
null holdout episodes, and 1,000 alternative episodes per scenario. The
holdout is audit-only: it is never fed back into threshold selection. The
generated `size_control_gate.csv` reports whether the 90% exact binomial
interval lies wholly inside the prespecified practical-equivalence region
0.04--0.06. Primary OCFR retains its theoretical boundary and is shown
separately from the empirically alpha-matched OCFR-AM comparison.

Demo and full modes use identical seeds, scenarios, data generation, methods,
and thresholds; their only difference is 10 versus 1000 replications per
scenario. Therefore a full run in the same output directory reuses the first
10 cached replications. Each replication is written atomically, so repeating a
command resumes an interrupted run. Add `--reset` to remove only the selected
runner's cache and generated simulation tables before recomputing.

Threshold training, independent null holdout, and alternatives always remain
separate. The output reports the primary prespecified OCFR rule separately
from an empirically alpha-matched OCFR comparator; benchmark power is therefore
interpreted beside, rather than conditionally suppressed by, held-out size.
With only 10 calibration replications, the conservative
`(B+1)` empirical 0.95 quantile is infinite; consequently demo output is a
software smoke test, not statistical evidence. Use full mode for reported
results.

`analysis.py` consumes the versioned E1--E6 aggregate evidence contract used
for the paper and writes five PDF/PNG figures plus their derived CSV tables.
Point it to an assembled evidence directory with
`python analysis.py --input PATH --output PATH`; it rejects missing experiments,
failed rows, broken pairing hashes, or manifest/hash mismatches rather than
silently plotting a partial run.

## Real-data applications

The complete workflow downloads and caches public aggregate data, records
source hashes, preprocesses revisions, runs OCFR and benchmark analyses, and
creates figures:

```bash
python real_data.py --stage all --backend c
```

Stages can be run independently with `--stage fetch`, `run`, or `plot`.
`--refresh` redownloads the raw sources. Use `--backend python` for the
reference implementation or `--backend auto` for a C-first fallback. See
`python real_data.py --help` for custom data and output directories. Downloaded
source data remain subject to their providers' terms and should be reviewed
before redistribution.

## Reproducibility and outputs

All pseudorandom streams are derived from locked master seeds and stable task
identifiers; results do not depend on process completion order. Outputs are
written below `outputs/` by default. Simulation manifests record the scientific
configuration, backend request, cache reuse, and failure counts. The real-data
workflow stores cleaned inputs below `data/processed/` and writes method traces,
comparison statistics, figures, and application summaries below `outputs/`.

This is research software and is not a clinical decision system. Citation
metadata are provided in `CITATION.cff`; the manuscript DOI or preprint link
can be added when it becomes available.
