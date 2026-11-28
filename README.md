# OCFR

OCFR detects a persistent change in epidemic case-fatality risk from daily
case and death counts. A negative-binomial model links earlier case cohorts
to later deaths through a supplied delay distribution. At the first alarm,
OCFR estimates the date separating the two case cohorts.

## Monitor your own data

Install from the repository root with Python 3.10 or later.

```bash
git clone https://github.com/BobZhangHT/OCFR.git
cd OCFR
python -m pip install -e .
```

Choose a monitoring horizon and analysis days before monitoring starts.
Supply one row per day in calendar order, with columns `date,cases,deaths`,
and choose a delay distribution that matches how the two counts are dated.
The interface counts delays from one day onward. This example uses a 182-day
horizon, weekly analyses starting on day 35, and Gamma delay probabilities
for lags 1 to 45 days.

```python
from csv import DictReader
from itertools import islice

from methods import OnlineGaussianProcessScore, gamma_delay_pmf, make_sampler

backend = "c"
horizon = 182
detector = OnlineGaussianProcessScore(
    gamma_delay_pmf(14, shape=4, maximum_lag=45),
    horizon=horizon,
    planned_looks=tuple(range(35, horizon + 1, 7)),
    alpha=0.05,
    rng=make_sampler(2026111404, backend=backend),
    backend=backend,
)

dates = []
with open("daily_counts.csv", newline="", encoding="utf-8-sig") as stream:
    for row in islice(DictReader(stream), horizon):
        dates.append(row["date"])
        result = detector.update(int(row["cases"]), int(row["deaths"]))
        if result.alarm:
            print("First alarm", dates[-1])
            print("Estimated cohort change date", dates[result.estimated_change_day])
            break
    else:
        print("No alarm on the supplied monitoring days")
```

The C backend requires a supported compiler on first use. Set
`backend = "python"` to run without a compiler. For live monitoring, call
`detector.update` once per day as new counts arrive.

## Reproduce the paper

### Simulation study

The full replication workflow requires the C backend.

```bash
python simulation.py --stage paper --mode full --jobs 8 --output outputs/paper
```

The full workflow generates the simulation results, tables, and Figures 2 to 6.
Run the same command again to resume from its saved stages.

### Surveillance applications

The applications use [WHO COVID-19 data](https://data.who.int/dashboards/covid19/data)
for Japan, South Korea, and the United Kingdom, and
[SIVEP-Gripe records](https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026)
for Brazilian influenza in 2024.

```bash
python real_data.py --stage all --data data/paper --output outputs/surveillance --require-manuscript-inputs
python real_data.py --stage benchmarks --data data/paper --output outputs/surveillance --workers 8 --require-manuscript-inputs
python real_data.py --stage audit --data data/paper --output outputs/surveillance --workers 8 --require-manuscript-inputs
```

The first command downloads the data, runs OCFR and the sensitivity analyses,
and generates Figures 7 to 9. The next two reproduce the benchmark comparisons
and fitted-null alarm rates. The input check compares the processed counts
with the snapshots used in the paper.

## Contact

[Hengtao Zhang](https://bobzhanght.github.io/), Guangdong Ocean University.
Email [zhanght@gdou.edu.cn](mailto:zhanght@gdou.edu.cn) or open a
[GitHub issue](https://github.com/BobZhangHT/OCFR/issues).
