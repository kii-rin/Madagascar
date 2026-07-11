# Madagascar — V0.5

Minimal practice version:

- one synthetic 365-day environment
- one fixed elevation: 900 m
- daily temperature, sunlight, rain, humidity, and wind
- one simple plant genome
- NEAT creates and mutates offspring genomes
- five separate lifetime adaptation scores

The plant has no coordinates and no global map. It only experiences the weather at
its own location.

## How the genome works

A NEAT genome is read once at birth and becomes:

- preferred temperature + tolerance
- preferred sunlight + tolerance
- preferred rain + tolerance
- preferred humidity + tolerance
- preferred wind + tolerance
- seed amount

The plant is evaluated for one full synthetic year. NEAT-Python then selects,
crosses, and mutates genomes to create the next generation.

## Install

Python 3.11 or 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

For a quick test:

```bash
python main.py --generations 5
```

## Output

- `outputs/synthetic_year.csv` — the 365 environment inputs
- `outputs/winner_daily_adaptation.csv` — daily adaptation breakdown
- `outputs/winner.json` — winning genome traits and lifetime scores

V0.5 is intentionally synthetic. No real Madagascar data or multiple plants yet.
