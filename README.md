# Madagascar — V0.5

Minimal synthetic plant-evolution practice build.

## Included

- one synthetic 365-day environment
- one fixed elevation: 900 m
- daily temperature, sunlight, rain, humidity, and wind
- one simple inherited plant genome
- crossover and mutation create offspring genomes
- five separate lifetime adaptation scores
- seed amount is part of the genome

The plant has no coordinates and no global map. It is evaluated only against the weather at its own location.

## Important

V0.5 uses a small built-in evolutionary loop written in plain Python. It does **not** yet implement full NEAT topology evolution. This keeps the first version runnable without external packages while preserving population selection, crossover, mutation, and inherited traits.

## Run

```bash
python main.py --generations 10
```

No package installation is required.

## Outputs

- `outputs/synthetic_year.csv`
- `outputs/winner_daily_adaptation.csv`
- `outputs/winner.json`
