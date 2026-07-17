# Madagascar — V0.5

Minimal synthetic plant evolution with a self-contained NEAT engine written in plain Python.

## Environment

- one synthetic 365-day year
- one fixed elevation: 900 m
- daily temperature, sunlight, rain, humidity, and wind
- the plant receives no map coordinates or global information

## Plant genome outputs

The evolved neural network is activated once at birth and produces:

- preferred temperature + tolerance
- preferred sunlight + tolerance
- preferred rain + tolerance
- preferred humidity + tolerance
- preferred wind + tolerance
- seed amount

## NEAT features included

- node genes
- connection genes
- connection weights and node biases
- historical innovation numbers
- add-node mutation
- add-connection mutation
- enabled and disabled connection genes
- feed-forward topology validation
- NEAT crossover aligned by innovation number
- excess and disjoint gene compatibility distance
- speciation
- adjusted fitness sharing
- species stagnation removal
- species elitism
- full evolving neural-network topology

There are no external packages. The implementation contains only the NEAT machinery needed for this project.

## Run

```bash
python main.py --generations 30
```

For a shorter run:

```bash
python main.py --generations 8
```

The terminal log shows fitness, species count, node count, hidden nodes, enabled connections, adaptation, and expected seeds.

## Outputs

- `outputs/synthetic_year.csv`
- `outputs/winner_daily_adaptation.csv`
- `outputs/winner.json`

`winner.json` includes the complete winning topology: every node, connection, weight, enabled state, and innovation number.
