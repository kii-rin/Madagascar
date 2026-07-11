from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import neat

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
ELEVATION_M = 900.0


@dataclass(frozen=True)
class WeatherDay:
    day: int
    date: str
    elevation_m: float
    temperature_c: float
    sunlight: float
    rain: float
    humidity: float
    wind: float


@dataclass(frozen=True)
class PlantTraits:
    preferred_temperature_c: float
    temperature_tolerance_c: float
    preferred_sunlight: float
    sunlight_tolerance: float
    preferred_rain: float
    rain_tolerance: float
    preferred_humidity: float
    humidity_tolerance: float
    preferred_wind: float
    wind_tolerance: float
    seed_amount: int


@dataclass(frozen=True)
class Evaluation:
    temperature_adaptation: float
    sunlight_adaptation: float
    rain_adaptation: float
    humidity_adaptation: float
    wind_adaptation: float
    overall_adaptation: float
    expected_seeds: float
    fitness: float


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def generate_synthetic_year(seed: int) -> list[WeatherDay]:
    """Create one repeatable synthetic year at a single 900 m location."""
    rng = random.Random(seed)
    first_day = date(2025, 1, 1)
    year: list[WeatherDay] = []

    for day_number in range(365):
        # Southern-hemisphere warm/wet season peaks near January.
        season = math.cos(2.0 * math.pi * (day_number - 15) / 365.0)
        wet_season = (season + 1.0) / 2.0

        sea_level_temperature = 27.5 + 2.5 * season
        temperature = sea_level_temperature - ELEVATION_M * 0.0055
        temperature += rng.gauss(0.0, 0.8)

        rain = clamp(0.12 + 0.72 * wet_season + rng.gauss(0.0, 0.08))
        humidity = clamp(0.42 + 0.50 * rain + rng.gauss(0.0, 0.04))
        sunlight = clamp(0.82 - 0.22 * rain + rng.gauss(0.0, 0.05))
        wind = clamp(0.22 + 0.16 * (1.0 - wet_season) + rng.gauss(0.0, 0.05))

        year.append(
            WeatherDay(
                day=day_number + 1,
                date=str(first_day + timedelta(days=day_number)),
                elevation_m=ELEVATION_M,
                temperature_c=round(temperature, 3),
                sunlight=round(sunlight, 4),
                rain=round(rain, 4),
                humidity=round(humidity, 4),
                wind=round(wind, 4),
            )
        )

    return year


def express_traits(genome: neat.DefaultGenome, config: neat.Config) -> PlantTraits:
    """Read the NEAT genome once at birth and turn it into inherited traits."""
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    outputs = [clamp(value) for value in network.activate([1.0])]

    return PlantTraits(
        preferred_temperature_c=10.0 + outputs[0] * 25.0,
        temperature_tolerance_c=1.0 + outputs[1] * 14.0,
        preferred_sunlight=outputs[2],
        sunlight_tolerance=0.05 + outputs[3] * 0.45,
        preferred_rain=outputs[4],
        rain_tolerance=0.05 + outputs[5] * 0.45,
        preferred_humidity=outputs[6],
        humidity_tolerance=0.05 + outputs[7] * 0.45,
        preferred_wind=outputs[8],
        wind_tolerance=0.05 + outputs[9] * 0.45,
        seed_amount=1 + round(outputs[10] * 19.0),
    )


def range_match(value: float, preferred: float, tolerance: float) -> float:
    """Return 1 at the preferred value and fall smoothly toward 0 outside it."""
    distance = (value - preferred) / max(tolerance, 1e-9)
    return math.exp(-0.5 * distance * distance)


def evaluate_traits(
    traits: PlantTraits,
    weather: list[WeatherDay],
    connection_count: int = 0,
) -> tuple[Evaluation, list[dict[str, float | int | str]]]:
    totals = {
        "temperature": 0.0,
        "sunlight": 0.0,
        "rain": 0.0,
        "humidity": 0.0,
        "wind": 0.0,
    }
    daily_rows: list[dict[str, float | int | str]] = []

    for current in weather:
        scores = {
            "temperature": range_match(
                current.temperature_c,
                traits.preferred_temperature_c,
                traits.temperature_tolerance_c,
            ),
            "sunlight": range_match(
                current.sunlight,
                traits.preferred_sunlight,
                traits.sunlight_tolerance,
            ),
            "rain": range_match(
                current.rain,
                traits.preferred_rain,
                traits.rain_tolerance,
            ),
            "humidity": range_match(
                current.humidity,
                traits.preferred_humidity,
                traits.humidity_tolerance,
            ),
            "wind": range_match(
                current.wind,
                traits.preferred_wind,
                traits.wind_tolerance,
            ),
        }

        for key, value in scores.items():
            totals[key] += value

        daily_rows.append(
            {
                **asdict(current),
                "temperature_adaptation": scores["temperature"],
                "sunlight_adaptation": scores["sunlight"],
                "rain_adaptation": scores["rain"],
                "humidity_adaptation": scores["humidity"],
                "wind_adaptation": scores["wind"],
                "overall_adaptation": sum(scores.values()) / len(scores),
            }
        )

    days = float(len(weather))
    temperature = totals["temperature"] / days
    sunlight = totals["sunlight"] / days
    rain = totals["rain"] / days
    humidity = totals["humidity"] / days
    wind = totals["wind"] / days
    overall = (temperature + sunlight + rain + humidity + wind) / 5.0

    expected_seeds = traits.seed_amount * overall * overall

    # Small costs prevent the easiest solution from always being maximum
    # tolerance, maximum seeds, and maximum network size.
    tolerance_cost = (
        (traits.temperature_tolerance_c - 1.0) / 14.0
        + (traits.sunlight_tolerance - 0.05) / 0.45
        + (traits.rain_tolerance - 0.05) / 0.45
        + (traits.humidity_tolerance - 0.05) / 0.45
        + (traits.wind_tolerance - 0.05) / 0.45
    ) / 5.0
    seed_cost = (traits.seed_amount - 1) / 19.0

    fitness = (
        overall * 80.0
        + (expected_seeds / 20.0) * 20.0
        - tolerance_cost * 4.0
        - seed_cost * 2.0
        - connection_count * 0.01
    )

    return (
        Evaluation(
            temperature_adaptation=temperature,
            sunlight_adaptation=sunlight,
            rain_adaptation=rain,
            humidity_adaptation=humidity,
            wind_adaptation=wind,
            overall_adaptation=overall,
            expected_seeds=expected_seeds,
            fitness=fitness,
        ),
        daily_rows,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_config() -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(ROOT / "config-neat.ini"),
    )


def run(generations: int, seed: int) -> None:
    weather = generate_synthetic_year(seed)
    config = build_config()

    def evaluate_population(genomes, neat_config) -> None:
        for _, genome in genomes:
            traits = express_traits(genome, neat_config)
            enabled_connections = sum(
                1 for connection in genome.connections.values() if connection.enabled
            )
            evaluation, _ = evaluate_traits(traits, weather, enabled_connections)
            genome.fitness = evaluation.fitness

    population = neat.Population(config, seed=seed)
    population.add_reporter(neat.StdOutReporter(show_species_detail=False))
    winner = population.run(evaluate_population, generations)

    traits = express_traits(winner, config)
    enabled_connections = sum(
        1 for connection in winner.connections.values() if connection.enabled
    )
    evaluation, daily_rows = evaluate_traits(traits, weather, enabled_connections)

    OUTPUT_DIR.mkdir(exist_ok=True)
    write_csv(OUTPUT_DIR / "synthetic_year.csv", [asdict(day) for day in weather])
    write_csv(OUTPUT_DIR / "winner_daily_adaptation.csv", daily_rows)

    result = {
        "version": "0.5",
        "environment": {
            "synthetic": True,
            "days": 365,
            "elevation_m": ELEVATION_M,
            "inputs_seen_by_plant": [
                "temperature_c",
                "sunlight",
                "rain",
                "humidity",
                "wind",
            ],
        },
        "plant_genome": asdict(traits),
        "lifetime_scores": asdict(evaluation),
        "neat": {
            "generations": generations,
            "enabled_connections": enabled_connections,
            "hidden_nodes": max(
                0, len(winner.nodes) - config.genome_config.num_outputs
            ),
        },
    }
    (OUTPUT_DIR / "winner.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print("\nWinner")
    print(json.dumps(result, indent=2))
    print("\nFiles written to outputs/.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Madagascar synthetic plant V0.5")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(generations=arguments.generations, seed=arguments.seed)
