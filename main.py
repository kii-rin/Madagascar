from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
ELEVATION_M = 900.0
POPULATION_SIZE = 80
ELITE_COUNT = 8


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
class PlantGenome:
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
    rng = random.Random(seed)
    first_day = date(2025, 1, 1)
    year: list[WeatherDay] = []

    for day_number in range(365):
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


def random_genome(rng: random.Random) -> PlantGenome:
    return PlantGenome(
        preferred_temperature_c=rng.uniform(10.0, 35.0),
        temperature_tolerance_c=rng.uniform(1.0, 15.0),
        preferred_sunlight=rng.random(),
        sunlight_tolerance=rng.uniform(0.05, 0.50),
        preferred_rain=rng.random(),
        rain_tolerance=rng.uniform(0.05, 0.50),
        preferred_humidity=rng.random(),
        humidity_tolerance=rng.uniform(0.05, 0.50),
        preferred_wind=rng.random(),
        wind_tolerance=rng.uniform(0.05, 0.50),
        seed_amount=rng.randint(1, 20),
    )


def mutate(genome: PlantGenome, rng: random.Random) -> PlantGenome:
    def move(value: float, sigma: float, low: float, high: float) -> float:
        return max(low, min(high, value + rng.gauss(0.0, sigma)))

    return PlantGenome(
        preferred_temperature_c=move(genome.preferred_temperature_c, 1.2, 10.0, 35.0),
        temperature_tolerance_c=move(genome.temperature_tolerance_c, 0.8, 1.0, 15.0),
        preferred_sunlight=move(genome.preferred_sunlight, 0.07, 0.0, 1.0),
        sunlight_tolerance=move(genome.sunlight_tolerance, 0.04, 0.05, 0.50),
        preferred_rain=move(genome.preferred_rain, 0.07, 0.0, 1.0),
        rain_tolerance=move(genome.rain_tolerance, 0.04, 0.05, 0.50),
        preferred_humidity=move(genome.preferred_humidity, 0.07, 0.0, 1.0),
        humidity_tolerance=move(genome.humidity_tolerance, 0.04, 0.05, 0.50),
        preferred_wind=move(genome.preferred_wind, 0.07, 0.0, 1.0),
        wind_tolerance=move(genome.wind_tolerance, 0.04, 0.05, 0.50),
        seed_amount=max(1, min(20, genome.seed_amount + rng.choice([-1, 0, 0, 0, 1]))),
    )


def crossover(a: PlantGenome, b: PlantGenome, rng: random.Random) -> PlantGenome:
    values = {}
    for field_name in PlantGenome.__dataclass_fields__:
        parent = a if rng.random() < 0.5 else b
        values[field_name] = getattr(parent, field_name)
    return PlantGenome(**values)


def range_match(value: float, preferred: float, tolerance: float) -> float:
    distance = (value - preferred) / max(tolerance, 1e-9)
    return math.exp(-0.5 * distance * distance)


def evaluate_genome(
    genome: PlantGenome, weather: list[WeatherDay]
) -> tuple[Evaluation, list[dict]]:
    totals = {
        "temperature": 0.0,
        "sunlight": 0.0,
        "rain": 0.0,
        "humidity": 0.0,
        "wind": 0.0,
    }
    daily_rows: list[dict] = []

    for current in weather:
        scores = {
            "temperature": range_match(
                current.temperature_c,
                genome.preferred_temperature_c,
                genome.temperature_tolerance_c,
            ),
            "sunlight": range_match(
                current.sunlight,
                genome.preferred_sunlight,
                genome.sunlight_tolerance,
            ),
            "rain": range_match(
                current.rain, genome.preferred_rain, genome.rain_tolerance
            ),
            "humidity": range_match(
                current.humidity,
                genome.preferred_humidity,
                genome.humidity_tolerance,
            ),
            "wind": range_match(
                current.wind, genome.preferred_wind, genome.wind_tolerance
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
                "overall_adaptation": sum(scores.values()) / 5.0,
            }
        )

    days = len(weather)
    temperature = totals["temperature"] / days
    sunlight = totals["sunlight"] / days
    rain = totals["rain"] / days
    humidity = totals["humidity"] / days
    wind = totals["wind"] / days
    overall = (temperature + sunlight + rain + humidity + wind) / 5.0
    expected_seeds = genome.seed_amount * overall * overall

    tolerance_cost = (
        (genome.temperature_tolerance_c - 1.0) / 14.0
        + (genome.sunlight_tolerance - 0.05) / 0.45
        + (genome.rain_tolerance - 0.05) / 0.45
        + (genome.humidity_tolerance - 0.05) / 0.45
        + (genome.wind_tolerance - 0.05) / 0.45
    ) / 5.0
    seed_cost = (genome.seed_amount - 1) / 19.0
    fitness = (
        overall * 80.0
        + expected_seeds
        - tolerance_cost * 4.0
        - seed_cost * 2.0
    )

    return (
        Evaluation(
            temperature,
            sunlight,
            rain,
            humidity,
            wind,
            overall,
            expected_seeds,
            fitness,
        ),
        daily_rows,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(generations: int, seed: int) -> None:
    rng = random.Random(seed)
    weather = generate_synthetic_year(seed)
    population = [random_genome(rng) for _ in range(POPULATION_SIZE)]

    best_genome = population[0]
    best_evaluation, _ = evaluate_genome(best_genome, weather)

    for generation in range(generations):
        ranked = []
        for genome in population:
            evaluation, _ = evaluate_genome(genome, weather)
            ranked.append((evaluation.fitness, genome, evaluation))
        ranked.sort(key=lambda item: item[0], reverse=True)

        best_fitness, best_genome, best_evaluation = ranked[0]
        average_fitness = sum(item[0] for item in ranked) / len(ranked)
        print(
            f"Generation {generation:03d} | "
            f"avg={average_fitness:7.3f} | "
            f"best={best_fitness:7.3f} | "
            f"adaptation={best_evaluation.overall_adaptation:.3f} | "
            f"seeds={best_evaluation.expected_seeds:.2f}"
        )

        elites = [item[1] for item in ranked[:ELITE_COUNT]]
        parent_pool = [item[1] for item in ranked[: POPULATION_SIZE // 3]]
        next_population = list(elites)
        while len(next_population) < POPULATION_SIZE:
            child = crossover(rng.choice(parent_pool), rng.choice(parent_pool), rng)
            next_population.append(mutate(child, rng))
        population = next_population

    final_evaluation, daily_rows = evaluate_genome(best_genome, weather)
    OUTPUT_DIR.mkdir(exist_ok=True)
    write_csv(OUTPUT_DIR / "synthetic_year.csv", [asdict(day) for day in weather])
    write_csv(OUTPUT_DIR / "winner_daily_adaptation.csv", daily_rows)

    result = {
        "version": "0.5",
        "engine": "built-in minimal evolutionary loop",
        "environment": {
            "synthetic": True,
            "days": 365,
            "elevation_m": ELEVATION_M,
        },
        "plant_genome": asdict(best_genome),
        "lifetime_scores": asdict(final_evaluation),
        "evolution": {
            "generations": generations,
            "population_size": POPULATION_SIZE,
            "elite_count": ELITE_COUNT,
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
    run(arguments.generations, arguments.seed)
