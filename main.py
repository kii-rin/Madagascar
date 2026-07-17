from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
ELEVATION_M = 900.0
POPULATION_SIZE = 80
INPUT_IDS = (0,)
OUTPUT_IDS = tuple(range(1, 12))


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


@dataclass
class NodeGene:
    node_id: int
    kind: str
    bias: float = 0.0


@dataclass
class ConnectionGene:
    source: int
    target: int
    weight: float
    enabled: bool
    innovation: int


@dataclass
class Genome:
    genome_id: int
    nodes: dict[int, NodeGene]
    connections: dict[int, ConnectionGene]
    fitness: float = 0.0
    adjusted_fitness: float = 0.0

    def clone(self, new_id: int) -> "Genome":
        return Genome(
            genome_id=new_id,
            nodes={key: NodeGene(node.node_id, node.kind, node.bias) for key, node in self.nodes.items()},
            connections={
                key: ConnectionGene(gene.source, gene.target, gene.weight, gene.enabled, gene.innovation)
                for key, gene in self.connections.items()
            },
            fitness=self.fitness,
        )


@dataclass
class Species:
    species_id: int
    representative: Genome
    members: list[Genome] = field(default_factory=list)
    best_fitness: float = float("-inf")
    stagnant_generations: int = 0


class InnovationTracker:
    def __init__(self) -> None:
        self.next_innovation = 0
        self.next_node_id = max(OUTPUT_IDS) + 1
        self.connection_innovations: dict[tuple[int, int], int] = {}
        self.split_history: dict[int, tuple[int, int, int]] = {}

    def connection(self, source: int, target: int) -> int:
        key = (source, target)
        if key not in self.connection_innovations:
            self.connection_innovations[key] = self.next_innovation
            self.next_innovation += 1
        return self.connection_innovations[key]

    def split(self, old_innovation: int, source: int, target: int) -> tuple[int, int, int]:
        if old_innovation not in self.split_history:
            node_id = self.next_node_id
            self.next_node_id += 1
            first = self.connection(source, node_id)
            second = self.connection(node_id, target)
            self.split_history[old_innovation] = (node_id, first, second)
        return self.split_history[old_innovation]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def generate_synthetic_year(seed: int) -> list[WeatherDay]:
    rng = random.Random(seed)
    first_day = date(2025, 1, 1)
    year: list[WeatherDay] = []
    for day_number in range(365):
        season = math.cos(2.0 * math.pi * (day_number - 15) / 365.0)
        wet_season = (season + 1.0) / 2.0
        temperature = 27.5 + 2.5 * season - ELEVATION_M * 0.0055
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


def creates_cycle(genome: Genome, source: int, target: int) -> bool:
    graph: dict[int, list[int]] = {}
    for gene in genome.connections.values():
        if gene.enabled:
            graph.setdefault(gene.source, []).append(gene.target)
    graph.setdefault(source, []).append(target)
    stack = [target]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node == source:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, []))
    return False


def topological_order(genome: Genome) -> list[int]:
    enabled = [gene for gene in genome.connections.values() if gene.enabled]
    indegree = {node_id: 0 for node_id in genome.nodes}
    outgoing: dict[int, list[int]] = {}
    for gene in enabled:
        indegree[gene.target] = indegree.get(gene.target, 0) + 1
        outgoing.setdefault(gene.source, []).append(gene.target)
    queue = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    order: list[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for target in outgoing.get(node, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort()
    if len(order) != len(indegree):
        raise ValueError("Genome contains a cycle")
    return order


def activate(genome: Genome) -> list[float]:
    values = {INPUT_IDS[0]: 1.0}
    incoming: dict[int, list[ConnectionGene]] = {}
    for gene in genome.connections.values():
        if gene.enabled:
            incoming.setdefault(gene.target, []).append(gene)
    for node_id in topological_order(genome):
        node = genome.nodes[node_id]
        if node.kind == "input":
            continue
        total = node.bias
        for gene in incoming.get(node_id, []):
            total += values.get(gene.source, 0.0) * gene.weight
        values[node_id] = sigmoid(total)
    return [values.get(node_id, 0.5) for node_id in OUTPUT_IDS]


def express_traits(genome: Genome) -> PlantTraits:
    outputs = activate(genome)
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
    distance = (value - preferred) / max(tolerance, 1e-9)
    return math.exp(-0.5 * distance * distance)


def evaluate_traits(traits: PlantTraits, weather: list[WeatherDay], complexity: int) -> tuple[Evaluation, list[dict]]:
    totals = {key: 0.0 for key in ("temperature", "sunlight", "rain", "humidity", "wind")}
    rows: list[dict] = []
    for current in weather:
        scores = {
            "temperature": range_match(current.temperature_c, traits.preferred_temperature_c, traits.temperature_tolerance_c),
            "sunlight": range_match(current.sunlight, traits.preferred_sunlight, traits.sunlight_tolerance),
            "rain": range_match(current.rain, traits.preferred_rain, traits.rain_tolerance),
            "humidity": range_match(current.humidity, traits.preferred_humidity, traits.humidity_tolerance),
            "wind": range_match(current.wind, traits.preferred_wind, traits.wind_tolerance),
        }
        for key, value in scores.items():
            totals[key] += value
        rows.append(
            {
                **asdict(current),
                **{f"{key}_adaptation": value for key, value in scores.items()},
                "overall_adaptation": sum(scores.values()) / 5.0,
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
    tolerance_cost = (
        (traits.temperature_tolerance_c - 1.0) / 14.0
        + (traits.sunlight_tolerance - 0.05) / 0.45
        + (traits.rain_tolerance - 0.05) / 0.45
        + (traits.humidity_tolerance - 0.05) / 0.45
        + (traits.wind_tolerance - 0.05) / 0.45
    ) / 5.0
    seed_cost = (traits.seed_amount - 1) / 19.0
    fitness = overall * 80.0 + expected_seeds - tolerance_cost * 4.0 - seed_cost * 2.0 - complexity * 0.01
    return Evaluation(temperature, sunlight, rain, humidity, wind, overall, expected_seeds, fitness), rows


def initial_genome(genome_id: int, tracker: InnovationTracker, rng: random.Random) -> Genome:
    nodes = {0: NodeGene(0, "input", 0.0)}
    for node_id in OUTPUT_IDS:
        nodes[node_id] = NodeGene(node_id, "output", rng.gauss(0.0, 1.0))
    connections: dict[int, ConnectionGene] = {}
    for target in OUTPUT_IDS:
        innovation = tracker.connection(0, target)
        connections[innovation] = ConnectionGene(0, target, rng.gauss(0.0, 1.0), True, innovation)
    return Genome(genome_id, nodes, connections)


def mutate_weights(genome: Genome, rng: random.Random) -> None:
    for gene in genome.connections.values():
        if rng.random() < 0.8:
            gene.weight += rng.gauss(0.0, 0.5)
        elif rng.random() < 0.1:
            gene.weight = rng.gauss(0.0, 1.0)
        gene.weight = max(-8.0, min(8.0, gene.weight))
    for node in genome.nodes.values():
        if node.kind != "input" and rng.random() < 0.7:
            node.bias += rng.gauss(0.0, 0.3)
            node.bias = max(-8.0, min(8.0, node.bias))


def mutate_add_connection(genome: Genome, tracker: InnovationTracker, rng: random.Random) -> None:
    sources = [node_id for node_id, node in genome.nodes.items() if node.kind != "output"]
    targets = [node_id for node_id, node in genome.nodes.items() if node.kind != "input"]
    existing = {(gene.source, gene.target) for gene in genome.connections.values()}
    candidates = [
        (source, target)
        for source in sources
        for target in targets
        if source != target and (source, target) not in existing and not creates_cycle(genome, source, target)
    ]
    if not candidates:
        return
    source, target = rng.choice(candidates)
    innovation = tracker.connection(source, target)
    genome.connections[innovation] = ConnectionGene(source, target, rng.gauss(0.0, 1.0), True, innovation)


def mutate_add_node(genome: Genome, tracker: InnovationTracker, rng: random.Random) -> None:
    candidates = [gene for gene in genome.connections.values() if gene.enabled]
    if not candidates:
        return
    old = rng.choice(candidates)
    old.enabled = False
    node_id, first_innovation, second_innovation = tracker.split(old.innovation, old.source, old.target)
    genome.nodes.setdefault(node_id, NodeGene(node_id, "hidden", 0.0))
    genome.connections[first_innovation] = ConnectionGene(old.source, node_id, 1.0, True, first_innovation)
    genome.connections[second_innovation] = ConnectionGene(node_id, old.target, old.weight, True, second_innovation)


def mutate_toggle_connection(genome: Genome, rng: random.Random) -> None:
    if not genome.connections:
        return
    gene = rng.choice(list(genome.connections.values()))
    if gene.enabled:
        gene.enabled = False
    elif not creates_cycle(genome, gene.source, gene.target):
        gene.enabled = True


def mutate(genome: Genome, tracker: InnovationTracker, rng: random.Random) -> None:
    mutate_weights(genome, rng)
    if rng.random() < 0.10:
        mutate_add_connection(genome, tracker, rng)
    if rng.random() < 0.05:
        mutate_add_node(genome, tracker, rng)
    if rng.random() < 0.01:
        mutate_toggle_connection(genome, rng)


def compatibility_distance(first: Genome, second: Genome) -> float:
    innovations_first = set(first.connections)
    innovations_second = set(second.connections)
    matching = innovations_first & innovations_second
    max_first = max(innovations_first, default=-1)
    max_second = max(innovations_second, default=-1)
    excess = 0
    disjoint = 0
    for innovation in innovations_first ^ innovations_second:
        if innovation > min(max_first, max_second):
            excess += 1
        else:
            disjoint += 1
    average_weight_difference = (
        sum(abs(first.connections[index].weight - second.connections[index].weight) for index in matching) / len(matching)
        if matching
        else 0.0
    )
    normalizer = max(len(innovations_first), len(innovations_second))
    if normalizer < 20:
        normalizer = 1
    return excess / normalizer + disjoint / normalizer + 0.4 * average_weight_difference


def crossover(fitter: Genome, other: Genome, child_id: int, rng: random.Random) -> Genome:
    if other.fitness > fitter.fitness:
        fitter, other = other, fitter
    equal = math.isclose(fitter.fitness, other.fitness)
    child_connections: dict[int, ConnectionGene] = {}
    for innovation, fit_gene in fitter.connections.items():
        other_gene = other.connections.get(innovation)
        chosen = other_gene if other_gene is not None and rng.random() < 0.5 else fit_gene
        if equal and innovation not in other.connections and rng.random() < 0.5:
            continue
        enabled = chosen.enabled
        if other_gene is not None and (not fit_gene.enabled or not other_gene.enabled):
            enabled = rng.random() >= 0.75
        child_connections[innovation] = ConnectionGene(
            chosen.source, chosen.target, chosen.weight, enabled, innovation
        )
    required_nodes = set(INPUT_IDS) | set(OUTPUT_IDS)
    for gene in child_connections.values():
        required_nodes.add(gene.source)
        required_nodes.add(gene.target)
    child_nodes: dict[int, NodeGene] = {}
    for node_id in required_nodes:
        source_node = fitter.nodes.get(node_id) or other.nodes[node_id]
        if node_id in fitter.nodes and node_id in other.nodes and rng.random() < 0.5:
            source_node = other.nodes[node_id]
        child_nodes[node_id] = NodeGene(source_node.node_id, source_node.kind, source_node.bias)
    return Genome(child_id, child_nodes, child_connections)


def speciate(
    population: list[Genome], previous: list[Species], next_species_id: int, threshold: float = 3.0
) -> tuple[list[Species], int]:
    species = [
        Species(group.species_id, group.representative, [], group.best_fitness, group.stagnant_generations)
        for group in previous
    ]
    for genome in population:
        placed = False
        for group in species:
            if compatibility_distance(genome, group.representative) < threshold:
                group.members.append(genome)
                placed = True
                break
        if not placed:
            species.append(Species(next_species_id, genome, [genome]))
            next_species_id += 1
    species = [group for group in species if group.members]
    for group in species:
        group.representative = random.choice(group.members)
        current_best = max(member.fitness for member in group.members)
        if current_best > group.best_fitness + 1e-9:
            group.best_fitness = current_best
            group.stagnant_generations = 0
        else:
            group.stagnant_generations += 1
    return species, next_species_id


def allocate_offspring(species: list[Species], population_size: int) -> dict[int, int]:
    for group in species:
        size = len(group.members)
        for member in group.members:
            member.adjusted_fitness = member.fitness / size
    totals = {
        group.species_id: sum(max(0.0, member.adjusted_fitness) for member in group.members)
        for group in species
    }
    grand_total = sum(totals.values())
    if grand_total <= 0:
        base = population_size // len(species)
        allocation = {group.species_id: base for group in species}
    else:
        raw = {
            group.species_id: population_size * totals[group.species_id] / grand_total
            for group in species
        }
        allocation = {species_id: int(value) for species_id, value in raw.items()}
        remainder = population_size - sum(allocation.values())
        order = sorted(raw, key=lambda species_id: raw[species_id] - allocation[species_id], reverse=True)
        for species_id in order[:remainder]:
            allocation[species_id] += 1
    return allocation


def reproduce(
    species: list[Species], tracker: InnovationTracker, next_genome_id: int, rng: random.Random
) -> tuple[list[Genome], int]:
    champion_fitness = max(group.best_fitness for group in species)
    viable = [
        group
        for group in species
        if group.stagnant_generations < 15 or group.best_fitness == champion_fitness
    ]
    allocation = allocate_offspring(viable, POPULATION_SIZE)
    children: list[Genome] = []
    for group in viable:
        members = sorted(group.members, key=lambda genome: genome.fitness, reverse=True)
        count = allocation.get(group.species_id, 0)
        if count <= 0:
            continue
        children.append(members[0].clone(next_genome_id))
        next_genome_id += 1
        count -= 1
        parent_pool = members[: max(1, math.ceil(len(members) * 0.25))]
        while count > 0:
            child = crossover(rng.choice(parent_pool), rng.choice(parent_pool), next_genome_id, rng)
            next_genome_id += 1
            mutate(child, tracker, rng)
            children.append(child)
            count -= 1
    while len(children) < POPULATION_SIZE:
        parent = max(
            (member for group in viable for member in group.members),
            key=lambda genome: genome.fitness,
        )
        child = parent.clone(next_genome_id)
        next_genome_id += 1
        mutate(child, tracker, rng)
        children.append(child)
    return children[:POPULATION_SIZE], next_genome_id


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(generations: int, seed: int) -> None:
    rng = random.Random(seed)
    weather = generate_synthetic_year(seed)
    tracker = InnovationTracker()
    next_genome_id = 0
    population: list[Genome] = []
    for _ in range(POPULATION_SIZE):
        population.append(initial_genome(next_genome_id, tracker, rng))
        next_genome_id += 1
    species: list[Species] = []
    next_species_id = 0
    best_ever: Genome | None = None
    best_evaluation: Evaluation | None = None

    for generation in range(generations):
        evaluations: dict[int, Evaluation] = {}
        for genome in population:
            traits = express_traits(genome)
            complexity = sum(1 for gene in genome.connections.values() if gene.enabled)
            evaluation, _ = evaluate_traits(traits, weather, complexity)
            genome.fitness = evaluation.fitness
            evaluations[genome.genome_id] = evaluation
        species, next_species_id = speciate(population, species, next_species_id)
        best = max(population, key=lambda genome: genome.fitness)
        current_evaluation = evaluations[best.genome_id]
        if best_ever is None or best.fitness > best_ever.fitness:
            best_ever = best.clone(best.genome_id)
            best_evaluation = current_evaluation
        average = sum(genome.fitness for genome in population) / len(population)
        hidden = sum(1 for node in best.nodes.values() if node.kind == "hidden")
        enabled = sum(1 for gene in best.connections.values() if gene.enabled)
        print(
            f"Generation {generation:03d} | avg={average:7.3f} | best={best.fitness:7.3f} | "
            f"species={len(species):2d} | nodes={len(best.nodes):2d} | hidden={hidden:2d} | "
            f"connections={enabled:2d} | adaptation={current_evaluation.overall_adaptation:.3f} | "
            f"seeds={current_evaluation.expected_seeds:.2f}"
        )
        population, next_genome_id = reproduce(species, tracker, next_genome_id, rng)

    assert best_ever is not None and best_evaluation is not None
    traits = express_traits(best_ever)
    final_evaluation, daily_rows = evaluate_traits(
        traits,
        weather,
        sum(1 for gene in best_ever.connections.values() if gene.enabled),
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    write_csv(OUTPUT_DIR / "synthetic_year.csv", [asdict(day) for day in weather])
    write_csv(OUTPUT_DIR / "winner_daily_adaptation.csv", daily_rows)
    network = {
        "nodes": [asdict(node) for node in sorted(best_ever.nodes.values(), key=lambda item: item.node_id)],
        "connections": [
            asdict(gene)
            for gene in sorted(best_ever.connections.values(), key=lambda item: item.innovation)
        ],
    }
    result = {
        "version": "0.5",
        "engine": "self-contained minimal NEAT",
        "environment": {"synthetic": True, "days": 365, "elevation_m": ELEVATION_M},
        "plant_genome": asdict(traits),
        "lifetime_scores": asdict(final_evaluation),
        "neat": {
            "generations": generations,
            "population_size": POPULATION_SIZE,
            "node_count": len(best_ever.nodes),
            "hidden_nodes": sum(1 for node in best_ever.nodes.values() if node.kind == "hidden"),
            "enabled_connections": sum(
                1 for gene in best_ever.connections.values() if gene.enabled
            ),
            "total_connections": len(best_ever.connections),
            "network": network,
        },
    }
    (OUTPUT_DIR / "winner.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nWinner")
    print(json.dumps(result, indent=2))
    print("\nFiles written to outputs/.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Madagascar synthetic plant V0.5 with self-contained NEAT"
    )
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.generations, arguments.seed)
