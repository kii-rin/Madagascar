from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

POPULATION_SIZE = 120
INPUT_IDS = (-4, -3, -2, -1)
OUTPUT_IDS = (0,)
MAX_STEPS = 500
TEST_EPISODES = 20


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
            new_id,
            {key: NodeGene(value.node_id, value.kind, value.bias) for key, value in self.nodes.items()},
            {
                key: ConnectionGene(
                    value.source,
                    value.target,
                    value.weight,
                    value.enabled,
                    value.innovation,
                )
                for key, value in self.connections.items()
            },
            self.fitness,
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
        self.next_node_id = 1
        self.connection_history: dict[tuple[int, int], int] = {}
        self.split_history: dict[int, tuple[int, int, int]] = {}

    def connection(self, source: int, target: int) -> int:
        key = (source, target)
        if key not in self.connection_history:
            self.connection_history[key] = self.next_innovation
            self.next_innovation += 1
        return self.connection_history[key]

    def split(self, innovation: int, source: int, target: int) -> tuple[int, int, int]:
        if innovation not in self.split_history:
            node_id = self.next_node_id
            self.next_node_id += 1
            self.split_history[innovation] = (
                node_id,
                self.connection(source, node_id),
                self.connection(node_id, target),
            )
        return self.split_history[innovation]


def sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


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
    indegree = {node_id: 0 for node_id in genome.nodes}
    outgoing: dict[int, list[int]] = {}
    for gene in genome.connections.values():
        if gene.enabled:
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


def activate(genome: Genome, inputs: tuple[float, float, float, float]) -> float:
    values = {node_id: value for node_id, value in zip(INPUT_IDS, inputs)}
    incoming: dict[int, list[ConnectionGene]] = {}
    for gene in genome.connections.values():
        if gene.enabled:
            incoming.setdefault(gene.target, []).append(gene)

    for node_id in topological_order(genome):
        node = genome.nodes[node_id]
        if node.kind == "input":
            continue
        total = node.bias + sum(
            values.get(gene.source, 0.0) * gene.weight
            for gene in incoming.get(node_id, [])
        )
        values[node_id] = sigmoid(total)
    return values.get(OUTPUT_IDS[0], 0.5)


def initial_genome(
    genome_id: int,
    tracker: InnovationTracker,
    rng: random.Random,
) -> Genome:
    nodes = {node_id: NodeGene(node_id, "input") for node_id in INPUT_IDS}
    nodes[OUTPUT_IDS[0]] = NodeGene(OUTPUT_IDS[0], "output", rng.gauss(0.0, 1.0))
    connections: dict[int, ConnectionGene] = {}
    for source in INPUT_IDS:
        innovation = tracker.connection(source, OUTPUT_IDS[0])
        connections[innovation] = ConnectionGene(
            source,
            OUTPUT_IDS[0],
            rng.gauss(0.0, 1.0),
            True,
            innovation,
        )
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


def mutate_add_connection(
    genome: Genome,
    tracker: InnovationTracker,
    rng: random.Random,
) -> None:
    sources = [node_id for node_id, node in genome.nodes.items() if node.kind != "output"]
    targets = [node_id for node_id, node in genome.nodes.items() if node.kind != "input"]
    existing = {(gene.source, gene.target) for gene in genome.connections.values()}
    candidates = [
        (source, target)
        for source in sources
        for target in targets
        if source != target
        and (source, target) not in existing
        and not creates_cycle(genome, source, target)
    ]
    if not candidates:
        return
    source, target = rng.choice(candidates)
    innovation = tracker.connection(source, target)
    genome.connections[innovation] = ConnectionGene(
        source,
        target,
        rng.gauss(0.0, 1.0),
        True,
        innovation,
    )


def mutate_add_node(
    genome: Genome,
    tracker: InnovationTracker,
    rng: random.Random,
) -> None:
    candidates = [gene for gene in genome.connections.values() if gene.enabled]
    if not candidates:
        return
    old = rng.choice(candidates)
    old.enabled = False
    node_id, first, second = tracker.split(old.innovation, old.source, old.target)
    genome.nodes.setdefault(node_id, NodeGene(node_id, "hidden", 0.0))
    genome.connections[first] = ConnectionGene(old.source, node_id, 1.0, True, first)
    genome.connections[second] = ConnectionGene(node_id, old.target, old.weight, True, second)


def mutate(
    genome: Genome,
    tracker: InnovationTracker,
    rng: random.Random,
) -> None:
    mutate_weights(genome, rng)
    if rng.random() < 0.12:
        mutate_add_connection(genome, tracker, rng)
    if rng.random() < 0.06:
        mutate_add_node(genome, tracker, rng)
    if rng.random() < 0.01 and genome.connections:
        gene = rng.choice(list(genome.connections.values()))
        if gene.enabled:
            gene.enabled = False
        elif not creates_cycle(genome, gene.source, gene.target):
            gene.enabled = True


def compatibility_distance(first: Genome, second: Genome) -> float:
    first_ids = set(first.connections)
    second_ids = set(second.connections)
    matching = first_ids & second_ids
    first_max = max(first_ids, default=-1)
    second_max = max(second_ids, default=-1)
    excess = 0
    disjoint = 0

    for innovation in first_ids ^ second_ids:
        if innovation > min(first_max, second_max):
            excess += 1
        else:
            disjoint += 1

    weight_difference = (
        sum(
            abs(first.connections[index].weight - second.connections[index].weight)
            for index in matching
        )
        / len(matching)
        if matching
        else 0.0
    )
    normalizer = max(len(first_ids), len(second_ids))
    if normalizer < 20:
        normalizer = 1
    return excess / normalizer + disjoint / normalizer + 0.4 * weight_difference


def crossover(
    fitter: Genome,
    other: Genome,
    child_id: int,
    rng: random.Random,
) -> Genome:
    if other.fitness > fitter.fitness:
        fitter, other = other, fitter
    equal = math.isclose(fitter.fitness, other.fitness)
    child_connections: dict[int, ConnectionGene] = {}

    for innovation, fitter_gene in fitter.connections.items():
        other_gene = other.connections.get(innovation)
        chosen = other_gene if other_gene and rng.random() < 0.5 else fitter_gene
        if equal and innovation not in other.connections and rng.random() < 0.5:
            continue
        enabled = chosen.enabled
        if other_gene and (not fitter_gene.enabled or not other_gene.enabled):
            enabled = rng.random() >= 0.75
        child_connections[innovation] = ConnectionGene(
            chosen.source,
            chosen.target,
            chosen.weight,
            enabled,
            innovation,
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
        child_nodes[node_id] = NodeGene(
            source_node.node_id,
            source_node.kind,
            source_node.bias,
        )
    return Genome(child_id, child_nodes, child_connections)


def speciate(
    population: list[Genome],
    previous: list[Species],
    next_species_id: int,
    rng: random.Random,
    threshold: float = 3.0,
) -> tuple[list[Species], int]:
    groups = [
        Species(
            group.species_id,
            group.representative,
            [],
            group.best_fitness,
            group.stagnant_generations,
        )
        for group in previous
    ]

    for genome in population:
        for group in groups:
            if compatibility_distance(genome, group.representative) < threshold:
                group.members.append(genome)
                break
        else:
            groups.append(Species(next_species_id, genome, [genome]))
            next_species_id += 1

    groups = [group for group in groups if group.members]
    for group in groups:
        group.representative = rng.choice(group.members)
        current_best = max(member.fitness for member in group.members)
        if current_best > group.best_fitness + 1e-9:
            group.best_fitness = current_best
            group.stagnant_generations = 0
        else:
            group.stagnant_generations += 1
    return groups, next_species_id


def reproduce(
    groups: list[Species],
    tracker: InnovationTracker,
    next_genome_id: int,
    rng: random.Random,
) -> tuple[list[Genome], int]:
    champion_fitness = max(group.best_fitness for group in groups)
    viable = [
        group
        for group in groups
        if group.stagnant_generations < 15 or group.best_fitness == champion_fitness
    ]

    for group in viable:
        for genome in group.members:
            genome.adjusted_fitness = max(0.0, genome.fitness) / len(group.members)

    totals = {
        group.species_id: sum(member.adjusted_fitness for member in group.members)
        for group in viable
    }
    grand_total = sum(totals.values())
    if grand_total:
        raw = {
            group.species_id: POPULATION_SIZE * totals[group.species_id] / grand_total
            for group in viable
        }
    else:
        raw = {
            group.species_id: POPULATION_SIZE / len(viable)
            for group in viable
        }

    allocation = {species_id: int(value) for species_id, value in raw.items()}
    remainder = POPULATION_SIZE - sum(allocation.values())
    order = sorted(
        raw,
        key=lambda species_id: raw[species_id] - allocation[species_id],
        reverse=True,
    )
    for species_id in order[:remainder]:
        allocation[species_id] += 1

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
        while count:
            child = crossover(
                rng.choice(parent_pool),
                rng.choice(parent_pool),
                next_genome_id,
                rng,
            )
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


def cartpole_episode(genome: Genome, seed: int) -> int:
    rng = random.Random(seed)
    position = rng.uniform(-0.05, 0.05)
    velocity = rng.uniform(-0.05, 0.05)
    angle = rng.uniform(-0.05, 0.05)
    angular_velocity = rng.uniform(-0.05, 0.05)

    gravity = 9.8
    cart_mass = 1.0
    pole_mass = 0.1
    total_mass = cart_mass + pole_mass
    half_pole_length = 0.5
    pole_mass_length = pole_mass * half_pole_length
    force_magnitude = 10.0
    timestep = 0.02
    angle_limit = 12.0 * math.pi / 180.0

    for step in range(MAX_STEPS):
        inputs = (
            position / 2.4,
            velocity / 3.0,
            angle / angle_limit,
            angular_velocity / 3.5,
        )
        force = force_magnitude if activate(genome, inputs) >= 0.5 else -force_magnitude

        cosine = math.cos(angle)
        sine = math.sin(angle)
        temporary = (
            force + pole_mass_length * angular_velocity**2 * sine
        ) / total_mass
        angle_acceleration = (
            gravity * sine - cosine * temporary
        ) / (
            half_pole_length
            * (4.0 / 3.0 - pole_mass * cosine**2 / total_mass)
        )
        position_acceleration = (
            temporary
            - pole_mass_length * angle_acceleration * cosine / total_mass
        )

        position += timestep * velocity
        velocity += timestep * position_acceleration
        angle += timestep * angular_velocity
        angular_velocity += timestep * angle_acceleration

        if abs(position) > 2.4 or abs(angle) > angle_limit:
            return step + 1

    return MAX_STEPS


def evaluate_cartpole(genome: Genome) -> float:
    scores = [cartpole_episode(genome, seed) for seed in range(TEST_EPISODES)]
    return float(min(scores))


def run(generations: int, seed: int) -> None:
    rng = random.Random(seed)
    tracker = InnovationTracker()
    population: list[Genome] = []
    next_genome_id = 0

    for _ in range(POPULATION_SIZE):
        population.append(initial_genome(next_genome_id, tracker, rng))
        next_genome_id += 1

    species: list[Species] = []
    next_species_id = 0
    best_ever: Genome | None = None

    for generation in range(generations):
        for genome in population:
            genome.fitness = evaluate_cartpole(genome)

        species, next_species_id = speciate(
            population,
            species,
            next_species_id,
            rng,
        )
        best = max(population, key=lambda genome: genome.fitness)
        if best_ever is None or best.fitness > best_ever.fitness:
            best_ever = best.clone(best.genome_id)

        average = sum(genome.fitness for genome in population) / len(population)
        hidden = sum(node.kind == "hidden" for node in best.nodes.values())
        enabled = sum(gene.enabled for gene in best.connections.values())
        print(
            f"Generation {generation:03d} | "
            f"avg={average:7.2f} | best={best.fitness:7.2f} | "
            f"species={len(species):2d} | hidden={hidden:2d} | "
            f"connections={enabled:2d}"
        )

        if best.fitness >= MAX_STEPS:
            print(f"SOLVED at generation {generation}")
            break

        population, next_genome_id = reproduce(
            species,
            tracker,
            next_genome_id,
            rng,
        )

    assert best_ever is not None
    result = {
        "benchmark": "CartPole-v1 compatible physics",
        "solved": best_ever.fitness >= MAX_STEPS,
        "best_fitness": best_ever.fitness,
        "population_size": POPULATION_SIZE,
        "test_episodes": TEST_EPISODES,
        "max_steps": MAX_STEPS,
        "nodes": [
            asdict(node)
            for node in sorted(best_ever.nodes.values(), key=lambda item: item.node_id)
        ],
        "connections": [
            asdict(gene)
            for gene in sorted(
                best_ever.connections.values(),
                key=lambda item: item.innovation,
            )
        ],
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/cartpole_winner.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    summary = {
        "solved": result["solved"],
        "best_fitness": result["best_fitness"],
        "hidden_nodes": sum(node["kind"] == "hidden" for node in result["nodes"]),
        "enabled_connections": sum(
            connection["enabled"] for connection in result["connections"]
        ),
    }
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dependency-free CartPole test for the Madagascar NEAT engine"
    )
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.generations, arguments.seed)
