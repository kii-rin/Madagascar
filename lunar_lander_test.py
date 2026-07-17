from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cartpole_test as neat

POPULATION_SIZE = 80
INPUT_IDS = tuple(range(-8, 0))
OUTPUT_IDS = (0, 1, 2, 3)
MAX_STEPS = 400
DT = 0.05
TRAIN_EPISODES = 6
TEST_EPISODES = 40

# Reuse the same tested NEAT machinery from cartpole_test.py.
neat.POPULATION_SIZE = POPULATION_SIZE
neat.INPUT_IDS = INPUT_IDS
neat.OUTPUT_IDS = OUTPUT_IDS


@dataclass
class EpisodeResult:
    score: float
    landed: bool
    crashed: bool
    steps: int
    fuel: float


def initial_genome(genome_id, tracker, rng):
    nodes = {node_id: neat.NodeGene(node_id, "input") for node_id in INPUT_IDS}
    nodes.update(
        {
            node_id: neat.NodeGene(node_id, "output", rng.gauss(0.0, 1.0))
            for node_id in OUTPUT_IDS
        }
    )
    connections = {}
    for source in INPUT_IDS:
        for target in OUTPUT_IDS:
            innovation = tracker.connection(source, target)
            connections[innovation] = neat.ConnectionGene(
                source, target, rng.gauss(0.0, 1.0), True, innovation
            )
    return neat.Genome(genome_id, nodes, connections)


def activate(genome, inputs):
    values = {node_id: value for node_id, value in zip(INPUT_IDS, inputs)}
    incoming = {}
    for gene in genome.connections.values():
        if gene.enabled:
            incoming.setdefault(gene.target, []).append(gene)

    for node_id in neat.topological_order(genome):
        node = genome.nodes[node_id]
        if node.kind == "input":
            continue
        total = node.bias + sum(
            values.get(gene.source, 0.0) * gene.weight
            for gene in incoming.get(node_id, [])
        )
        values[node_id] = neat.sigmoid(total)
    return [values.get(node_id, 0.5) for node_id in OUTPUT_IDS]


def lunar_episode(genome, seed):
    rng = random.Random(seed)
    x = rng.uniform(-0.8, 0.8)
    y = rng.uniform(1.0, 1.5)
    vx = rng.uniform(-0.15, 0.15)
    vy = rng.uniform(-0.10, 0.05)
    angle = rng.uniform(-0.18, 0.18)
    angular_velocity = rng.uniform(-0.08, 0.08)
    fuel = 1.0
    previous_distance = math.hypot(x, y)
    score = 0.0

    for step in range(MAX_STEPS):
        inputs = (
            x / 2.0,
            y / 1.5,
            vx,
            vy,
            angle / math.pi,
            angular_velocity,
            fuel,
            1.0,
        )
        outputs = activate(genome, inputs)
        action = max(range(4), key=lambda index: outputs[index])

        main_thrust = 0.0
        side_thrust = 0.0
        if fuel > 0.0:
            if action == 1:
                main_thrust = 1.0
                fuel = max(0.0, fuel - 0.0025)
            elif action == 2:
                side_thrust = -1.0
                fuel = max(0.0, fuel - 0.001)
            elif action == 3:
                side_thrust = 1.0
                fuel = max(0.0, fuel - 0.001)

        ax = side_thrust * 0.9 + math.sin(angle) * main_thrust * 0.35
        ay = -0.55 + main_thrust * 1.25 * math.cos(angle)
        angular_acceleration = -side_thrust * 0.9 - angle * 0.03

        vx += ax * DT
        vy += ay * DT
        angular_velocity += angular_acceleration * DT
        x += vx * DT
        y += vy * DT
        angle += angular_velocity * DT

        distance = math.hypot(x, y)
        score += (previous_distance - distance) * 12.0 - 0.002
        previous_distance = distance

        if abs(x) > 2.2 or y > 2.2:
            return EpisodeResult(score - 120.0, False, True, step + 1, fuel)

        if y <= 0.0:
            safe = (
                abs(x) < 0.18
                and abs(vx) < 0.22
                and abs(vy) < 0.28
                and abs(angle) < 0.18
                and abs(angular_velocity) < 0.35
            )
            if safe:
                score += 220.0 + 40.0 * fuel - 30.0 * abs(x) - 20.0 * abs(angle)
                return EpisodeResult(score, True, False, step + 1, fuel)
            score -= 100.0 + 30.0 * abs(vy) + 20.0 * abs(angle) + 10.0 * abs(x)
            return EpisodeResult(score, False, True, step + 1, fuel)

    score -= 40.0 * math.hypot(x, y) + 15.0 * abs(angle)
    return EpisodeResult(score, False, False, MAX_STEPS, fuel)


def evaluate(genome, seeds):
    results = [lunar_episode(genome, episode_seed) for episode_seed in seeds]
    landings = sum(result.landed for result in results)
    average_score = sum(result.score for result in results) / len(results)
    genome.fitness = average_score + landings * 30.0
    return landings, results


def run(generations, seed):
    rng = random.Random(seed)
    tracker = neat.InnovationTracker()
    tracker.next_node_id = max(OUTPUT_IDS) + 1

    population = []
    next_genome_id = 0
    for _ in range(POPULATION_SIZE):
        population.append(initial_genome(next_genome_id, tracker, rng))
        next_genome_id += 1

    groups = []
    next_species_id = 0
    best_ever = None
    best_landings = 0
    training_seeds = list(range(TRAIN_EPISODES))

    for generation in range(generations):
        metadata = {}
        for genome in population:
            metadata[genome.genome_id] = evaluate(genome, training_seeds)

        groups, next_species_id = neat.speciate(
            population, groups, next_species_id, rng
        )
        best = max(population, key=lambda genome: genome.fitness)
        landings, _ = metadata[best.genome_id]
        if best_ever is None or best.fitness > best_ever.fitness:
            best_ever = best.clone(best.genome_id)
            best_landings = landings

        hidden = sum(node.kind == "hidden" for node in best.nodes.values())
        enabled = sum(gene.enabled for gene in best.connections.values())
        average = sum(genome.fitness for genome in population) / len(population)
        print(
            f"Generation {generation:03d} | avg={average:8.2f} | "
            f"best={best.fitness:8.2f} | landed={landings:2d}/{TRAIN_EPISODES} | "
            f"species={len(groups):2d} | hidden={hidden:2d} | connections={enabled:2d}"
        )

        if landings == TRAIN_EPISODES:
            print(f"SOLVED at generation {generation}")
            break

        population, next_genome_id = neat.reproduce(
            groups, tracker, next_genome_id, rng
        )

    assert best_ever is not None
    test_seeds = list(range(100, 100 + TEST_EPISODES))
    test_landings, test_results = evaluate(best_ever, test_seeds)
    result = {
        "benchmark": "dependency-free simplified Lunar Lander",
        "solved_training": best_landings == TRAIN_EPISODES,
        "training_landings": best_landings,
        "training_episodes": TRAIN_EPISODES,
        "test_landings": test_landings,
        "test_episodes": TEST_EPISODES,
        "test_fitness": best_ever.fitness,
        "average_test_fuel": sum(item.fuel for item in test_results) / TEST_EPISODES,
        "hidden_nodes": sum(
            node.kind == "hidden" for node in best_ever.nodes.values()
        ),
        "enabled_connections": sum(
            gene.enabled for gene in best_ever.connections.values()
        ),
        "nodes": [asdict(node) for node in best_ever.nodes.values()],
        "connections": [asdict(gene) for gene in best_ever.connections.values()],
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/lunar_lander_winner.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.generations, args.seed)


if __name__ == "__main__":
    main()
