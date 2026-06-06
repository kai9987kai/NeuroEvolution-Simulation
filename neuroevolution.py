from __future__ import annotations

import ast
import gzip
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


INPUT_LABELS = (
    "food forward",
    "food side",
    "food proximity",
    "energy",
    "speed",
    "center forward",
    "center side",
    "wall clearance",
)
OUTPUT_LABELS = ("turn", "thrust")


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def angle_delta(target: float, source: float) -> float:
    return (target - source + math.pi) % math.tau - math.pi


def _deep_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_deep_tuple(item) for item in value)
    return value


@dataclass(slots=True)
class SimulationConfig:
    width: int = 820
    height: int = 680
    population_size: int = 72
    food_count: int = 90
    generation_steps: int = 450
    hidden_size: int = 10
    archive_bins: int = 10
    descriptor_grid: int = 12
    elite_count: int = 4
    dns_neighbors: int = 5
    mutation_rate: float = 0.16
    base_mutation_sigma: float = 0.20
    max_mutation_sigma: float = 0.85
    crossover_rate: float = 0.75
    archive_parent_rate: float = 0.25
    immigrant_rate: float = 0.04
    max_speed: float = 3.5
    acceleration: float = 0.28
    turn_rate: float = 0.24
    initial_energy: float = 1.0
    food_energy: float = 0.48
    food_radius: float = 4.0
    agent_radius: float = 5.0

    def validate(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if self.food_count < 1:
            raise ValueError("food_count must be positive")
        if self.generation_steps < 1:
            raise ValueError("generation_steps must be positive")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.archive_bins < 2:
            raise ValueError("archive_bins must be at least 2")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1")
        if self.base_mutation_sigma <= 0.0:
            raise ValueError("base_mutation_sigma must be positive")


@dataclass(slots=True)
class Genome:
    weights: list[float]

    @staticmethod
    def weight_count(hidden_size: int) -> int:
        return (len(INPUT_LABELS) + 1) * hidden_size + (hidden_size + 1) * len(
            OUTPUT_LABELS
        )

    @classmethod
    def random(cls, rng: random.Random, hidden_size: int) -> Genome:
        scale = math.sqrt(2.0 / (len(INPUT_LABELS) + hidden_size))
        return cls(
            [
                rng.gauss(0.0, scale)
                for _ in range(cls.weight_count(hidden_size))
            ]
        )

    def copy(self) -> Genome:
        return Genome(self.weights.copy())

    def activate(self, inputs: Iterable[float], hidden_size: int) -> tuple[float, float]:
        input_values = tuple(inputs)
        if len(input_values) != len(INPUT_LABELS):
            raise ValueError(f"expected {len(INPUT_LABELS)} neural inputs")

        cursor = 0
        hidden: list[float] = []
        for _ in range(hidden_size):
            total = self.weights[cursor]
            cursor += 1
            for input_value in input_values:
                total += input_value * self.weights[cursor]
                cursor += 1
            hidden.append(math.tanh(total))

        outputs: list[float] = []
        for _ in OUTPUT_LABELS:
            total = self.weights[cursor]
            cursor += 1
            for hidden_value in hidden:
                total += hidden_value * self.weights[cursor]
                cursor += 1
            outputs.append(math.tanh(total))
        return outputs[0], outputs[1]

    def crossover(self, other: Genome, rng: random.Random) -> Genome:
        if len(self.weights) != len(other.weights):
            raise ValueError("genomes must use the same network topology")
        child = []
        for left, right in zip(self.weights, other.weights):
            roll = rng.random()
            if roll < 0.45:
                child.append(left)
            elif roll < 0.90:
                child.append(right)
            else:
                child.append((left + right) * 0.5)
        return Genome(child)

    def mutate(self, rng: random.Random, rate: float, sigma: float) -> Genome:
        weights = self.weights.copy()
        changed = False
        for index, value in enumerate(weights):
            if rng.random() < rate:
                weights[index] = clamp(value + rng.gauss(0.0, sigma), -5.0, 5.0)
                changed = True
        if not changed:
            index = rng.randrange(len(weights))
            weights[index] = clamp(
                weights[index] + rng.gauss(0.0, sigma), -5.0, 5.0
            )
        return Genome(weights)


@dataclass(slots=True)
class Agent:
    genome: Genome
    x: float
    y: float
    heading: float
    energy: float
    speed: float = 0.0
    alive: bool = True
    age: int = 0
    food_eaten: int = 0
    distance_travelled: float = 0.0
    wall_hits: int = 0
    turn_activity: float = 0.0
    visited: set[tuple[int, int]] = field(default_factory=set)
    fitness: float = 0.0
    descriptor: tuple[float, float] = (0.0, 0.0)
    dns_score: float = 0.0

    def provisional_fitness(self, config: SimulationConfig) -> float:
        survival = self.age / config.generation_steps
        return (
            self.food_eaten * 35.0
            + survival * 5.0
            + self.energy * 4.0
            + self.distance_travelled * 0.008
            - self.wall_hits * 0.35
        )

    def finalize(self, config: SimulationConfig) -> None:
        self.fitness = self.provisional_fitness(config)
        grid_cells = config.descriptor_grid * config.descriptor_grid
        roaming = clamp(len(self.visited) / max(1, grid_cells * 0.45), 0.0, 1.0)
        movement_style = clamp(
            self.distance_travelled
            / max(1.0, config.max_speed * config.generation_steps),
            0.0,
            1.0,
        )
        self.descriptor = (roaming, movement_style)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "energy": self.energy,
            "speed": self.speed,
            "alive": self.alive,
            "age": self.age,
            "food_eaten": self.food_eaten,
            "distance_travelled": self.distance_travelled,
            "wall_hits": self.wall_hits,
            "turn_activity": self.turn_activity,
            "visited": [list(cell) for cell in sorted(self.visited)],
            "fitness": self.fitness,
            "descriptor": list(self.descriptor),
            "dns_score": self.dns_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], genome: Genome) -> Agent:
        return cls(
            genome=genome,
            x=float(data["x"]),
            y=float(data["y"]),
            heading=float(data["heading"]),
            energy=float(data["energy"]),
            speed=float(data["speed"]),
            alive=bool(data["alive"]),
            age=int(data["age"]),
            food_eaten=int(data["food_eaten"]),
            distance_travelled=float(data["distance_travelled"]),
            wall_hits=int(data["wall_hits"]),
            turn_activity=float(data.get("turn_activity", 0.0)),
            visited={tuple(cell) for cell in data["visited"]},
            fitness=float(data["fitness"]),
            descriptor=tuple(data["descriptor"]),
            dns_score=float(data["dns_score"]),
        )


@dataclass(slots=True)
class ArchiveEntry:
    genome: Genome
    fitness: float
    descriptor: tuple[float, float]
    generation: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.genome.weights,
            "fitness": self.fitness,
            "descriptor": list(self.descriptor),
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchiveEntry:
        return cls(
            genome=Genome([float(value) for value in data["weights"]]),
            fitness=float(data["fitness"]),
            descriptor=tuple(data["descriptor"]),
            generation=int(data["generation"]),
        )


def dominated_novelty_scores(
    fitnesses: list[float],
    descriptors: list[tuple[float, float]],
    neighbors: int = 5,
) -> list[float]:
    """Distance to the nearest fitter behaviors, as proposed by DNS."""
    if len(fitnesses) != len(descriptors):
        raise ValueError("fitness and descriptor lengths differ")
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    if not fitnesses:
        return []

    scores: list[float] = []
    for index, (fitness, descriptor) in enumerate(zip(fitnesses, descriptors)):
        distances = []
        for other_index, (other_fitness, other_descriptor) in enumerate(
            zip(fitnesses, descriptors)
        ):
            if other_index == index or other_fitness <= fitness:
                continue
            distances.append(math.dist(descriptor, other_descriptor))
        if not distances:
            scores.append(math.inf)
            continue
        distances.sort()
        local = distances[:neighbors]
        scores.append(sum(local) / len(local))
    return scores


class EvolutionSimulation:
    STATE_VERSION = 1

    def __init__(self, config: SimulationConfig | None = None, seed: int = 7):
        self.config = config or SimulationConfig()
        self.config.validate()
        self.seed = seed
        self.rng = random.Random(seed)
        self.generation = 1
        self.step_count = 0
        self.stagnation = 0
        self.mutation_sigma = self.config.base_mutation_sigma
        self.best_ever_fitness: float | None = None
        self.best_ever_genome: Genome | None = None
        self.history: list[dict[str, float | int]] = []
        self.archive: dict[tuple[int, int], ArchiveEntry] = {}
        self.genomes = [
            Genome.random(self.rng, self.config.hidden_size)
            for _ in range(self.config.population_size)
        ]
        self.foods: list[tuple[float, float]] = []
        self.agents: list[Agent] = []
        self._reset_world()

    @property
    def archive_coverage(self) -> float:
        return len(self.archive) / (self.config.archive_bins**2)

    @property
    def alive_count(self) -> int:
        return sum(agent.alive for agent in self.agents)

    def _random_position(self, margin: float = 12.0) -> tuple[float, float]:
        return (
            self.rng.uniform(margin, self.config.width - margin),
            self.rng.uniform(margin, self.config.height - margin),
        )

    def _reset_world(self) -> None:
        self.foods = [self._random_position() for _ in range(self.config.food_count)]
        self.agents = []
        for genome in self.genomes:
            x, y = self._random_position(20.0)
            self.agents.append(
                Agent(
                    genome=genome,
                    x=x,
                    y=y,
                    heading=self.rng.uniform(0.0, math.tau),
                    energy=self.config.initial_energy,
                )
            )
        self.step_count = 0

    def _nearest_food(self, agent: Agent) -> tuple[int, float, float, float]:
        best_index = 0
        food_x, food_y = self.foods[0]
        best_dx = food_x - agent.x
        best_dy = food_y - agent.y
        best_distance_sq = best_dx * best_dx + best_dy * best_dy
        for index in range(1, len(self.foods)):
            food_x, food_y = self.foods[index]
            dx = food_x - agent.x
            dy = food_y - agent.y
            distance_sq = dx * dx + dy * dy
            if distance_sq < best_distance_sq:
                best_index = index
                best_dx = dx
                best_dy = dy
                best_distance_sq = distance_sq
        return best_index, best_dx, best_dy, math.sqrt(best_distance_sq)

    def _sensors(
        self, agent: Agent, food_dx: float, food_dy: float, food_distance: float
    ) -> tuple[float, ...]:
        config = self.config
        diagonal = math.hypot(config.width, config.height)
        food_angle = math.atan2(food_dy, food_dx)
        relative_food = angle_delta(food_angle, agent.heading)
        food_signal = 1.0 - clamp(food_distance / diagonal, 0.0, 1.0)

        center_dx = config.width * 0.5 - agent.x
        center_dy = config.height * 0.5 - agent.y
        center_angle = math.atan2(center_dy, center_dx)
        relative_center = angle_delta(center_angle, agent.heading)

        wall_distance = min(
            agent.x, config.width - agent.x, agent.y, config.height - agent.y
        )
        wall_clearance = clamp(
            wall_distance / (min(config.width, config.height) * 0.5), 0.0, 1.0
        )
        return (
            math.cos(relative_food) * food_signal,
            math.sin(relative_food) * food_signal,
            food_signal * 2.0 - 1.0,
            clamp(agent.energy / config.initial_energy, 0.0, 1.5) - 0.5,
            (agent.speed / config.max_speed) * 2.0 - 1.0,
            math.cos(relative_center),
            math.sin(relative_center),
            wall_clearance * 2.0 - 1.0,
        )

    def _move_agent(self, agent: Agent) -> None:
        config = self.config
        food_index, food_dx, food_dy, food_distance = self._nearest_food(agent)
        turn, thrust = agent.genome.activate(
            self._sensors(agent, food_dx, food_dy, food_distance),
            config.hidden_size,
        )

        agent.heading = (agent.heading + turn * config.turn_rate) % math.tau
        agent.speed = clamp(
            agent.speed + thrust * config.acceleration, 0.0, config.max_speed
        )
        old_x, old_y = agent.x, agent.y
        agent.x += math.cos(agent.heading) * agent.speed
        agent.y += math.sin(agent.heading) * agent.speed

        radius = config.agent_radius
        if agent.x < radius or agent.x > config.width - radius:
            agent.x = clamp(agent.x, radius, config.width - radius)
            agent.heading = (math.pi - agent.heading) % math.tau
            agent.speed *= 0.55
            agent.wall_hits += 1
        if agent.y < radius or agent.y > config.height - radius:
            agent.y = clamp(agent.y, radius, config.height - radius)
            agent.heading = (-agent.heading) % math.tau
            agent.speed *= 0.55
            agent.wall_hits += 1

        agent.distance_travelled += math.hypot(agent.x - old_x, agent.y - old_y)
        agent.turn_activity += abs(turn)
        agent.age += 1
        drain = 0.00135 + max(0.0, thrust) * 0.00105 + agent.speed * 0.00010
        agent.energy -= drain

        cell_x = min(
            config.descriptor_grid - 1,
            int(agent.x / config.width * config.descriptor_grid),
        )
        cell_y = min(
            config.descriptor_grid - 1,
            int(agent.y / config.height * config.descriptor_grid),
        )
        agent.visited.add((cell_x, cell_y))

        food_x, food_y = self.foods[food_index]
        if math.hypot(food_x - agent.x, food_y - agent.y) <= (
            config.food_radius + config.agent_radius
        ):
            agent.food_eaten += 1
            agent.energy = min(1.6 * config.initial_energy, agent.energy + config.food_energy)
            self.foods[food_index] = self._random_position()

        if agent.energy <= 0.0:
            agent.energy = 0.0
            agent.alive = False
            agent.speed = 0.0

    def step(self) -> dict[str, float | int] | None:
        for agent in self.agents:
            if agent.alive:
                self._move_agent(agent)
        self.step_count += 1
        if self.step_count >= self.config.generation_steps or self.alive_count == 0:
            return self.evolve()
        return None

    def _archive_cell(self, descriptor: tuple[float, float]) -> tuple[int, int]:
        bins = self.config.archive_bins
        return (
            min(bins - 1, max(0, int(descriptor[0] * bins))),
            min(bins - 1, max(0, int(descriptor[1] * bins))),
        )

    def _update_archive(self) -> None:
        for agent in self.agents:
            cell = self._archive_cell(agent.descriptor)
            incumbent = self.archive.get(cell)
            if incumbent is None or agent.fitness > incumbent.fitness:
                self.archive[cell] = ArchiveEntry(
                    genome=agent.genome.copy(),
                    fitness=agent.fitness,
                    descriptor=agent.descriptor,
                    generation=self.generation,
                )

    def _select_parent(self, survivors: list[Agent]) -> Genome:
        if self.archive and self.rng.random() < self.config.archive_parent_rate:
            return self.rng.choice(list(self.archive.values())).genome
        return self.rng.choice(survivors).genome

    def _next_generation(self) -> list[Genome]:
        ranked_by_fitness = sorted(
            self.agents, key=lambda agent: agent.fitness, reverse=True
        )
        ranked_by_dns = sorted(
            self.agents, key=lambda agent: agent.dns_score, reverse=True
        )
        survivor_count = max(2, math.ceil(len(ranked_by_dns) / 2))
        survivors = ranked_by_dns[:survivor_count]
        next_genomes = [
            agent.genome.copy()
            for agent in ranked_by_fitness[: self.config.elite_count]
        ]

        while len(next_genomes) < self.config.population_size:
            if self.rng.random() < self.config.immigrant_rate:
                next_genomes.append(Genome.random(self.rng, self.config.hidden_size))
                continue
            parent_a = self._select_parent(survivors)
            if self.rng.random() < self.config.crossover_rate:
                parent_b = self._select_parent(survivors)
                child = parent_a.crossover(parent_b, self.rng)
            else:
                child = parent_a.copy()
            next_genomes.append(
                child.mutate(
                    self.rng, self.config.mutation_rate, self.mutation_sigma
                )
            )
        return next_genomes

    def evolve(self) -> dict[str, float | int]:
        for agent in self.agents:
            agent.finalize(self.config)

        fitnesses = [agent.fitness for agent in self.agents]
        descriptors = [agent.descriptor for agent in self.agents]
        scores = dominated_novelty_scores(
            fitnesses, descriptors, self.config.dns_neighbors
        )
        for agent, score in zip(self.agents, scores):
            agent.dns_score = score

        self._update_archive()
        best_agent = max(self.agents, key=lambda agent: agent.fitness)
        mean_fitness = sum(fitnesses) / len(fitnesses)
        mean_food = sum(agent.food_eaten for agent in self.agents) / len(self.agents)

        improved = (
            self.best_ever_fitness is None
            or best_agent.fitness > self.best_ever_fitness + 1e-9
        )
        if improved:
            self.best_ever_fitness = best_agent.fitness
            self.best_ever_genome = best_agent.genome.copy()
            self.stagnation = 0
            self.mutation_sigma = max(
                self.config.base_mutation_sigma, self.mutation_sigma * 0.90
            )
        else:
            self.stagnation += 1
            self.mutation_sigma = min(
                self.config.max_mutation_sigma,
                self.config.base_mutation_sigma * (1.0 + self.stagnation * 0.16),
            )

        report: dict[str, float | int] = {
            "generation": self.generation,
            "best_fitness": best_agent.fitness,
            "mean_fitness": mean_fitness,
            "best_food": best_agent.food_eaten,
            "mean_food": mean_food,
            "archive_size": len(self.archive),
            "archive_coverage": self.archive_coverage,
            "mutation_sigma": self.mutation_sigma,
            "stagnation": self.stagnation,
        }
        self.history.append(report)
        self.genomes = self._next_generation()
        self.generation += 1
        self._reset_world()
        return report

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": self.STATE_VERSION,
            "config": asdict(self.config),
            "seed": self.seed,
            "generation": self.generation,
            "step_count": self.step_count,
            "stagnation": self.stagnation,
            "mutation_sigma": self.mutation_sigma,
            "best_ever_fitness": self.best_ever_fitness,
            "best_ever_weights": (
                self.best_ever_genome.weights if self.best_ever_genome else None
            ),
            "history": self.history,
            "archive": {
                f"{cell[0]},{cell[1]}": entry.to_dict()
                for cell, entry in self.archive.items()
            },
            "genomes": [genome.weights for genome in self.genomes],
            "agents": [agent.to_dict() for agent in self.agents],
            "foods": [list(food) for food in self.foods],
            "random_state": repr(self.rng.getstate()),
        }
        with gzip.open(destination, "wt", encoding="utf-8") as handle:
            json.dump(state, handle, separators=(",", ":"))

    @classmethod
    def load(cls, path: str | Path) -> EvolutionSimulation:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("version") != cls.STATE_VERSION:
            raise ValueError("unsupported simulation state version")

        config = SimulationConfig(**state["config"])
        simulation = cls(config=config, seed=int(state["seed"]))
        simulation.generation = int(state["generation"])
        simulation.step_count = int(state["step_count"])
        simulation.stagnation = int(state["stagnation"])
        simulation.mutation_sigma = float(state["mutation_sigma"])
        simulation.best_ever_fitness = state["best_ever_fitness"]
        best_weights = state["best_ever_weights"]
        simulation.best_ever_genome = (
            Genome([float(value) for value in best_weights])
            if best_weights is not None
            else None
        )
        simulation.history = list(state["history"])
        simulation.archive = {}
        for key, entry_data in state["archive"].items():
            x, y = (int(value) for value in key.split(","))
            simulation.archive[(x, y)] = ArchiveEntry.from_dict(entry_data)
        simulation.genomes = [
            Genome([float(value) for value in weights])
            for weights in state["genomes"]
        ]
        simulation.agents = [
            Agent.from_dict(agent_data, genome)
            for agent_data, genome in zip(state["agents"], simulation.genomes)
        ]
        simulation.foods = [tuple(food) for food in state["foods"]]
        random_state = ast.literal_eval(state["random_state"])
        simulation.rng.setstate(_deep_tuple(random_state))
        return simulation
