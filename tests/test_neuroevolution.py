import math
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution import (
    EvolutionSimulation,
    Genome,
    SimulationConfig,
    dominated_novelty_scores,
)


class GenomeTests(unittest.TestCase):
    def test_activation_is_deterministic_and_bounded(self) -> None:
        rng = random.Random(4)
        genome = Genome.random(rng, hidden_size=5)
        inputs = [0.25] * 8

        first = genome.activate(inputs, hidden_size=5)
        second = genome.activate(inputs, hidden_size=5)

        self.assertEqual(first, second)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in first))

    def test_mutation_always_changes_at_least_one_weight(self) -> None:
        rng = random.Random(9)
        genome = Genome([0.0] * Genome.weight_count(hidden_size=3))

        mutated = genome.mutate(rng, rate=0.0, sigma=0.2)

        self.assertNotEqual(genome.weights, mutated.weights)


class SelectionTests(unittest.TestCase):
    def test_dominated_novelty_rewards_distance_from_fitter_solutions(self) -> None:
        fitnesses = [10.0, 5.0, 5.0]
        descriptors = [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]

        scores = dominated_novelty_scores(fitnesses, descriptors, neighbors=1)

        self.assertGreater(scores[0], scores[2])
        self.assertEqual(scores[1], 0.0)
        self.assertAlmostEqual(scores[2], math.sqrt(2.0))


class SimulationTests(unittest.TestCase):
    def small_config(self) -> SimulationConfig:
        return SimulationConfig(
            width=240,
            height=180,
            population_size=10,
            food_count=12,
            generation_steps=12,
            hidden_size=4,
            archive_bins=5,
            descriptor_grid=5,
            elite_count=2,
        )

    def test_generation_evolves_and_preserves_population_size(self) -> None:
        simulation = EvolutionSimulation(self.small_config(), seed=12)

        report = None
        while report is None:
            report = simulation.step()

        self.assertEqual(report["generation"], 1)
        self.assertEqual(simulation.generation, 2)
        self.assertEqual(len(simulation.agents), 10)
        self.assertEqual(len(simulation.genomes), 10)
        self.assertEqual(len(simulation.history), 1)
        self.assertGreater(len(simulation.archive), 0)

    def test_seed_reproduces_world_dynamics(self) -> None:
        left = EvolutionSimulation(self.small_config(), seed=21)
        right = EvolutionSimulation(self.small_config(), seed=21)

        for _ in range(7):
            left.step()
            right.step()

        left_state = [(agent.x, agent.y, agent.energy) for agent in left.agents]
        right_state = [(agent.x, agent.y, agent.energy) for agent in right.agents]
        self.assertEqual(left_state, right_state)
        self.assertEqual(left.foods, right.foods)

    def test_checkpoint_round_trip_continues_identically(self) -> None:
        simulation = EvolutionSimulation(self.small_config(), seed=33)
        for _ in range(5):
            simulation.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json.gz"
            simulation.save(path)
            restored = EvolutionSimulation.load(path)

            self.assertEqual(simulation.generation, restored.generation)
            self.assertEqual(simulation.step_count, restored.step_count)
            self.assertEqual(simulation.foods, restored.foods)

            simulation.step()
            restored.step()
            original_agents = [
                (agent.x, agent.y, agent.energy) for agent in simulation.agents
            ]
            restored_agents = [
                (agent.x, agent.y, agent.energy) for agent in restored.agents
            ]
            self.assertEqual(original_agents, restored_agents)


if __name__ == "__main__":
    unittest.main()
