from __future__ import annotations

import argparse
import colorsys
import csv
import math
import sys
import tkinter as tk
from pathlib import Path

from neuroevolution import EvolutionSimulation, SimulationConfig


BACKGROUND = "#07111c"
PANEL = "#0d1b2a"
GRID = "#163047"
TEXT = "#d7e3ee"
MUTED = "#7890a4"
ACCENT = "#42d6c7"
FOOD = "#f5c451"
DEAD = "#334656"


def genome_color(weights: list[float]) -> str:
    hue = (abs(weights[0]) * 0.173 + abs(weights[1]) * 0.071) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.95)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


class SimulationApp:
    def __init__(
        self,
        root: tk.Tk,
        simulation: EvolutionSimulation,
        state_path: Path,
    ):
        self.root = root
        self.simulation = simulation
        self.state_path = state_path
        self.paused = False
        self.steps_per_frame = 3
        self.status = "Running"
        self.canvas_width = simulation.config.width + 390
        self.canvas_height = max(simulation.config.height + 40, 740)
        self.world_x = 20
        self.world_y = 30
        self.panel_x = simulation.config.width + 50

        root.title("NeuroEvolution Lab")
        root.configure(bg=BACKGROUND)
        root.minsize(self.canvas_width, self.canvas_height)
        self.canvas = tk.Canvas(
            root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg=BACKGROUND,
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self._bind_keys()
        self._tick()

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda _event: self._toggle_pause())
        self.root.bind("n", lambda _event: self._single_step())
        self.root.bind("s", lambda _event: self._save())
        self.root.bind("l", lambda _event: self._load())
        self.root.bind("r", lambda _event: self._reset())
        self.root.bind("q", lambda _event: self.root.destroy())
        self.root.bind("<plus>", lambda _event: self._change_speed(1))
        self.root.bind("<equal>", lambda _event: self._change_speed(1))
        self.root.bind("<minus>", lambda _event: self._change_speed(-1))

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.status = "Paused" if self.paused else "Running"

    def _single_step(self) -> None:
        if self.paused:
            report = self.simulation.step()
            if report:
                self.status = (
                    f"Generation {report['generation']} complete, "
                    f"best {report['best_fitness']:.1f}"
                )
            self._draw()

    def _change_speed(self, direction: int) -> None:
        self.steps_per_frame = max(1, min(30, self.steps_per_frame + direction))
        self.status = f"Speed: {self.steps_per_frame} steps/frame"

    def _save(self) -> None:
        try:
            self.simulation.save(self.state_path)
            self.status = f"Saved {self.state_path.name}"
        except OSError as error:
            self.status = f"Save failed: {error}"

    def _load(self) -> None:
        try:
            self.simulation = EvolutionSimulation.load(self.state_path)
            self.status = f"Loaded {self.state_path.name}"
        except (OSError, ValueError) as error:
            self.status = f"Load failed: {error}"

    def _reset(self) -> None:
        next_seed = self.simulation.seed + 1
        self.simulation = EvolutionSimulation(self.simulation.config, next_seed)
        self.status = f"Reset with seed {next_seed}"

    def _tick(self) -> None:
        if not self.paused:
            for _ in range(self.steps_per_frame):
                report = self.simulation.step()
                if report:
                    self.status = (
                        f"Generation {report['generation']} complete, "
                        f"best {report['best_fitness']:.1f}"
                    )
        self._draw()
        self.root.after(33, self._tick)

    def _draw(self) -> None:
        self.canvas.delete("all")
        self._draw_world()
        self._draw_dashboard()

    def _draw_world(self) -> None:
        config = self.simulation.config
        x0, y0 = self.world_x, self.world_y
        x1, y1 = x0 + config.width, y0 + config.height
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=PANEL, outline=GRID, width=2)

        spacing = 68
        for x in range(spacing, config.width, spacing):
            self.canvas.create_line(x0 + x, y0, x0 + x, y1, fill="#10273a")
        for y in range(spacing, config.height, spacing):
            self.canvas.create_line(x0, y0 + y, x1, y0 + y, fill="#10273a")

        for food_x, food_y in self.simulation.foods:
            x = x0 + food_x
            y = y0 + food_y
            radius = config.food_radius
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=FOOD,
                outline="#fff0a6",
            )

        live_agents = [agent for agent in self.simulation.agents if agent.alive]
        leader = (
            max(live_agents, key=lambda agent: agent.provisional_fitness(config))
            if live_agents
            else None
        )
        for agent in self.simulation.agents:
            x = x0 + agent.x
            y = y0 + agent.y
            if not agent.alive:
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=DEAD, outline="")
                continue
            radius = config.agent_radius + (1.5 if agent is leader else 0.0)
            heading = agent.heading
            points = []
            for angle, distance in (
                (heading, radius * 1.65),
                (heading + 2.45, radius),
                (heading - 2.45, radius),
            ):
                points.extend((x + math.cos(angle) * distance, y + math.sin(angle) * distance))
            color = genome_color(agent.genome.weights)
            self.canvas.create_polygon(
                points,
                fill=color,
                outline="#ffffff" if agent is leader else "",
                width=2 if agent is leader else 1,
            )

        progress = self.simulation.step_count / config.generation_steps
        self.canvas.create_rectangle(x0, y1 + 8, x1, y1 + 13, fill=GRID, outline="")
        self.canvas.create_rectangle(
            x0, y1 + 8, x0 + config.width * progress, y1 + 13, fill=ACCENT, outline=""
        )

    def _text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        fill: str = TEXT,
        size: int = 11,
        weight: str = "normal",
        anchor: str = "nw",
    ) -> None:
        self.canvas.create_text(
            x,
            y,
            text=text,
            fill=fill,
            font=("Segoe UI", size, weight),
            anchor=anchor,
        )

    def _draw_dashboard(self) -> None:
        simulation = self.simulation
        config = simulation.config
        x = self.panel_x
        self._text(x, 30, "NEUROEVOLUTION LAB", fill=ACCENT, size=16, weight="bold")
        self._text(x, 61, self.status, fill=MUTED, size=10)

        best_live = max(
            (
                agent.provisional_fitness(config)
                for agent in simulation.agents
                if agent.alive
            ),
            default=0.0,
        )
        best_ever = simulation.best_ever_fitness or 0.0
        metrics = (
            ("Generation", str(simulation.generation)),
            ("Step", f"{simulation.step_count} / {config.generation_steps}"),
            ("Alive", f"{simulation.alive_count} / {config.population_size}"),
            ("Live best", f"{best_live:.1f}"),
            ("All-time best", f"{best_ever:.1f}"),
            ("Archive", f"{len(simulation.archive)} cells"),
            ("Coverage", f"{simulation.archive_coverage * 100:.0f}%"),
            ("Mutation sigma", f"{simulation.mutation_sigma:.3f}"),
        )
        top = 96
        for index, (label, value) in enumerate(metrics):
            row_y = top + index * 24
            self._text(x, row_y, label, fill=MUTED, size=10)
            self._text(x + 290, row_y, value, size=10, weight="bold", anchor="ne")

        self._text(x, 304, "BEHAVIOR REPERTOIRE", size=11, weight="bold")
        self._draw_archive(x, 330, 220)
        self._text(x, 560, "roaming  →", fill=MUTED, size=9)
        self._text(x + 232, 438, "speed", fill=MUTED, size=9)

        self._text(x, 592, "FITNESS HISTORY", size=11, weight="bold")
        self._draw_history(x, 620, 300, 70)

        controls = "Space pause   N step   +/- speed\nS save   L load   R reset   Q quit"
        self._text(x, 706, controls, fill=MUTED, size=9)

    def _draw_archive(self, x: float, y: float, size: float) -> None:
        simulation = self.simulation
        bins = simulation.config.archive_bins
        cell_size = size / bins
        fitness_values = [entry.fitness for entry in simulation.archive.values()]
        low = min(fitness_values, default=0.0)
        high = max(fitness_values, default=1.0)
        spread = max(1e-9, high - low)
        for grid_y in range(bins):
            for grid_x in range(bins):
                entry = simulation.archive.get((grid_x, grid_y))
                if entry is None:
                    color = "#102638"
                else:
                    strength = (entry.fitness - low) / spread
                    red, green, blue = colorsys.hsv_to_rgb(
                        0.50 - strength * 0.35, 0.75, 0.35 + strength * 0.60
                    )
                    color = (
                        f"#{int(red * 255):02x}{int(green * 255):02x}"
                        f"{int(blue * 255):02x}"
                    )
                x0 = x + grid_x * cell_size
                y0 = y + (bins - 1 - grid_y) * cell_size
                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + cell_size - 1,
                    y0 + cell_size - 1,
                    fill=color,
                    outline="",
                )

    def _draw_history(self, x: float, y: float, width: float, height: float) -> None:
        history = self.simulation.history[-60:]
        self.canvas.create_rectangle(
            x, y, x + width, y + height, fill="#091827", outline=GRID
        )
        if len(history) < 2:
            self._text(x + 8, y + 8, "Complete a generation to plot", fill=MUTED, size=9)
            return
        values = [float(item["best_fitness"]) for item in history]
        mean_values = [float(item["mean_fitness"]) for item in history]
        maximum = max(values) or 1.0

        def points_for(series: list[float]) -> list[float]:
            points: list[float] = []
            for index, value in enumerate(series):
                px = x + index / (len(series) - 1) * width
                py = y + height - value / maximum * (height - 8) - 4
                points.extend((px, py))
            return points

        self.canvas.create_line(points_for(mean_values), fill="#56738a", width=1)
        self.canvas.create_line(points_for(values), fill=ACCENT, width=2, smooth=True)


def run_headless(simulation: EvolutionSimulation, generations: int) -> None:
    target = len(simulation.history) + generations
    while len(simulation.history) < target:
        report = simulation.step()
        if report:
            print(
                "generation={generation} best={best_fitness:.2f} "
                "mean={mean_fitness:.2f} food={best_food} "
                "coverage={archive_coverage:.1%} sigma={mutation_sigma:.3f}".format(
                    **report
                )
            )


def export_metrics(simulation: EvolutionSimulation, path: Path) -> None:
    if not simulation.history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=simulation.history[0].keys())
        writer.writeheader()
        writer.writerows(simulation.history)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research-inspired embodied neuroevolution simulation"
    )
    parser.add_argument("--headless", action="store_true", help="run without the GUI")
    parser.add_argument(
        "--generations", type=int, default=20, help="generations in headless mode"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--population", type=int, default=72)
    parser.add_argument("--steps", type=int, default=450, help="steps per generation")
    parser.add_argument("--food", type=int, default=90)
    parser.add_argument(
        "--state", type=Path, default=Path("simulation_state.json.gz")
    )
    parser.add_argument("--load", action="store_true", help="load --state at startup")
    parser.add_argument(
        "--save-at-end", action="store_true", help="save after a headless run"
    )
    parser.add_argument(
        "--metrics", type=Path, help="write generation metrics to a CSV file"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.load:
        simulation = EvolutionSimulation.load(args.state)
    else:
        config = SimulationConfig(
            population_size=args.population,
            food_count=args.food,
            generation_steps=args.steps,
        )
        simulation = EvolutionSimulation(config=config, seed=args.seed)

    if args.headless:
        if args.generations < 1:
            raise ValueError("--generations must be positive")
        run_headless(simulation, args.generations)
        if args.metrics:
            export_metrics(simulation, args.metrics)
            print(f"metrics={args.metrics}")
        if args.save_at_end:
            simulation.save(args.state)
            print(f"saved={args.state}")
        return 0

    try:
        root = tk.Tk()
        SimulationApp(root, simulation, args.state)
        root.mainloop()
    except tk.TclError as error:
        print(f"GUI unavailable: {error}. Use --headless.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
