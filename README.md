# NeuroEvolution Lab

An embodied neuroevolution experiment built entirely with the Python standard
library. Agents use evolved neural networks to sense food, manage energy, and
move through a shared environment.

## What changed

- Real feed-forward neural policies with continuous turn and thrust outputs.
- Food, energy, mortality, wall collisions, exploration, and behavioral traits.
- Dominated Novelty Search (DNS) selection: agents survive by being fitter or
  behaviorally distinct from fitter agents.
- A MAP-Elites-style archive indexed by roaming and movement behavior.
- Elitism, crossover, mutation, random immigrants, archive parent sampling, and
  automatic mutation expansion during stagnation.
- Interactive dashboard with a live population, fitness graph, archive heatmap,
  speed controls, reset, and checkpoints.
- Reproducible seeds, a headless experiment mode, and unit tests.
- Compressed JSON checkpoints instead of unsafe pickle deserialization.

## Run

```powershell
python main.py
```

Controls:

- `Space`: pause or resume
- `N`: advance one step while paused
- `+` / `-`: change simulation speed
- `S` / `L`: save or load `simulation_state.json.gz`
- `R`: reset with a new seed
- `Q`: quit

Run a reproducible experiment without the UI:

```powershell
python main.py --headless --generations 20 --seed 7
```

Export generation metrics for analysis:

```powershell
python main.py --headless --generations 50 --metrics runs/seed-7.csv
```

Use `python main.py --help` for population, food, generation length, and
checkpoint options.

## Test

```powershell
python -m unittest discover -s tests -v
```

## Research basis

This is a compact educational implementation, not a reproduction of a full
benchmark system. Its main algorithmic choices follow current
quality-diversity research:

- [Dominated Novelty Search (2025)](https://arxiv.org/abs/2502.00593) replaces
  a weighted fitness/novelty objective with distance from nearest fitter
  behaviors, creating local competition without fixed descriptor bounds.
- [MAP-Elites](https://arxiv.org/abs/1504.04909) motivates retaining a
  repertoire of high-quality policies across behavior niches rather than only
  one global champion.
- [Sample-Efficient Quality-Diversity (ICLR 2024)](https://openreview.net/forum?id=JDud6zbpFv)
  motivates reusing components of successful solutions; this project samples
  parents from both current DNS survivors and archived elites.
- [Quality-Diversity Actor-Critic (ICML 2024)](https://openreview.net/forum?id=ISG3l8nXrI)
  reinforces the value of learning diverse skills that can adapt under changed
  conditions. Here the archive makes those distinct policies visible and
  reusable.

The default behavior descriptor is intentionally interpretable: how much of
the arena an agent visits and how much it moves. DNS itself does not depend on
the archive grid or its bounds.
