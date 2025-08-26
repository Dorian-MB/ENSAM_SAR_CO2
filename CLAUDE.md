# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation & Environment
```bash
# Install dependencies (recommended)
poetry install

# Alternative via pip
pip install -r requirements.txt
```

### Running the Application
```bash
# Launch Streamlit UI
poetry run streamlit run streamlit_app.py
# or
make st

# Run a quick simulation test
poetry run python -m eco2_normandy.simulation
```

### Development & Testing
```bash
# Run tests (alpha version ,not important for now)
poetry run pytest -q

# Format code
poetry run black .

# View Python file structure
make py-tree

# Build Sphinx documentation
make -C docs html
```

## Architecture Overview

This is a CO₂ transport simulation and optimization system with discrete event simulation (SimPy) at its core. The system models CO₂ transport from a factory (Le Havre) to storage terminals, with multi-objective optimization capabilities.

### Core Components

**eco2_normandy/**: Main simulation engine
- `simulation.py`: Orchestrates the discrete event simulation using SimPy
- `factory.py`, `storage.py`, `ship.py`: Core entities with state management
- `weather.py`: Environmental conditions affecting ship speeds
- `stateSaver.py`: Collects simulation data at each time period
- `tools.py`: Configuration parsing and data processing utilities

**optimizer/**: Multi-objective optimization framework
- `orchestrator.py`: Main optimization coordinator (`OptimizationOrchestrator` class)
- `ga_model.py`: Genetic algorithm implementation (NSGA3) using pymoo
- `cp_model.py`: Constraint programming model using OR-Tools CP-SAT
- `boundaries.py`: Parameter bounds for optimization variables
- `utils.py`: Configuration building from optimization solutions

**KPIS/**: Key Performance Indicators calculation
- `kpis.py`: Core KPI calculations (costs, efficiency metrics)
- `KpisGraphsGenerator.py`: Plotly visualization generation
- Results include investment costs, operational costs, waiting times, waste metrics

**GUI/**: Pygame-based animation system
- `PGAnime.py`: Real-time visualization of simulation state

### Configuration System

YAML-based configuration with special parsing rules:
- All scalar values must be wrapped in arrays: `name: [Ship 1]` not `name: Ship 1`
- Range generation: `capacity_max: {range: [20, 30000, 1000]}` generates 20, 1020, 2020...
- Multiple entities supported for ships/storages, single factory only
- Auto-duplication when `num_ships` exceeds defined ship count

### Optimization Workflow

1. **GA Optimization** (NSGA3):
```python
from optimizer.ga_model import GAModel
from optimizer.orchestrator import OptimizationOrchestrator

model = GAModel(config, pop_size=100, n_gen=10, parallelization=True)
opt = OptimizationOrchestrator(model, verbose=1)
opt.optimize()
```

2. **CP-SAT Optimization**:
```python
from optimizer.cp_model import CpModel

model = CpModel(config)
opt = OptimizationOrchestrator(model)
opt.optimize(max_evals=5)
```

3. **Solution Analysis**:
- Solutions saved to `saved/` directory (CSV format)
- Pareto front analysis with `opt.plot_pareto()`
- Config reconstruction via `ConfigBuilderFromSolution.build(solution)`

### Key Design Patterns

**State Management**: Ships maintain states (DOCKED, NAVIGATING, LOADING, etc.) with SimPy environment coordination.

**Event-Driven Architecture**: StateSaver collects entity states at each simulation period for post-processing.

**Multi-Objective Optimization**: Simultaneous optimization of investment costs, operational costs, and efficiency metrics.

**Modular KPI System**: Extensible cost calculation framework supporting various business metrics.

### Important File Locations

- Configuration scenarios: `scenarios/` (organized by phases)
- Optimization results: `saved/` (scores, solutions, pareto fronts)
- Animation assets: `assets/` (PNG images for pygame)
- Documentation: `docs/` (Sphinx-based)

### Testing & Validation

Use `scenarios/dev/` for development testing. The system supports:
- Single simulation runs for debugging
- Batch optimization with result comparison
- Visual validation through Pygame animation
- KPI validation through Plotly graphs in Streamlit UI