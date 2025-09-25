<!-- uft-8 -->
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

## Python
Using python >= 3.10 
(usefull for annotation type, do not use typing module)

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

### Run any python script via:
```bash
poetry run python <mon_script>.py

# Exemple 
poetry run python optimizer/orchestractor.py
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

### Global python tree-scructure:

.
├── GUI
│   ├── PGAnime.py
│   └── __init__.py
├── KPIS
│   ├── KpisGraphsGenerator.py
│   ├── LiveKpisGraphsGenerator.py
│   ├── __init__.py
│   ├── kpis.py
│   └── utils.py
├── __init__.py
├── eco2_normandy
│   ├── __init__.py
│   ├── factory.py
│   ├── logger.py
│   ├── port.py
│   ├── ship.py
│   ├── simulation.py
│   ├── stateSaver.py
│   ├── storage.py
│   ├── tools.py
│   └── weather.py
├── optimizer
│   ├── CPModel
│   │   ├── __init__.py
│   │   ├── callback.py
│   │   ├── cp_model.py
│   │   └── utils.py
│   ├── GAModel
│   │   ├── __init__.py
│   │   ├── ga_model.py
│   │   ├── history_analyzer.py
│   │   ├── problem.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── boundaries.py
│   ├── compare_scenarios.py
│   ├── orchestrator.py
│   └── utils.py
└── streamlit_app.py

### Testing & Validation

Use `scenarios/dev/` for development testing. The system supports:
- Single simulation runs for debugging
- Batch optimization with result comparison
- Visual validation through Pygame animation
- KPI validation through Plotly graphs in Streamlit UI

## User internship report

### Project Goal
The end goal of the project is writing an internship report following the example report `Exemple_Rapport_de_stage.pdf` and the instructions in `instruction_rapport_de_stage.pdf` in `Rapport` folder.

Reminder: The project aim to simulate and optimize co2 transport between 2 or 3 ports.
Simulation is done with simpy and have constraint (ship capacity, etc (see .yaml file)).
Optimization aim to minimize well chosen metrics that discribe the simulation, The optimization is done on 4 parts, each part aim to increase factory co2 production (i.e. more co2 to transport), and each part represent 1 years (for computation efficiency simulation is done on 2000 steps). 
The project came from "Air liquid" company.

### Report Writing Guidelines

#### Structure and Format
- Follow the global plan defined in `plan_rapport_global.md`
- Use narrative prose style instead of bullet points or numbered lists
- Use bullet points sparingly, only when absolutely necessary for clarity
- Max 50 pages total (academic M2 level)

#### Content Guidelines
- **Chapter 1**: Introduction générale (already written - covers context, problem, objectives, structure)
- **Chapter 2**: Focus on TNP Consultants (host company) rather than ENSAM
  - Brief general presentation of TNP Consultants as consulting company
  - Will be completed by the user with specific company details
- **Chapters 3-9**: To be developed following the detailed plan structure

#### Writing Style
- Convert all bullet points to flowing narrative text
- Construct complete sentences and paragraphs
- Maintain academic rigor while ensuring readability
- Use technical precision without excessive jargon

#### Preliminary Sections (User will complete)
- Sommaire (Table of contents)
- Liste des figures (List of figures)
- Liste des tableaux (List of tables)
- Acronymes et symboles (Acronyms and symbols)

#### Key Technical Focus Areas
- SimPy discrete event simulation
- Multi-objective optimization (NSGA-III)
- Hybrid optimization approaches
- Real-time visualization (Streamlit/Pygame)
- Le Havre to North Sea case study
- CAPEX/OPEX optimization objectives

#### Files for Reference
In the `Rapport` folder you can find the following files:
- "rapport_de_stage.md": Main report file (currently has Chapters 1-2 complete)
- "plan_rapport_global.md": Complete structure reference
- "Exemple_Rapport_de_stage_M2.pdf": Example report to follow
- "instruction_rapport_de_stage.pdf": Official instructions
- "Grille notation des stges M2.pdf": Intruction grid for notation.
- "AIR LIQUIDE FINAL PRESENTATION V2.pdf": difine project
- "Aide_rapport_stage.pdf": powerpoint of the report
