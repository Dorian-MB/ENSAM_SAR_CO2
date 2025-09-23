
# eco2_normandy – Simulation & Optimisation du transport de CO₂

Simulation discrète (SimPy) du transport de CO₂ liquéfié entre une usine (factory) et un ou plusieurs terminaux de stockage (storage), génération de KPI, visualisation (Streamlit + Pygame) et optimisation multi‑objectifs (CP-SAT / Algorithmes évolutionnaires).

## Sommaire
1. Installation rapide
2. Lancer une simulation (code & UI Streamlit)
3. Visualisation Pygame
4. Structure d’un fichier de configuration
5. Scénarios & optimisation (GA / CP)
6. KPI disponibles
7. Structure du projet
8. Développement & commandes utiles
9. Documentation Sphinx

---
## 1. Installation rapide

Prérequis: Python 3.10 – 3.13.

Via Poetry (recommandé):
```bash
poetry install
poetry run python -m eco2_normandy.simulation  # test rapide
```

Ou via requirements.txt (environnement actif):
```bash
pip install -r requirements.txt
```

Lancer l’app Streamlit:
```bash
poetry run streamlit run streamlit_app.py
```
Ou via Make:
```bash
make st
```


---
## 2. Lancer une simulation par code
```python
from eco2_normandy.simulation import Simulation
from eco2_normandy.tools import get_simlulation_variable

config, *_ = get_simlulation_variable("config.yaml")  # ou un fichier dans scenarios/
config["general"]["num_period"] = 2000
sim = Simulation(config=config, verbose=True)
sim.run()
results_df = sim.result  # DataFrame multi‑entités
```

### Interface Streamlit
Lancez l’UI (formulaire interactif pour définir factory / storages / ships) :
```bash
poetry run streamlit run streamlit_app.py
# ou 
make st
```
Une fois la simulation terminée, des graphiques Plotly (capacités, waste, temps d’attente, coûts…) sont affichés.

---
## 3. Visualisation Pygame
Depuis l’UI Streamlit bouton “PyGame Animation 🚀” ou directement:
```python
from GUI.PGAnime import PGAnime
PGAnime(config).run()
```

---
## 4. Fichier de configuration (YAML)
Principales sections: `general`, `factory`, `storages`, `ships`, `KPIS`, `allowed_speeds`, `weather_probability`.

### Format des valeurs (important)
Toutes les valeurs “scalaires” susceptibles d’être générées en série peuvent être écrites sous forme de liste pour un parsing homogène.

Incorrect:
```yaml
name: Ship 1
```
Correct:
```yaml
name: [Ship 1]
```

### Génération de plages (range)
```yaml
ships:
  - name: [Ship 1]
    capacity_max:
      range: [20, 30000, 1000]  # start, end, step
```
Génère: 20, 1020, 2020, …, 29000.

### Multiplicité
- Plusieurs ships & storages.
- Une seule factory (sinon erreur).

### Paramètre `general.num_ships`
Si moins d’objets définis que `num_ships`, duplication automatique des navires pour atteindre le nombre requis.

### Exemple minimal
```yaml
general:
  num_period: 2000
  num_period_per_hours: 1
  num_ships: 2
  distances:
    Le Havre:
      Rotterdam: 263
factory:
  name: Le Havre
storages:
  - name: Rotterdam
ships:
  - name: Ship 1
  - name: Ship 2
KPIS:
  fuel_price_per_ton: 520
allowed_speeds:
  wind: {"6": 12, "10": 10, "20": 0}
  wave: {"6": 12, "10": 10, "20": 0}
  current: {"6": 12, "10": 10, "20": 0}
weather_probability:
  wind: 0.1
  waves: 0.1
  current: 0.1
```

---
## 5. Scénarios & Optimisation
Les scénarios YAML se trouvent dans `scenarios/` (ex: `scenarios/dev/...`).

### Lancer une optimisation GA (NSGA3)
```python
from optimizer.ga_model import GAModel
from optimizer.orchestrator import OptimizationOrchestrator
from eco2_normandy.tools import get_simlulation_variable

config, *_ = get_simlulation_variable("scenarios/dev/phase3_bergen_18k_2boats.yaml")
model = GAModel(config, pop_size=100, n_gen=10, parallelization=True, algorithm="NSGA3")
opt = OptimizationOrchestrator(model, verbose=1)
opt.optimize()
opt.log_score()
opt.save_solution()
```

### CP-SAT
```python
from optimizer.cp_model import CpModel
from optimizer.orchestrator import OptimizationOrchestrator
model = CpModel(config)
opt = OptimizationOrchestrator(model)
opt.optimize(max_evals=5)
```

### Comparaison / Pareto
```python
opt.plot_pareto()  # scatter matrices
```
Résultats CSV dans `saved/` (scores, solutions, pareto...).

---
## 6. KPI principaux
Calculés dans `KPIS/kpis.py`:
- Initial Investment:
  - Storage Tank Purchase Cost
  - Boat Purchase Cost
- Functional Costs:
  - Fuel Cost
  - Boat Operational Costs (navigation + usage + staff)
  - Boat Stoppage Cost (attente / immobilisation)
  - Navigation Cost (décomposé dans operational)
  - CO2 Storage Cost
  - co2_released_cost (venting)
  - Delay Cost (placeholder)
  - Total Cost
- Combined Total Cost = Investissement initial + Total Cost
Autres métriques: wasted_production_over_time, waiting_time, underfill_rate, filling rates, navigating vs waiting time.

---
## 7. Structure (extrait)
```
eco2_normandy/     # Entités simulation (Factory, Storage, Ship, Weather, ...)
KPIS/              # Calcul & graphiques KPI
optimizer/         # Modèles d'optimisation (GA, CP) + orchestrateur
GUI/               # Animation Pygame
scenarios/         # Scénarios YAML
streamlit_app.py   # UI interactive
docs/              # Sphinx
saved/             # Export des solutions & scores
```

---
## 8. Développement
Tests (pytest):
```bash
poetry run pytest -q
```
Formatage:
```bash
poetry run black .
```
Lister l’arborescence Python:
```bash
make py-tree
```

---
## 9. Documentation Sphinx
Construire la doc HTML:
```bash
make -C docs html
open docs/build/html/index.html
```

---
## 10. Entités & logique (résumé)
Factory: production, maintenance (scheduled / unscheduled), pompes de chargement, pénalités de venting.
Storage: réception, pompes de déchargement, capacité & coûts de stockage.
Ship: états (DOCKED, DOCKING, WAITING, NAVIGATING, LOADING, UNLOADING), vitesse contrainte par météo, pilotes / lock / docks.
WeatherStation: génère conditions impactant la vitesse.
StateSaver: collecte des états à chaque période.

---
## 11. Notes
- Les valeurs de coût peuvent être ajustées dans la section `KPIS` du config.
- `num_period` * (1/`num_period_per_hours`) = nombre d’heures simulées.
- Pour analyser une solution optimisée, reconstruire le config via `OptimizationOrchestrator.build_config_from_solution` puis rejouer Simulation / Pygame.

---
## Licence
Ajouter ici la licence si nécessaire (MIT / Proprietary…).

---
Pull requests & issues bienvenus.

