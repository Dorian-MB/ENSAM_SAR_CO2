import sys
from pathlib import Path
if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))
from copy import deepcopy
from optimizer.utils import get_all_scenarios

def get_configs_list():
    return list(get_all_scenarios("scenarios/", ignore_cte=True))

def ensure_constante_are_ignore(config):
    ignore_keys = {"KPIS", "general", "allowed_speeds", "weather_probability"}
    config_ = deepcopy(config)
    for key in ignore_keys:
        config_.pop(key, None)
    return config_

def compare_scenario(d1, d2, path=""):
    """Compare récursivement deux dictionnaires ou listes."""
    diffs = []
    if isinstance(d1, dict) and isinstance(d2, dict):
        keys = set(d1.keys()) | set(d2.keys())
        for k in keys:
            v1 = d1.get(k, "<absent>")
            v2 = d2.get(k, "<absent>")
            if v1 != v2:
                diffs.extend(compare_scenario(v1, v2, path + f".{k}"))
    elif isinstance(d1, list) and isinstance(d2, list):
        for i, (v1, v2) in enumerate(zip(d1, d2)):
            if v1 != v2:
                diffs.extend(compare_scenario(v1, v2, path + f"[{i}]"))
        if len(d1) != len(d2):
            diffs.append(f"{path}: list length {len(d1)} != {len(d2)}")
    else:
        if d1 != d2:
            diffs.append(f"{path}: {d1} != {d2}")
    return diffs

def print_diffs(cfg1, cfg2, all_keys=None):
    if isinstance(cfg1, tuple) and isinstance(cfg2, tuple):
        _, cfg1 = cfg1
        _, cfg2 = cfg2
    if all_keys is None:
        all_keys = set(cfg1.keys()) | set(cfg2.keys())
    cfg1 = ensure_constante_are_ignore(cfg1)
    cfg2 = ensure_constante_are_ignore(cfg2)

    for key in all_keys:
        v1 = cfg1.get(key, "<absent>")
        v2 = cfg2.get(key, "<absent>")
        if v1 != v2:
            print(f"  Différence sur '{key}':")
            diffs = compare_scenario(v1, v2, key)
            for diff in diffs:
                print("   ", diff)

def main():
    configs = get_configs_list()
    # Affiche la liste des scénarios trouvés
    print("Scénarios trouvés :")
    for idx, (name, _) in enumerate(configs):
        print(f"{idx}: {name}")

    # Demande à l'utilisateur de choisir deux scénarios à comparer
    try:
        idx1 = int(input("Numéro du 1er scénario à comparer : "))
        idx2 = int(input("Numéro du 2ème scénario à comparer : "))
    except ValueError:
        print("Entrée invalide. Veuillez entrer des numéros valides.")
        sys.exit(1)

    if idx1 < 0 or idx1 >= len(configs) or idx2 < 0 or idx2 >= len(configs) or idx1 == idx2:
        print("Numéros invalides ou identiques.")
        sys.exit(1)

    # Liste des clés présentes dans au moins un scénario
    print_diffs(configs[idx1], configs[idx2])

if __name__ == "__main__":
    main()