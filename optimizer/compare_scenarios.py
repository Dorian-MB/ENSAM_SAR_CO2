import sys
from optimizer.utils import get_all_scenarios
configs = get_all_scenarios("scenarios/", ignore_cte=True)

# Liste des clés présentes dans au moins un scénario
all_keys = set()
for _, config in configs:
    all_keys.update(config.keys())

def compare_dicts(d1, d2, path=""):
    """Compare récursivement deux dictionnaires ou listes."""
    diffs = []
    if isinstance(d1, dict) and isinstance(d2, dict):
        keys = set(d1.keys()) | set(d2.keys())
        for k in keys:
            v1 = d1.get(k, "<absent>")
            v2 = d2.get(k, "<absent>")
            if v1 != v2:
                diffs.extend(compare_dicts(v1, v2, path + f".{k}"))
    elif isinstance(d1, list) and isinstance(d2, list):
        for i, (v1, v2) in enumerate(zip(d1, d2)):
            if v1 != v2:
                diffs.extend(compare_dicts(v1, v2, path + f"[{i}]"))
        if len(d1) != len(d2):
            diffs.append(f"{path}: list length {len(d1)} != {len(d2)}")
    else:
        if d1 != d2:
            diffs.append(f"{path}: {d1} != {d2}")
    return diffs

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

name1, conf1 = configs[idx1]
name2, conf2 = configs[idx2]

print(f"\nComparaison entre {name1} et {name2}:")
for key in all_keys:
    v1 = conf1.get(key, "<absent>")
    v2 = conf2.get(key, "<absent>")
    if v1 != v2:
        print(f"  Différence sur '{key}':")
        diffs = compare_dicts(v1, v2, key)
        for diff in diffs:
            print("   ", diff)






