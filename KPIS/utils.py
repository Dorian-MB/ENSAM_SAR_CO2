from typing import Dict, List, Tuple
import pandas as pd
from IPython.display import display
from pprint import pprint


def to_MultiIndex_dfs(dic):
    dfs = []
    for name in dic.keys():
        d = pd.DataFrame(dic[name])
        d.index.name = "step"
        d.columns = pd.MultiIndex.from_product([[name], d.columns])
        dfs.append(d)
    return pd.concat(dfs, axis=1)


def normalize_absolute(
    kpis: Dict[str, float], bounds: Dict[str, Tuple[float, float]], clip: bool = True
) -> Dict[str, float]:
    """
    Normalize KPIs by fixed absolute bounds.
    bounds = {k: (min_value, max_value)} for each KPI key.
    clip: bool, if True, clips values to [0,1].
    Returns values clipped to [0,1].
    """
    normed: Dict[str, float] = {}
    for key, value in kpis.items():
        if key not in bounds:
            raise KeyError(f"No absolute bounds provided for KPI '{key}'")
        lo, hi = bounds[key]
        if hi == lo:
            normed[key] = 0.0
        else:
            raw = (value - lo) / (hi - lo)
            normed[key] = raw if not clip else max(0.0, min(1.0, raw))
    return normed


def compute_dynamic_bounds(kpis_list: List[dict]) -> Dict[str, Tuple[float, float]]:
    """
    Compute (min,max) for each KPI across a list of raw KPI dicts.
    For dynamic normalization.
    """
    df = pd.DataFrame(kpis_list)
    bounds = {}
    for col in df.columns:
        low = df[col].min()
        hight = df[col].max()
        std = df[col].std()
        bounds[col] = (max(low - std, 0), hight + std)  # add variance for robustness
    return bounds


def normalize_dynamic(kpis, dynamic_bounds, clip: bool = True) -> Dict[str, float]:
    return normalize_absolute(kpis, dynamic_bounds, clip)


# Example usage:
if __name__ == "__main__":
    raw_list = [
        {"cost": 500e6, "wasted": 1.2e6, "waiting_time": 1e4, "underfill_rate": 0.9},
        {"cost": 550e6, "wasted": 1.0e6, "waiting_time": 2e4, "underfill_rate": 0.8},
        {"cost": 600e6, "wasted": 0.5e6, "waiting_time": 3e4, "underfill_rate": 0.7},
        {"cost": 450e6, "wasted": 1.5e6, "waiting_time": 4e4, "underfill_rate": 0.6},
        {"cost": 400e6, "wasted": 2.0e6, "waiting_time": 5e4, "underfill_rate": 0.5},
    ]

    dyn_bounds = compute_dynamic_bounds(raw_list)
    normed_dynamic = [normalize_dynamic(r, dyn_bounds) for r in raw_list]

    abs_bounds = {
        "cost": (400e6, 600e6),
        "wasted": (0, 2e6),
        "waiting_time": (0, 5e4),
        "underfill_rate": (0, 1),
    }
    normed_absolute = [normalize_absolute(r, abs_bounds) for r in raw_list]

    print("Dynamic normalization:")
    display(pd.DataFrame(normed_dynamic).astype(float).round(3))
    print("Dynamic bounds:")
    display(pd.DataFrame(dyn_bounds).astype(float).round(3))
    print("\nAbsolute normalization:")
    display(pd.DataFrame(normed_absolute).astype(float).round(3))
    print("Difference between dynamic and absolute normalization:\n\n")
    display((pd.DataFrame(normed_dynamic) - pd.DataFrame(normed_absolute)).astype(float).round(3))
