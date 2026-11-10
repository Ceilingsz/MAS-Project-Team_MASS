"""Turning run rows into numbers: means with confidence intervals, success rates, paired differences.

Rows are the dicts in runs.jsonl. Time-to-coverage metrics are None when a run never got there; by default such
runs are CENSORED at their final time (the run's end), and the share of censored runs is reported, so a method that
fails to finish is not silently dropped.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

TIME_METRICS = {"coverage_time", "t50", "t80", "t90", "t95", "t99", "t100"}
_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 11: 2.20, 12: 2.18,
        13: 2.16, 14: 2.14, 15: 2.13, 16: 2.12, 17: 2.11, 18: 2.10, 19: 2.09, 20: 2.09, 25: 2.06, 30: 2.04}


def t_crit(df: int) -> float:
    """Two-sided 95% Student-t critical value."""
    if df < 1:
        return float("nan")
    if df in _T95:
        return _T95[df]
    keys = [k for k in _T95 if k > df]
    return _T95[min(keys)] if keys else 1.96


def load_rows(path) -> list[dict]:
    p = Path(path)
    p = p / "runs.jsonl" if p.is_dir() else p
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def load_curves(path) -> dict:
    p = Path(path)
    p = p / "curves.jsonl" if p.is_dir() else p
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["job_id"]] = d
    return out


def ok_rows(rows, where=None) -> list[dict]:
    return [r for r in rows if r.get("status", "ok") == "ok" and all(r.get(k) == v for k, v in (where or {}).items())]


def value(row: dict, metric: str, censor: bool = True):
    """The row's value of a metric; censored time metrics fall back to the run's final time."""
    v = row.get(metric)
    if v is None and censor and metric in TIME_METRICS:
        return row.get("final_time")
    if isinstance(v, bool):
        return float(v)
    return v


def mean_ci(values) -> dict:
    """Mean, 95% half-width, n of the finite values."""
    x = np.array([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return dict(mean=float("nan"), ci=float("nan"), n=0, std=float("nan"))
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    return dict(mean=float(x.mean()), ci=t_crit(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan"), n=n, std=sd)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Success rate with its Wilson 95% interval: (p, low, high)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def group_stats(rows, by, metric: str, censor: bool = True) -> list[dict]:
    """One entry per distinct value of the columns in ``by``: mean, ci, n (+ share censored for time metrics)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in by), []).append(r)
    out = []
    for key, rs in groups.items():
        s = mean_ci([value(r, metric, censor) for r in rs])
        s.update(zip(by, key))
        if metric in TIME_METRICS:
            s["censored"] = sum(r.get(metric) is None for r in rs) / len(rs)
        out.append(s)
    return out


def rate_stats(rows, by, metric: str = "safe_complete") -> list[dict]:
    """Share of runs where a boolean metric is true, per group, with Wilson intervals."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in by), []).append(r)
    out = []
    for key, rs in groups.items():
        k = sum(bool(r.get(metric)) for r in rs)
        p, lo, hi = wilson(k, len(rs))
        out.append(dict(zip(by, key), rate=p, lo=lo, hi=hi, k=k, n=len(rs)))
    return out


def paired(rows, a: str, b: str, metric: str, match=("map", "n_agents", "seed"), censor: bool = True,
           label_col: str = "label") -> dict:
    """Paired difference a - b of a metric between two labels, matched on the same map / team size / seed
    (plus any other columns in ``match``, for example a lever column). Returns mean difference, 95% half-width,
    the mean relative difference (a - b) / b and n pairs."""
    def keyed(label):
        return {tuple(r.get(k) for k in match): value(r, metric, censor) for r in rows if r.get(label_col) == label}
    ka, kb = keyed(a), keyed(b)
    diffs, rel = [], []
    for k in ka.keys() & kb.keys():
        x, y = ka[k], kb[k]
        if x is None or y is None:
            continue
        diffs.append(x - y)
        if y:
            rel.append((x - y) / y)
    s = mean_ci(diffs)
    r = mean_ci(rel)
    return dict(a=a, b=b, metric=metric, mean_diff=s["mean"], ci=s["ci"], rel=r["mean"], rel_ci=r["ci"], n=s["n"])


def summary_table(rows, by, metrics, censor: bool = True) -> list[dict]:
    """Wide table: one row per group, mean and ci of each metric."""
    base = group_stats(rows, by, metrics[0], censor)
    table = {tuple(g[k] for k in by): {k: g[k] for k in by} | {"n": g["n"]} for g in base}
    for m in metrics:
        for g in group_stats(rows, by, m, censor):
            row = table[tuple(g[k] for k in by)]
            row[f"{m}_mean"], row[f"{m}_ci"] = g["mean"], g["ci"]
    return list(table.values())


def write_csv(path, rows) -> None:
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
