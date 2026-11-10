"""Run experiment jobs (in parallel), write results as they finish, resume where a sweep stopped.

Output directory ``<out>/<experiment>/``:
    runs.jsonl    one JSON object per finished run (scalar measurables + the lever columns)   <- source of truth
    curves.jsonl  one object per run: job_id, t, coverage, team_believed (thinned time series)
    runs.csv      the same rows as runs.jsonl, rewritten at the end for spreadsheets
    skipped.txt   jobs that were not run and why (model not registered yet, known-map-only model on an unknown map)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mascov.experiments.spec import Job, lever_column


def _ensure_registered() -> None:
    """Import the models package (repo root) so every model is in the registry, also inside worker processes."""
    for p in ("src", "."):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import models  # noqa: F401
    except ImportError:
        pass


def _model_interface(job: Job) -> str | None:
    """'known' / 'unknown' for a registered model, None if it is not registered."""
    from mascov.common.model import get_model, registered_models
    if job.model not in registered_models():
        return None
    return getattr(get_model(job.model, **job.kwargs), "map_interface", "known")


def filter_jobs(jobs: list[Job]) -> tuple[list[Job], list[tuple[Job, str]]]:
    """Split into runnable jobs and (job, reason) pairs that cannot run."""
    _ensure_registered()
    ok, skipped, cache = [], [], {}
    for j in jobs:
        key = (j.model, json.dumps(j.kwargs, sort_keys=True, default=str))
        if key not in cache:
            cache[key] = _model_interface(j)
        iface = cache[key]
        if iface is None:
            skipped.append((j, f"model '{j.model}' is not registered (not implemented yet?)"))
        elif iface == "known" and j.map_prior == "unknown":
            skipped.append((j, f"model '{j.model}' is a known-map model; it cannot run with map_prior='unknown'"))
        else:
            ok.append(j)
    return ok, skipped


def run_job(job: Job, world_builder=None) -> tuple[dict, dict | None]:
    """Run one job. Returns (row, curve); on an error the row has status='error' and the curve is None."""
    _ensure_registered()
    from mascov.common.config import SimConfig, WorldConfig
    import inspect
    from mascov.common.model import _REGISTRY, get_model
    from mascov.sim.simulator import Simulator
    if world_builder is None:
        from mascov.maps.benchmarks import make_world as world_builder

    # Build row with lever columns first so a lever named "map_prior" doesn't collide with the fixed column
    row = {lever_column(k): v for k, v in job.levers.items()}
    row.update(job_id=job.job_id, exp=job.exp, label=job.label, model=job.model, role=job.role, map=job.map,
               n_agents=job.n_agents, seed=job.seed, map_prior=job.map_prior)
    t0 = time.perf_counter()
    try:
        kwargs = dict(job.kwargs)
        if "seed" not in kwargs and job.model in _REGISTRY and "seed" in inspect.signature(_REGISTRY[job.model]).parameters:
            kwargs["seed"] = job.seed
        model = get_model(job.model, **kwargs)
        iface = getattr(model, "map_interface", "known")
        cfg = WorldConfig().with_(**job.world)
        world = world_builder(job.map, n_agents=job.n_agents, size=job.size, seed=job.seed, cfg=cfg)
        sim = dict(max_time=job.max_time, stop_when_covered=True, known_map=(iface == "known"),
                   reveal_obstacles_at_start=(iface == "unknown" and job.map_prior == "known"), seed=job.seed)
        sim.update(job.sim)
        res = Simulator(world, model, SimConfig(**sim)).run()
        row.update(res.metrics())
        row.update({f"x_{k}": v for k, v in res.extra.items() if isinstance(v, (int, float, bool))})
        t, c = res.coverage_curve()
        curve = dict(job_id=job.job_id, t=[round(float(x), 3) for x in t], coverage=[round(float(x), 5) for x in c])
        curve["team_believed"] = [round(float(x), 5) for x in res.coverage_curve(column="team_believed")[1]]
        row.update(status="ok", error="")
    except Exception as e:                                           # a failed run must not kill the sweep
        row.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
        curve = None
    row["wall_s"] = round(time.perf_counter() - t0, 2)
    return row, curve


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_all(jobs: list[Job], out_dir, n_jobs: int | None = None, progress: bool = True, world_builder=None) -> dict:
    """Run every job not already in runs.jsonl. ``world_builder`` (in-process runs only) replaces make_world."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runnable, skipped = filter_jobs(jobs)
    (out / "skipped.txt").write_text("\n".join(f"{j.label} | {j.map} n={j.n_agents} seed={j.seed}: {why}"
                                               for j, why in skipped))
    done = {r["job_id"] for r in _read_jsonl(out / "runs.jsonl") if r.get("status") == "ok"}
    todo = [j for j in runnable if j.job_id not in done]
    n_jobs = n_jobs or os.cpu_count() or 1
    if progress:
        print(f"{len(jobs)} jobs: {len(done & {j.job_id for j in runnable})} already done, {len(todo)} to run, "
              f"{len(skipped)} skipped ({len({w for _, w in skipped})} distinct reasons, see skipped.txt)")
    t0, n_done, n_err = time.perf_counter(), 0, 0
    with open(out / "runs.jsonl", "a") as fr, open(out / "curves.jsonl", "a") as fc:
        def record(row, curve):
            nonlocal n_done, n_err
            fr.write(json.dumps(row, default=str) + "\n"), fr.flush()
            if curve:
                fc.write(json.dumps(curve) + "\n"), fc.flush()
            n_done += 1
            n_err += row["status"] != "ok"
            if progress:
                el = time.perf_counter() - t0
                eta = el / n_done * (len(todo) - n_done)
                print(f"  [{n_done}/{len(todo)}] {row['label']} {row['map']} n={row['n_agents']} seed={row['seed']} "
                      f"{row['status']} ({row['wall_s']} s, eta {eta:.0f} s)" + (f"  {row['error']}" if row["error"] else ""))

        if n_jobs <= 1 or len(todo) <= 1 or world_builder is not None:
            for j in todo:
                record(*run_job(j, world_builder))
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                for fut in as_completed([ex.submit(run_job, j) for j in todo]):
                    record(*fut.result())
    export_csv(out)
    return dict(total=len(jobs), ran=n_done, errors=n_err, skipped=len(skipped), out=str(out))


def export_csv(out_dir) -> None:
    out = Path(out_dir)
    rows = _read_jsonl(out / "runs.jsonl")
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
