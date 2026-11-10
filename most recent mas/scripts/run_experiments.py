"""Run experiments from the study plan (see src/mascov/experiments/definitions.py).

  python scripts/run_experiments.py --list
  python scripts/run_experiments.py E1 --quick --jobs 4              # small smoke run
  python scripts/run_experiments.py E1 E2 --seeds 10 --jobs 8 --out results
  python scripts/run_experiments.py all --dry-run                    # job counts and which models are missing

Results go to <out>/<experiment>/ (runs.jsonl, runs.csv, curves.jsonl, skipped.txt). A run that is already in
runs.jsonl is not repeated, so an interrupted sweep resumes. Draw the figures with scripts/plot_experiments.py.
Run from the repository root.
"""
import argparse
import sys
from collections import Counter

sys.path.insert(0, "src"), sys.path.insert(0, ".")
from mascov.experiments.definitions import EXPERIMENTS  # noqa: E402
from mascov.experiments.runner import filter_jobs, run_all  # noqa: E402
from mascov.experiments.spec import expand  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="*", help="experiment names (E1 E2 ...) or 'all'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quick", action="store_true", help="the experiment's small smoke-run settings")
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds (0..n-1)")
    ap.add_argument("--maps", nargs="+", default=None)
    ap.add_argument("--n", nargs="+", type=int, default=None, help="team sizes")
    ap.add_argument("--max-time", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=None, help="parallel worker processes (default: all cores)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.list or not a.exps:
        for e in EXPERIMENTS.values():
            print(f"{e.name:4s} {e.title}: {e.question}")
        return
    names = list(EXPERIMENTS) if a.exps == ["all"] else a.exps
    for name in names:
        if name not in EXPERIMENTS:
            sys.exit(f"unknown experiment '{name}'. options: {', '.join(EXPERIMENTS)}")
        e = EXPERIMENTS[name]
        kw = dict(e.quick) if a.quick else {}
        if a.seeds is not None:
            kw["seeds"] = list(range(a.seeds))
        if a.maps:
            kw["maps"] = a.maps
        if a.n:
            kw["n_agents"] = a.n
        if a.max_time:
            e.max_time = a.max_time
        jobs = expand(e, seeds=kw.get("seeds"), maps=kw.get("maps"), n_agents=kw.get("n_agents"))
        print(f"== {e.name}: {e.title}")
        if a.dry_run:
            ok, skipped = filter_jobs(jobs)
            print(f"   {len(jobs)} jobs: {len(ok)} runnable, {len(skipped)} skipped")
            for why, n in Counter(w for _, w in skipped).items():
                print(f"     skipped {n}: {why}")
            continue
        info = run_all(jobs, f"{a.out}/{e.name}", n_jobs=a.jobs)
        print(f"   done: {info}")


if __name__ == "__main__":
    main()
