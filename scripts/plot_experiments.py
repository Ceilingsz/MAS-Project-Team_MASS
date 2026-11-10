"""Draw the standard figures and summary tables of experiments already run with scripts/run_experiments.py.

  python scripts/plot_experiments.py E1 E2 --results results           # figures -> results/<exp>/figures/
  python scripts/plot_experiments.py all --results results

Also writes <results>/<exp>/summary.csv: per label and lever value, mean and 95% interval of the headline
measurables (time to 95%, visits per target, collisions, messages per drone per second, ...), the safe-complete
rate, and how many runs were censored (never reached the coverage level). Run from the repository root.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src"), sys.path.insert(0, ".")
from mascov.experiments import analysis as an  # noqa: E402
from mascov.experiments.definitions import EXPERIMENTS  # noqa: E402
from mascov.experiments.plots import make_figures  # noqa: E402
from mascov.experiments.spec import lever_column  # noqa: E402

HEADLINE = ["t95", "final_coverage", "mean_visits", "collisions", "min_pair_dist", "path_len_mean",
            "msgs_per_drone_per_s", "solve_ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    for name in (list(EXPERIMENTS) if a.exps == ["all"] else a.exps):
        e = EXPERIMENTS[name]
        d = Path(a.results) / name
        if not (d / "runs.jsonl").exists():
            print(f"{name}: no results in {d} (run scripts/run_experiments.py {name} first)")
            continue
        rows = an.ok_rows(an.load_rows(d))
        curves = an.load_curves(d) if (d / "curves.jsonl").exists() else {}
        by = ["label", "map", "n_agents", "map_prior"] + [lever_column(k) for k in e.levers]
        table = an.summary_table(rows, by, HEADLINE)
        rate = {tuple(r[k] for k in by): r for r in an.rate_stats(rows, by)}
        for row in table:
            r = rate[tuple(row[k] for k in by)]
            row.update(safe_complete_rate=r["rate"], safe_complete_lo=r["lo"], safe_complete_hi=r["hi"])
            row["t95_censored"] = next(g["censored"] for g in an.group_stats(
                [x for x in rows if all(x.get(k) == row[k] for k in by)], [], "t95"))
        an.write_csv(d / "summary.csv", table)
        written = [p for spec in e.plots for p in make_figures(spec, rows, curves, d / "figures")]
        print(f"{name}: {len(rows)} runs -> {len(written)} figures in {d / 'figures'}, summary.csv")
        n_bad = sum(1 for r in an.load_rows(d) if r.get("status") == "error")
        if n_bad:
            print(f"   {n_bad} runs failed (status=error); see runs.csv")


if __name__ == "__main__":
    main()
