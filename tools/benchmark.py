"""Compare agents using the unchanged official evaluator; not used by Agent.act."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_benchmark(runs):
    from agent import Agent
    from agent_template import Agent as Baseline
    from local_eval import evaluate_agent

    report = {
        "environment": "Organizer mock environment, unchanged",
        "warning": "Synthetic data; mock results do not predict judging scores.",
        "seeds": list(range(runs)),
        "agents": {},
    }
    for name, cls in (("template", Baseline), ("adaptive", Agent)):
        records = []
        for seed in range(runs):
            start = time.monotonic()
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                result = evaluate_agent(cls(), seed=seed, verbose=False)
            if result is None:
                raise RuntimeError(f"{name}, seed {seed}: evaluator returned no result")
            record = {
                "seed": seed,
                "net_arpu_gain": float(result["net_arpu_gain"]),
                "n_pilots": int(result["n_pilots"]),
                "seconds": round(time.monotonic() - start, 3),
                "evaluator_messages": log.getvalue().strip(),
            }
            records.append(record)
            print(f"{name:8s} seed={seed:2d} net={record['net_arpu_gain']:,.0f} "
                  f"pilots={record['n_pilots']} time={record['seconds']}s", flush=True)
            if name == "adaptive" and any(
                marker in record["evaluator_messages"]
                for marker in ("Агент упал", "отброшена")
            ):
                raise RuntimeError(record["evaluator_messages"])
        values = [r["net_arpu_gain"] for r in records]
        report["agents"][name] = {
            "runs": records,
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "positive_runs": sum(value > 0 for value in values),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    report = run_benchmark(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
