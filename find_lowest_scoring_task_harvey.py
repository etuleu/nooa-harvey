"""Find a low(ish)-scoring task by running Harvey Labs' OWN harness -- `harness.run`
+ `evaluation.run_eval` as subprocesses in the harvey-labs repo itself -- instead of
nooa's `HarveyBenchmarkAgent` (see find_lowest_scoring_task.py for that variant).

This is useful as a reference baseline: how hard is a given domain for Harvey's
reference agent loop / sandboxed harness, independent of anything in this repo?

Sweeps a deterministically-sampled subset of tasks from a domain (or an explicit
--tasks list); does NOT exhaustively search every task, so the result is *a*
low-scoring task from the sample, not necessarily the single lowest-scoring task
in the whole domain. Each task run is a real sandboxed agent solve + judge eval
(real LLM tokens/time), so keep --sample-tasks small and --max-workers modest.

Usage:
    uv run python find_lowest_scoring_task_harvey.py --domain corporate-ma --sample-tasks 10

    # Explicit tasks, higher parallelism:
    uv run python find_lowest_scoring_task_harvey.py \\
        --tasks corporate-ma/draft-spa-markup,corporate-ma/draft-vendor-due-diligence-report \\
        --max-workers 2

    # Preview the sampled task list without spending anything:
    uv run python find_lowest_scoring_task_harvey.py --domain corporate-ma --sample-tasks 10 --dry-run
"""

import argparse
import concurrent.futures
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import main as harvey_main
from optimize_prompt import discover_tasks, sample_tasks

logger = logging.getLogger("nooa_harvey.find_lowest_scoring_task_harvey")

PYTHON_MODULE_PREFIX = ["uv", "run", "python", "-m"]


def run_one(
    harvey_path: Path,
    task: str,
    model: str,
    reasoning_effort: str | None,
    judge_model: str,
    max_turns: int,
    timeout: int,
) -> dict:
    """Run Harvey's own harness.run + evaluation.run_eval on one task; return a result dict with pass_rate."""
    started = time.perf_counter()
    print(f"  started: {task} @ {datetime.now().strftime('%H:%M:%S')}")
    run_id = f"vanilla-sweep-harvey/{harvey_main.sanitize_model_name(model)}/{task}/{uuid4().hex[:10]}"
    result: dict = {"task": task, "run_id": run_id, "pass_rate": 0.0}

    run_cmd = [
        *PYTHON_MODULE_PREFIX, "harness.run",
        "--model", model,
        "--task", task,
        "--run-id", run_id,
        "--max-turns", str(max_turns),
    ]
    if reasoning_effort:
        run_cmd.extend(["--reasoning-effort", reasoning_effort])

    proc = subprocess.run(run_cmd, cwd=harvey_path, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        result.update(stage="run", error=(proc.stderr or proc.stdout)[-2000:])
        result["elapsed_s"] = round(time.perf_counter() - started, 1)
        return result

    eval_cmd = [
        *PYTHON_MODULE_PREFIX, "evaluation.run_eval",
        "--run-id", run_id,
        "--task", task,
        "--judge-model", judge_model,
    ]
    proc = subprocess.run(eval_cmd, cwd=harvey_path, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        result.update(stage="eval", error=(proc.stderr or proc.stdout)[-2000:])
        result["elapsed_s"] = round(time.perf_counter() - started, 1)
        return result

    scores_path = harvey_path / "results" / run_id / "scores.json"
    if not scores_path.exists():
        result.update(stage="score", error=f"no scores.json at {scores_path}")
        result["elapsed_s"] = round(time.perf_counter() - started, 1)
        return result

    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    n_criteria = scores.get("n_criteria")
    n_passed = scores.get("n_passed")
    pass_rate = (
        n_passed / n_criteria if isinstance(n_criteria, int) and n_criteria > 0 and isinstance(n_passed, int) else 0.0
    )
    result.update(
        pass_rate=pass_rate,
        passed=n_passed,
        total=n_criteria,
        all_pass=scores.get("all_pass"),
        summary=scores.get("summary"),
    )
    result["elapsed_s"] = round(time.perf_counter() - started, 1)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--harvey-path", help="Path to a local clone of harvey-labs.")
    parser.add_argument("--domain", help="Task category to sweep, i.e. tasks/<domain>/. Ignored when --tasks is given.")
    parser.add_argument("--tasks", help="Comma-separated task ids to sweep (overrides --domain discovery).")
    parser.add_argument(
        "--sample-tasks",
        type=int,
        default=10,
        help="Cap the number of discovered tasks swept (deterministically sampled). Mutually exclusive with --sample-fraction.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Cap the discovered tasks to this fraction of the total. Mutually exclusive with --sample-tasks.",
    )
    parser.add_argument("--model", default="claude-sonnet-5", help="Model identifier passed to harness.run (Harvey's own naming, e.g. claude-sonnet-5).")
    parser.add_argument("--reasoning-effort", default="high", help="Reasoning effort passed to harness.run (harness default: high).")
    parser.add_argument("--judge-model", default="claude-sonnet-4-6", help="Judge model passed to evaluation.run_eval.")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel task sweeps. Keep low: each is a real sandboxed LLM run.")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-subprocess timeout in seconds (applies separately to run and eval).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Where to write full sweep results. Defaults to 'vanilla_sweep_harvey_<domain>_<timestamp>.json'.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved task list and exit without running anything.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    harvey_main.configure_logging(args.verbose)
    harvey_path = harvey_main.resolve_harvey_path(args.harvey_path)

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        if not args.domain:
            raise SystemExit("Provide --domain (to discover tasks) or --tasks (an explicit list).")
        if args.sample_tasks is not None and args.sample_fraction is not None:
            raise SystemExit("Provide only one of --sample-tasks or --sample-fraction.")
        discovered = discover_tasks(harvey_path, args.domain)
        sample_size = args.sample_tasks
        if args.sample_fraction is not None:
            if not (0 < args.sample_fraction <= 1):
                raise SystemExit("--sample-fraction must be in (0, 1].")
            sample_size = max(1, round(len(discovered) * args.sample_fraction))
        tasks = sample_tasks(discovered, sample_size, args.seed)

    if not tasks:
        raise SystemExit("No tasks to sweep.")

    print(f"Harvey-harness sweep (sampled, not exhaustive): {len(tasks)} task(s), model={args.model}, reasoning={args.reasoning_effort}")
    print(f"Tasks: {tasks}")

    if args.dry_run:
        print("\n--dry-run set: not running anything.")
        return

    results: list[dict] = []
    if args.max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(
                    run_one, harvey_path, task, args.model, args.reasoning_effort, args.judge_model, args.max_turns, args.timeout
                ): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"  done: {result['task']} -> pass_rate={result.get('pass_rate')} ({result.get('elapsed_s')}s)")
    else:
        for task in tasks:
            result = run_one(harvey_path, task, args.model, args.reasoning_effort, args.judge_model, args.max_turns, args.timeout)
            results.append(result)
            print(f"  done: {result['task']} -> pass_rate={result.get('pass_rate')} ({result.get('elapsed_s')}s)")

    results.sort(key=lambda r: r.get("pass_rate", 0.0))

    print("\nResults sorted by pass_rate (lowest first, from the sampled subset):")
    for r in results:
        passed_total = f"{r.get('passed')}/{r.get('total')} passed" if r.get("total") is not None else r.get("error", "?")
        print(f"  {r.get('pass_rate', 0.0):.3f}  {r['task']}  ({passed_total})")

    lowest = results[0]
    print(f"\nLow-scoring task (from sample, not necessarily the domain minimum): {lowest['task']}")
    print(f"  pass_rate: {lowest.get('pass_rate')}")
    print(f"  run_id:    {lowest.get('run_id')}")
    if lowest.get("error"):
        print(f"  error ({lowest.get('stage')}): {lowest['error']}")
    if lowest.get("summary"):
        print(f"  judge summary: {lowest['summary']}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        domain_tag = args.domain or "custom"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path(f"vanilla_sweep_harvey_{domain_tag}_{timestamp}.json").resolve()
    output_path.write_text(
        json.dumps({"domain": args.domain, "model": args.model, "reasoning_effort": args.reasoning_effort, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote full results to {output_path}")


if __name__ == "__main__":
    main()
