"""Sweep the *vanilla* (un-optimized) HarveyBenchmarkAgent harness -- i.e. no
--prompts/system_prompt override, just HarveyBenchmarkAgent's hardcoded class
docstring -- across every task in a domain (or an explicit/sampled subset) and
report which task scores lowest.

Useful as a baseline before running optimize_prompt.py, or to find the specific
task(s) the seed prompt struggles with most so you can inspect them by hand.

Each task run is a full real agent solve + judge eval, so it costs real LLM
tokens/time -- use --sample-tasks/--sample-fraction to cap the sweep, and start
with --max-workers 1 before parallelizing.

Usage:
    uv run python find_lowest_scoring_task.py --domain corporate-ma --sample-tasks 15

    # Sweep an explicit set of tasks:
    uv run python find_lowest_scoring_task.py \\
        --tasks corporate-ma/review-data-room-red-flag-review,corporate-ma/foo

    # Parallelize (each worker is a real LLM+CLI run, keep this modest):
    uv run python find_lowest_scoring_task.py --domain corporate-ma --sample-tasks 15 --max-workers 4
"""

import argparse
import asyncio
import concurrent.futures
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import main as harvey_main
from optimize_prompt import discover_tasks, sample_tasks

logger = logging.getLogger("nooa_harvey.find_lowest_scoring_task")


def run_one(harvey_path: Path, task: str, model: str, judge_model: str | None, dual: bool, max_turns: int) -> dict:
    """Run the vanilla harness end-to-end on one task and return a result dict with pass_rate."""
    started = time.perf_counter()
    print(f"  started: {task} @ {datetime.now().strftime('%H:%M:%S')}")
    run_id = f"vanilla-sweep/{harvey_main.sanitize_model_name(model)}/{task}/{uuid4().hex[:10]}"
    agent = harvey_main._build_agent(harvey_path=harvey_path, task=task, run_id=run_id, model=model)
    agent.write_run_config(model=model, max_turns=max_turns)
    agent.prepare_workspace()

    result: dict = {"task": task, "run_id": run_id, "pass_rate": 0.0}
    try:
        summary = asyncio.run(agent.solve_harvey_task())
    except Exception as exc:  # noqa: BLE001 - keep sweeping other tasks
        result.update(stage="solve", error=str(exc))
        result["elapsed_s"] = round(time.perf_counter() - started, 1)
        return result

    try:
        agent.run_eval_command(judge_model=judge_model, dual=dual)
    except Exception as exc:  # noqa: BLE001
        result.update(stage="eval", error=str(exc), agent_summary=summary[:500])
        result["elapsed_s"] = round(time.perf_counter() - started, 1)
        return result

    score_summary = agent.load_score_summary(dual=dual)
    if not score_summary:
        result.update(stage="score", error="no score artifact produced", agent_summary=summary[:500])
        result["elapsed_s"] = round(time.perf_counter() - started, 1)
        return result

    pass_rate = score_summary.get("pass_rate")
    result.update(
        pass_rate=float(pass_rate) if isinstance(pass_rate, (int, float)) else 0.0,
        passed=score_summary.get("passed"),
        total=score_summary.get("total"),
        all_pass=score_summary.get("all_pass"),
        judge_summary=score_summary.get("summary"),
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
        default=None,
        help="Cap the number of discovered tasks swept (deterministically sampled). Mutually exclusive with --sample-fraction.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Cap the discovered tasks to this fraction of the total. Mutually exclusive with --sample-tasks.",
    )
    parser.add_argument("--model", default=harvey_main.DEFAULT_MODEL, help="Solver model run by the vanilla agent.")
    parser.add_argument("--judge-model", help="Optional judge model override for evaluation.")
    parser.add_argument("--dual", action="store_true", help="Enable dual-judge evaluation mode.")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel evaluations. Keep low: each is a real LLM+CLI run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Where to write full sweep results. Defaults to 'vanilla_sweep_<domain>_<timestamp>.json'.")
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

    print(f"Vanilla harness sweep: {len(tasks)} task(s), model={args.model}")
    print(f"Tasks: {tasks}")

    if args.dry_run:
        print("\n--dry-run set: not running anything.")
        return

    results: list[dict] = []
    if args.max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(run_one, harvey_path, task, args.model, args.judge_model, args.dual, args.max_turns): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"  done: {result['task']} -> pass_rate={result.get('pass_rate')} ({result.get('elapsed_s')}s)")
    else:
        for task in tasks:
            result = run_one(harvey_path, task, args.model, args.judge_model, args.dual, args.max_turns)
            results.append(result)
            print(f"  done: {result['task']} -> pass_rate={result.get('pass_rate')} ({result.get('elapsed_s')}s)")

    results.sort(key=lambda r: r.get("pass_rate", 0.0))

    print("\nResults sorted by pass_rate (lowest first):")
    for r in results:
        passed_total = f"{r.get('passed')}/{r.get('total')} passed" if r.get("total") is not None else r.get("error", "?")
        print(f"  {r.get('pass_rate', 0.0):.3f}  {r['task']}  ({passed_total})")

    lowest = results[0]
    print(f"\nLowest-scoring task: {lowest['task']}")
    print(f"  pass_rate: {lowest.get('pass_rate')}")
    print(f"  run_id:    {lowest.get('run_id')}")
    if lowest.get("error"):
        print(f"  error ({lowest.get('stage')}): {lowest['error']}")
    if lowest.get("judge_summary"):
        print(f"  judge summary: {lowest['judge_summary']}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        domain_tag = args.domain or "custom"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path(f"vanilla_sweep_{domain_tag}_{timestamp}.json").resolve()
    output_path.write_text(
        json.dumps({"domain": args.domain, "model": args.model, "results": results}, indent=2), encoding="utf-8"
    )
    print(f"\nWrote full results to {output_path}")


if __name__ == "__main__":
    main()
