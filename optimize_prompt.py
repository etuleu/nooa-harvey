"""Optimize the HarveyBenchmarkAgent's system prompt using GEPA's
`optimize_anything` framework
(https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/).

This runs in *Generalization* mode: a set of training tasks is used to propose and
score candidate system prompts, a held-out validation set is used for candidate
selection, and the optimized prompt is expected to generalize to unseen Harvey Labs
tasks *within the same task category* (--domain).

Only the agent's system prompt is optimized:
  - "system_prompt" -> HarveyBenchmarkAgent's class docstring (resolved at call
                        time via MRO by nooa's Agent._resolve_system_prompt).

The task instructions themselves are not a tunable candidate: they come from the
Harvey task's own task_context() (title, work_type, deliverables, criteria) and
cannot be changed.

Each evaluation actually runs the agent end-to-end against a real Harvey Labs task
(read/write/edit/glob/grep/bash tool calls against LLM(s)), then shells out to
Harvey's judge (`evaluation.run_eval`) to score it — so every metric call costs real
LLM tokens/time. Start with a small --max-metric-calls budget.

Usage:
    uv run --extra optimize python optimize_prompt.py \\
        --domain corporate-ma \\
        --model anthropic/claude-sonnet-5 \\
        --max-metric-calls 20 --max-workers 2

    # Keep costs low / iterate quickly by capping how many tasks are considered:
    uv run --extra optimize python optimize_prompt.py \\
        --domain corporate-ma --sample-tasks 4 --max-metric-calls 6

    # PxN proposal sampling: 2 parents x 3 mutations each = 6 candidates/iteration,
    # keep the top 2, evaluated concurrently (raise --max-workers to parallelize):
    uv run --extra optimize python optimize_prompt.py \\
        --domain corporate-ma --sampling-strategy pxn --sampling-p 2 --sampling-n 3 \\
        --selection-strategy top-k --selection-k 2 --max-workers 4

    # Inspect the plan (train/val split, seed prompt) without spending anything:
    uv run --extra optimize python optimize_prompt.py --domain corporate-ma --dry-run

Each run writes its own `optimized_prompts_<domain>_<timestamp>.json` by default (pass
--output to pick a fixed path instead) so repeated runs never clobber each other.
Once you have one, use it for real runs:
    uv run python main.py solve --prompts optimized_prompts_corporate-ma_20260731-101500.json
"""

import argparse
import asyncio
import json
import logging
import shlex
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import main as harvey_main

logger = logging.getLogger("nooa_harvey.optimize_prompt")

OBJECTIVE = (
    "Improve the system prompt for an autonomous document-review agent (NOOA "
    "`HarveyBenchmarkAgent`) so that it maximizes the fraction of Harvey Labs "
    "benchmark rubric criteria satisfied on tasks in the {domain!r} category, "
    "while producing every required deliverable file under output/."
)

BACKGROUND = """\
The candidate is a single text component:
  - "system_prompt": the agent's class docstring. It is resolved at call time by
    walking the class MRO and formatting `{expr}` placeholders as Python expressions
    evaluated with `self` in scope (e.g. `{type(self).__name__}`). Keep any such
    placeholders syntactically valid, and avoid introducing stray literal `{` or `}`
    characters that are not part of an intended placeholder.

The task instructions themselves are NOT part of the candidate and cannot be
changed: they are fixed per-task and come from the Harvey task's own
task_context() (title, instructions, work_type, deliverables). The system prompt
should give the agent general strategy/behavior guidance that applies across every
task in the category, not attempt to restate or override any specific task's
instructions.

The agent has these tools, scoped to a per-run workspace with a read-only
documents/ tree (mirroring the task's source documents) and a writable output/
directory:
  - task_context() -> title, instructions, work_type, deliverables, criteria_count
  - read(path) -> reads a text file, or auto-extracts text from a .docx
  - write(path, content) -> writes a text file, or auto-generates a real .docx if
    the path ends in .docx
  - edit(path, old_string, new_string) -> replaces one exact, unique string
    occurrence in a file
  - glob(pattern) -> lists workspace-relative files matching a glob pattern
  - grep(pattern, path=".") -> regex-searches file contents (including .docx)
  - bash(command) -> runs a shell command with the workspace as cwd

Every deliverable named in task_context()["deliverables"] must be written under
output/. Criteria are graded by an LLM judge against a rubric defined per task; the
evaluator reports the fraction of criteria passed (pass_rate) as the score, plus the
judge's summary text and the agent's own run summary as diagnostic feedback (ASI).
Prompts should push the agent to be thorough (inspect all relevant source documents
before writing deliverables), precise (match deliverable filenames/formats exactly),
and to actually finish (write every required deliverable, not just a subset).
"""


def discover_tasks(harvey_path: Path, domain: str | None) -> list[str]:
    """List task ids (relative to harvey_path/tasks) with a task.json, optionally scoped to `domain`."""
    tasks_root = harvey_path / "tasks"
    search_root = (tasks_root / domain) if domain else tasks_root
    if not search_root.exists():
        raise FileNotFoundError(f"No tasks found under {search_root}")
    task_ids = sorted(
        str(task_json.parent.relative_to(tasks_root)).replace("\\", "/")
        for task_json in search_root.rglob("task.json")
    )
    if not task_ids:
        raise FileNotFoundError(f"No task.json files found under {search_root}")
    return task_ids


def sample_tasks(task_ids: list[str], sample_size: int | None, seed: int) -> list[str]:
    """Deterministically shuffle and truncate the discovered task pool to at most `sample_size` tasks.

    Used to keep costs low and iterate quickly (each task costs one real agent run + judge call
    per training example, times however many metric calls are spent on it).
    """
    import random

    if not sample_size or sample_size >= len(task_ids):
        return task_ids
    shuffled = list(task_ids)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:sample_size]


def split_train_val(task_ids: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    import random

    if len(task_ids) < 2:
        return task_ids, task_ids
    shuffled = list(task_ids)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    n_val = min(n_val, len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]


def build_seed_candidate() -> str:
    system_prompt = harvey_main.HarveyBenchmarkAgent.__doc__ or ""
    if not system_prompt:
        raise RuntimeError("Could not read seed system prompt from HarveyBenchmarkAgent")
    return system_prompt


def make_evaluator(harvey_path: Path, model: str, judge_model: str | None, dual: bool, max_turns: int):
    def evaluate(candidate: str, example: dict[str, str]):
        import gepa.optimize_anything as oa

        task = example["task"]
        run_id = f"gepa-optimize/{harvey_main.sanitize_model_name(model)}/{task}/{uuid4().hex[:10]}"
        agent = harvey_main._build_agent(
            harvey_path=harvey_path, task=task, run_id=run_id, model=model, system_prompt=candidate
        )
        agent.write_run_config(model=model, max_turns=max_turns)
        agent.prepare_workspace()

        oa.log(f"task={task} run_id={run_id}")
        try:
            summary = asyncio.run(agent.solve_harvey_task())
        except Exception as exc:  # noqa: BLE001 - surface any failure as ASI, not a crash
            oa.log(f"solve_harvey_task() raised: {exc!r}")
            return 0.0, {"task": task, "run_id": run_id, "stage": "solve", "error": str(exc)}

        try:
            agent.run_eval_command(judge_model=judge_model, dual=dual)
        except Exception as exc:  # noqa: BLE001
            oa.log(f"evaluation failed: {exc!r}")
            return 0.0, {
                "task": task,
                "run_id": run_id,
                "stage": "eval",
                "error": str(exc),
                "agent_summary": summary[:1500],
            }

        score_summary = agent.load_score_summary(dual=dual)
        if not score_summary:
            return 0.0, {
                "task": task,
                "run_id": run_id,
                "stage": "score",
                "error": "no score artifact produced",
                "agent_summary": summary[:1500],
            }

        pass_rate = score_summary.get("pass_rate")
        score = float(pass_rate) if isinstance(pass_rate, (int, float)) else 0.0
        side_info = {
            "task": task,
            "run_id": run_id,
            "pass_rate": pass_rate,
            "passed": score_summary.get("passed"),
            "failed": score_summary.get("failed"),
            "total": score_summary.get("total"),
            "all_pass": score_summary.get("all_pass"),
            "judge_summary": score_summary.get("summary"),
            "agent_summary": summary[:1500],
        }
        return score, side_info

    return evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--harvey-path", help="Path to a local clone of harvey-labs.")
    parser.add_argument(
        "--domain",
        required=True,
        help="Single task category to optimize for, i.e. tasks/<domain>/. The optimized prompt "
        "is only expected to generalize within this category.",
    )
    parser.add_argument("--train-tasks", help="Comma-separated task ids to use as the training set (overrides discovery).")
    parser.add_argument("--val-tasks", help="Comma-separated task ids to use as the held-out validation set (overrides discovery).")
    parser.add_argument(
        "--sample-tasks",
        type=int,
        default=None,
        help="Cap the number of discovered tasks considered (deterministically sampled) before the "
        "train/val split. Use this to keep costs low and iterate quickly. Ignored when "
        "--train-tasks/--val-tasks are given. Mutually exclusive with --sample-fraction.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Cap the discovered tasks to this fraction of the total (e.g. 0.1 for 10%%), rounded to "
        "at least 2. Deterministically sampled before the train/val split, same as --sample-tasks. "
        "Mutually exclusive with --sample-tasks.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.3, help="Fraction of discovered tasks held out for validation.")
    parser.add_argument("--model", default=harvey_main.DEFAULT_MODEL, help="Solver model run by the candidate agent.")
    parser.add_argument("--reflection-model", default="anthropic/claude-opus-4-6", help="LLM used by GEPA to propose improved candidates.")
    parser.add_argument("--judge-model", help="Optional judge model override for evaluation.")
    parser.add_argument("--dual", action="store_true", help="Enable dual-judge evaluation mode.")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--max-metric-calls", type=int, default=20, help="Evaluation budget (each call runs one real task).")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel evaluations. Keep low: each is a real LLM+CLI run.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the optimized candidate. Defaults to a unique "
        "'optimized_prompts_<domain>_<timestamp>.json' so repeated runs don't clobber each other's results.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan (tasks, seed prompts) and exit without optimizing.")
    parser.add_argument(
        "--display-progress-bar",
        action="store_true",
        help="Show a live tqdm progress bar over evaluation rollouts while optimizing.",
    )
    parser.add_argument(
        "--run-dir",
        help="Directory where GEPA persists run state and a run_log.txt as it progresses "
        "(useful to `tail -f` while it runs, or to inspect/resume afterward).",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["single", "same-parent", "independent", "pxn"],
        default="single",
        help="How many candidate mutations GEPA proposes per iteration (default: single, GEPA's "
        "original 1-parent/1-mutation behavior). 'same-parent': best-of-N mutations off one "
        "parent (--sampling-n). 'independent': N different parents, one mutation each "
        "(--sampling-n). 'pxn': P parents x N mutations each = P*N tasks (--sampling-p/--sampling-n). "
        "Tasks are evaluated concurrently up to --max-workers, so raise --max-workers to actually "
        "parallelize them.",
    )
    parser.add_argument(
        "--sampling-p",
        type=int,
        default=2,
        help="Number of parents for --sampling-strategy pxn.",
    )
    parser.add_argument(
        "--sampling-n",
        type=int,
        default=2,
        help="Mutations per parent for --sampling-strategy pxn/same-parent, or parent count for independent.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["all", "best", "top-k"],
        default="all",
        help="Which proposals to keep after a multi-task sampling iteration. 'all' (default): keep "
        "every proposal that improves on its parent. 'best': keep only the single best-improving "
        "proposal. 'top-k': keep the top --selection-k by improvement margin.",
    )
    parser.add_argument(
        "--selection-k",
        type=int,
        default=2,
        help="Number of proposals to keep with --selection-strategy top-k.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def build_sampling_strategy(args: argparse.Namespace):
    from gepa.strategies.proposal_sampling import IndependentSampling, PxNSampling, SameParentSampling

    if args.sampling_strategy == "single":
        return None  # engine defaults to SingleMutationSampling
    if args.sampling_strategy == "same-parent":
        return SameParentSampling(args.sampling_n)
    if args.sampling_strategy == "independent":
        return IndependentSampling(args.sampling_n)
    if args.sampling_strategy == "pxn":
        return PxNSampling(p=args.sampling_p, n=args.sampling_n)
    raise ValueError(f"Unknown --sampling-strategy: {args.sampling_strategy}")


def build_selection_strategy(args: argparse.Namespace):
    from gepa.strategies.proposal_selection import BestImprovement, TopKImprovements

    if args.selection_strategy == "all":
        return None  # engine defaults to AllImprovements
    if args.selection_strategy == "best":
        return BestImprovement()
    if args.selection_strategy == "top-k":
        return TopKImprovements(args.selection_k)
    raise ValueError(f"Unknown --selection-strategy: {args.selection_strategy}")


def main() -> None:
    args = build_parser().parse_args()
    harvey_main.configure_logging(args.verbose)
    print(f"Command: {shlex.join(sys.argv)}")
    harvey_path = harvey_main.resolve_harvey_path(args.harvey_path)

    if args.train_tasks or args.val_tasks:
        if not (args.train_tasks and args.val_tasks):
            raise SystemExit("Provide both --train-tasks and --val-tasks, or neither (for auto-discovery).")
        train_tasks = [t.strip() for t in args.train_tasks.split(",") if t.strip()]
        val_tasks = [t.strip() for t in args.val_tasks.split(",") if t.strip()]
    else:
        if args.sample_tasks is not None and args.sample_fraction is not None:
            raise SystemExit("Provide only one of --sample-tasks or --sample-fraction.")
        if args.sample_tasks is not None and args.sample_tasks < 2:
            raise SystemExit("--sample-tasks must be at least 2 (need at least one train and one val task).")
        if args.sample_fraction is not None and not (0 < args.sample_fraction <= 1):
            raise SystemExit("--sample-fraction must be in (0, 1].")
        discovered = discover_tasks(harvey_path, args.domain)
        sample_size = args.sample_tasks
        if args.sample_fraction is not None:
            sample_size = max(2, round(len(discovered) * args.sample_fraction))
            print(f"--sample-fraction {args.sample_fraction} of {len(discovered)} discovered tasks -> sample_size={sample_size}")
        sampled = sample_tasks(discovered, sample_size, args.seed)
        train_tasks, val_tasks = split_train_val(sampled, args.val_fraction, args.seed)

    seed_candidate = build_seed_candidate()

    print(f"Harvey path:   {harvey_path}")
    print(f"Train tasks ({len(train_tasks)}): {train_tasks}")
    print(f"Val tasks   ({len(val_tasks)}): {val_tasks}")
    print(f"Domain:            {args.domain}")
    print(f"Solver model:      {args.model}")
    print(f"Reflection model:  {args.reflection_model}")
    print(f"Max metric calls:  {args.max_metric_calls}")
    print(f"Max workers:       {args.max_workers}")
    if args.sampling_strategy == "pxn":
        print(f"Sampling strategy: pxn (p={args.sampling_p}, n={args.sampling_n} -> {args.sampling_p * args.sampling_n} tasks/iteration)")
    elif args.sampling_strategy in ("same-parent", "independent"):
        print(f"Sampling strategy: {args.sampling_strategy} (n={args.sampling_n})")
    else:
        print(f"Sampling strategy: {args.sampling_strategy}")
    selection_desc = f"top-k (k={args.selection_k})" if args.selection_strategy == "top-k" else args.selection_strategy
    print(f"Selection strategy: {selection_desc}")
    print("Seed system_prompt (first 200 chars):")
    print("  " + seed_candidate[:200].replace("\n", " "))

    if args.dry_run:
        print("\n--dry-run set: not calling optimize_anything.")
        return

    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    train_examples = [{"task": t} for t in train_tasks]
    val_examples = [{"task": t} for t in val_tasks]

    evaluator = make_evaluator(
        harvey_path=harvey_path, model=args.model, judge_model=args.judge_model, dual=args.dual, max_turns=args.max_turns
    )

    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=args.max_metric_calls,
            parallel=args.max_workers > 1,
            max_workers=args.max_workers,
            seed=args.seed,
            display_progress_bar=args.display_progress_bar,
            run_dir=args.run_dir,
            sampling_strategy=build_sampling_strategy(args),
            selection_strategy=build_selection_strategy(args),
        ),
        reflection=ReflectionConfig(reflection_lm=args.reflection_model),
    )

    result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=evaluator,
        dataset=train_examples,
        valset=val_examples,
        objective=OBJECTIVE.format(domain=args.domain),
        background=BACKGROUND,
        config=config,
    )

    best = result.best_candidate
    best_score = result.val_aggregate_scores[result.best_idx]
    output_payload = {
        "system_prompt": best if isinstance(best, str) else str(best),
        "domain": args.domain,
        "best_score": best_score,
        "train_tasks": train_tasks,
        "val_tasks": val_tasks,
    }

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        if output_path.exists():
            print(f"Warning: overwriting existing file at {output_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path(f"optimized_prompts_{args.domain}_{timestamp}.json").resolve()
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print(f"\nBest score: {best_score}")
    print(f"Total metric calls: {result.total_metric_calls}")
    print(f"Wrote optimized prompts to {output_path}")
    print(f"Try it with: uv run python main.py solve --prompts {output_path}")


if __name__ == "__main__":
    main()
