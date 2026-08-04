# nooa-harvey

Simple OO-agent script (built with [NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents)) that runs the Harvey task-solving loop inside the agent.

## Prerequisites

1. Clone and set up Harvey Labs (once):

	```bash
	git clone https://github.com/harveyai/harvey-labs.git
	cd harvey-labs
	./scripts/setup.sh
	```

2. Set API keys in `harvey-labs/.env` (at minimum `ANTHROPIC_API_KEY`).

3. Point this wrapper to your Harvey clone using either:
	- `HARVEY_LABS_PATH=/path/to/harvey-labs`
	- or `--harvey-path /path/to/harvey-labs`

## Quick start

Describe the tutorial task:

```bash
uv run python main.py describe
```

Run the task:

```bash
uv run python main.py run
```

Run + evaluate + generate report:

```bash
uv run python main.py run --eval --report
```

Solve a task end-to-end (run + eval + report + score summary):

```bash
uv run python main.py solve
```

Run and ask the NOOA agent to summarize the outcome:

```bash
uv run python main.py run --summary
```

## Common overrides

```bash
uv run python main.py run \
  --task corporate-ma/review-data-room-red-flag-review \
  --model anthropic/claude-sonnet-4-6 \
  --max-turns 200
```

Use a different solving model:

```bash
uv run python main.py run --summary --model openai/gpt-5.4
```

Dual-judge evaluation:

```bash
uv run python main.py run --eval --dual
```

Dual-judge solve:

```bash
uv run python main.py solve --dual
```

## Optimizing the prompt with GEPA's `optimize_anything`

`optimize_prompt.py` uses [GEPA's `optimize_anything` framework](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)
to optimize `HarveyBenchmarkAgent`'s system prompt (its class docstring). The task
instructions themselves are not a tunable candidate — they come from each Harvey
task's own `task_context()` and cannot be changed.

It runs in GEPA's *Generalization* mode against a single task category
(`--domain`, required): a set of training tasks from that category proposes and
scores candidate system prompts (each evaluation runs the real agent against a
real Harvey task, then Harvey's own judge), and a held-out validation set from the
same category selects the best-generalizing candidate. The optimized prompt is
only expected to generalize within that category. Every metric call costs real LLM
tokens/time, so start small.

Install the extra dependency once:

```bash
uv sync --extra optimize
```

Inspect the plan (train/val split, seed prompt) without spending anything:

```bash
uv run python optimize_prompt.py --domain corporate-ma --dry-run
```

Run the optimization:

```bash
uv run python optimize_prompt.py \
  --domain corporate-ma \
  --model anthropic/claude-sonnet-5 \
  --max-metric-calls 20 --max-workers 2
```

To keep costs low and iterate quickly, cap how many tasks are considered with
`--sample-tasks` (deterministically sampled before the train/val split):

```bash
uv run python optimize_prompt.py --domain corporate-ma --sample-tasks 4 --max-metric-calls 6
```

GEPA already prints iteration-level progress to the console by default (new best
candidates, rejected proposals, exceptions). For more visibility while it runs:
- `--display-progress-bar` — a live tqdm bar over evaluation rollouts.
- `--run-dir <dir>` — persists a `run_log.txt` (and other run state) to disk as it
  progresses, so you can `tail -f <dir>/run_log.txt` from another terminal.
- `-v`/`--verbose` — logs each agent tool call (`read`/`write`/`glob`/`grep`/`bash`)
  live, since every evaluation runs a real agent.

By default GEPA proposes one mutation per iteration (1 parent, 1 mutation). Use
`--sampling-strategy` to propose several candidates per iteration instead:
- `same-parent` (`--sampling-n`) — best-of-N mutations off the same parent.
- `independent` (`--sampling-n`) — N different parents, one mutation each.
- `pxn` (`--sampling-p`/`--sampling-n`) — P parents x N mutations each = P*N candidates.

Combine with `--selection-strategy` to decide which resulting proposals to keep
(`all` = keep every improving one, `best` = keep only the single best, `top-k` with
`--selection-k`), and raise `--max-workers` so the extra candidates are actually
evaluated concurrently:

```bash
uv run python optimize_prompt.py \
  --domain corporate-ma --sampling-strategy pxn --sampling-p 2 --sampling-n 3 \
  --selection-strategy top-k --selection-k 2 --max-workers 4
```

Each run writes its own `optimized_prompts_<domain>_<timestamp>.json` by default (pass
`--output <path>` to pick a fixed name instead — an existing file at that path gets
overwritten with a warning), so repeated runs never clobber each other. Use the
optimized prompt for a real run:

```bash
uv run python main.py solve --prompts optimized_prompts_corporate-ma_20260731-101500.json
```

## What "inside the agent" means

The solving loop is implemented entirely as methods on the NOOA agent class `HarveyBenchmarkAgent` in `main.py` — it does not shell out to the Harvey harness to solve the task. The agent gets real workspace tools, scoped to `results/<run-id>/` (a `documents/` tree mirroring the task's source files, plus a writable `output/` directory):

- `read(path)` — reads a text file, or auto-extracts text from a `.docx`
- `write(path, content)` — writes a text file, or auto-generates a real `.docx` if the path ends in `.docx`
- `edit(path, old_string, new_string)` — replaces one exact, unique string occurrence in a file
- `glob(pattern)` — lists workspace-relative files matching a glob pattern
- `grep(pattern, path=".")` — regex-searches file contents (including `.docx`) and returns matching lines
- `bash(command)` — runs a shell command with the workspace as cwd
- `task_context()` — returns the task title, instructions, deliverables, and criteria count

The agentic method `solve_harvey_task()` is the entrypoint: its docstring instructs the model to call these tools to inspect the task and produce every required deliverable under `output/`.

Only evaluation and reporting (`evaluation.run_eval`, `evaluation.report`) still shell out to the Harvey CLI, since that is Harvey's own judge/rubric logic.

