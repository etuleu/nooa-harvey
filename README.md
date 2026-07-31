# nooa-harvey

Simple OO-agent script (built with NOOA) that runs the Harvey task-solving loop inside the agent.

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

