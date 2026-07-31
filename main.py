import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from dotenv import load_dotenv

from nooa import Agent, CodeActStrategy, strategy
from nooa.config.strategy_config import CodeActConfig
from nooa.unifiedllm.registry import get_llm_client

load_dotenv(Path(__file__).resolve().parent / ".env")


logger = logging.getLogger("nooa_harvey")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # litellm is dumb
    for noisy_logger in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "httpx", "httpcore"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)


DEFAULT_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_TASK = "corporate-ma/review-data-room-red-flag-review"

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".log",
    ".py",
    ".js",
    ".ts",
}


def resolve_harvey_path(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg).expanduser().resolve()
    elif os.getenv("HARVEY_LABS_PATH"):
        path = Path(os.environ["HARVEY_LABS_PATH"]).expanduser().resolve()
    else:
        path = (Path(__file__).resolve().parent / "../harvey-labs").resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Harvey repo path not found: {path}. Set HARVEY_LABS_PATH or pass --harvey-path."
        )
    return path


def sanitize_model_name(model: str) -> str:
    name = model.split("/", maxsplit=1)[-1]
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name)


def default_run_id(task: str, model: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{task}/{sanitize_model_name(model)}/{timestamp}"


def _extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml_bytes = zf.read("word/document.xml")

    xml_text = xml_bytes.decode("utf-8", errors="ignore")
    # Turn paragraph breaks into newlines before stripping tags.
    xml_text = xml_text.replace("</w:p>", "\n")
    xml_text = re.sub(r"<[^>]+>", "", xml_text)
    return xml_text


class HarveyBenchmarkAgent(Agent):
    """You solve Harvey Labs benchmark tasks using read, write, edit, glob, grep, and bash tools scoped to a per-run workspace."""

    harvey_path: Path
    task: str
    run_id: str

    def task_dir(self) -> Path:
        path = self.harvey_path / "tasks" / self.task
        if not path.exists():
            raise FileNotFoundError(f"Task not found: {path}")
        return path

    def load_task_json(self) -> dict[str, object]:
        task_json_path = self.task_dir() / "task.json"
        return json.loads(task_json_path.read_text(encoding="utf-8"))

    def task_context(self) -> dict[str, object]:
        """Return the task title, instructions, required deliverables, and criteria count."""
        task_json = self.load_task_json()
        criteria = task_json.get("criteria", [])
        return {
            "title": task_json.get("title"),
            "task_id": self.task,
            "work_type": task_json.get("work_type"),
            "instructions": task_json.get("instructions"),
            "deliverables": task_json.get("deliverables", {}),
            "criteria_count": len(criteria) if isinstance(criteria, list) else 0,
        }

    def results_dir(self) -> Path:
        return self.harvey_path / "results" / self.run_id

    def output_dir(self) -> Path:
        path = self.results_dir() / "output"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def workspace_root(self) -> Path:
        return self.results_dir()

    def _register_metrics_hook(self) -> None:
        """Subscribe to LLMComplete events to log per-turn token/cost metrics (reasoning tokens included) at DEBUG level."""
        if getattr(self, "_metrics_hook_registered", False):
            return
        self.event_manager.on("LLMComplete", self._on_llm_complete)
        self._metrics_hook_registered = True

    def _on_llm_complete(self, event: object) -> None:
        logger.debug(
            "LLM turn complete: model=%s prompt=%d completion=%d reasoning=%d cached=%d cost=$%.4f",
            getattr(event, "model_name", ""),
            getattr(event, "prompt_tokens", 0),
            getattr(event, "completion_tokens", 0),
            getattr(event, "reasoning_tokens", 0),
            getattr(event, "cached_tokens", 0),
            getattr(event, "cost_usd", 0.0),
        )

    def prepare_workspace(self) -> Path:
        """Set up the per-run workspace: a read-only documents/ tree mirroring the task's source documents, and a writable output/ directory."""
        logger.info("Preparing workspace for run_id=%s", self.run_id)
        workspace = self.workspace_root()
        workspace.mkdir(parents=True, exist_ok=True)

        docs_link = workspace / "documents"
        source_docs = self.task_dir() / "documents"
        if not docs_link.exists():
            try:
                docs_link.symlink_to(source_docs, target_is_directory=True)
            except OSError:
                shutil.copytree(source_docs, docs_link)

        self.output_dir()
        logger.info("Workspace ready at %s", workspace)
        return workspace

    def _resolve_in_workspace(self, relative_path: str) -> Path:
        root = self.workspace_root().resolve()
        target = (root / relative_path).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Path escapes workspace: {relative_path}")
        return target

    def _context_stats_line(self) -> str:
        """One-line summary of current context-window utilization (tokens/window, event count)."""
        stats = self.context_stats
        if stats is None or stats.prompt_tokens is None:
            return "context: n/a (no completed turn yet)"
        window = stats.model_context_window
        if window:
            pct = 100 * stats.prompt_tokens / window
            return f"context: {stats.prompt_tokens}/{window} tokens ({pct:.1f}%), events={stats.events_count}"
        return f"context: {stats.prompt_tokens} tokens, events={stats.events_count}"

    def read(self, path: str, max_chars: int = 20000) -> str:
        """Read a file from the workspace (documents/... or output/...), returning extracted text."""
        logger.info("tool: read(path=%r, max_chars=%d) | %s", path, max_chars, self._context_stats_line())
        target = self._resolve_in_workspace(path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        if target.suffix.lower() == ".docx":
            content = _extract_docx_text(target)
        elif target.suffix.lower() in TEXT_EXTENSIONS:
            content = target.read_text(encoding="utf-8", errors="ignore")
        else:
            logger.debug("read(%s) -> unsupported binary file", path)
            return f"[Unsupported binary file for direct read: {path}; use bash for extraction if needed]"

        content = content[:max_chars]
        logger.debug("read(%s) -> %d chars", path, len(content))
        return content

    def write(self, path: str, content: str) -> str:
        """Write a file into the workspace (typically output/<deliverable>). .docx files are generated automatically from plain text content."""
        logger.info("tool: write(path=%r, content_len=%d) | %s", path, len(content), self._context_stats_line())
        target = self._resolve_in_workspace(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.suffix.lower() == ".docx":
            self._write_docx(target, content)
        else:
            target.write_text(content, encoding="utf-8")

        logger.debug("write(%s) -> %s", path, target)
        return str(target)

    def edit(self, path: str, old_string: str, new_string: str) -> str:
        """Replace an exact, unique string occurrence in a workspace text file."""
        logger.info("tool: edit(path=%r) | %s", path, self._context_stats_line())
        target = self._resolve_in_workspace(path)
        if target.suffix.lower() == ".docx":
            raise ValueError("edit() does not support .docx files; use write() to regenerate them")
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        content = target.read_text(encoding="utf-8", errors="ignore")
        occurrences = content.count(old_string)
        if occurrences == 0:
            raise ValueError(f"old_string not found in {path}")
        if occurrences > 1:
            raise ValueError(f"old_string is not unique in {path} ({occurrences} matches)")

        target.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
        logger.debug("edit(%s) -> applied", path)
        return str(target)

    def glob(self, pattern: str) -> list[str]:
        """List workspace-relative file paths matching a glob pattern, e.g. 'documents/**/*.docx'."""
        logger.info("tool: glob(pattern=%r) | %s", pattern, self._context_stats_line())
        root = self.workspace_root()
        matches = [
            str(path.relative_to(root)).replace("\\", "/")
            for path in root.glob(pattern)
            if path.is_file()
        ]
        matches.sort()
        logger.debug("glob(%s) -> %d matches", pattern, len(matches))
        return matches

    def grep(self, pattern: str, path: str = ".", max_results: int = 200) -> list[str]:
        """Regex search text file contents under a workspace-relative path, returning 'file:line: text' matches."""
        logger.info("tool: grep(pattern=%r, path=%r) | %s", pattern, path, self._context_stats_line())
        search_root = self._resolve_in_workspace(path)
        regex = re.compile(pattern)
        results: list[str] = []

        candidates = [search_root] if search_root.is_file() else sorted(search_root.rglob("*"))
        for file_path in candidates:
            if len(results) >= max_results:
                break
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in TEXT_EXTENSIONS and file_path.suffix.lower() != ".docx":
                continue

            try:
                text = (
                    _extract_docx_text(file_path)
                    if file_path.suffix.lower() == ".docx"
                    else file_path.read_text(encoding="utf-8", errors="ignore")
                )
            except (OSError, KeyError):
                continue

            rel = str(file_path.relative_to(self.workspace_root())).replace("\\", "/")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    results.append(f"{rel}:{line_no}: {line.strip()}")
                    if len(results) >= max_results:
                        break

        logger.debug("grep(%s) -> %d matches", pattern, len(results))
        return results

    def bash(self, command: str, timeout: int = 60) -> str:
        """Run a shell command with the workspace directory as cwd. Returns combined stdout/stderr."""
        logger.info("tool: bash(command=%r) | %s", command, self._context_stats_line())
        started = time.perf_counter()
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=self.workspace_root(),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - started
        output = result.stdout
        if result.stderr:
            output += result.stderr
        if result.returncode != 0:
            output += f"\n[exit code {result.returncode}]"
        logger.debug("bash(%s) -> exit %d in %.2fs", command, result.returncode, elapsed)
        return output

    def _write_docx(self, output_path: Path, content: str) -> None:
        content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
        rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
        document_xml = self._build_minimal_docx_xml(content)

        with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types_xml)
            zf.writestr("_rels/.rels", rels_xml)
            zf.writestr("word/document.xml", document_xml)

    def _build_minimal_docx_xml(self, content: str) -> str:
        paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
        if not paragraphs:
            paragraphs = ["No content generated."]

        p_xml = ""
        for paragraph in paragraphs:
            p_xml += (
                "<w:p><w:r><w:t xml:space=\"preserve\">"
                f"{xml_escape(paragraph)}"
                "</w:t></w:r></w:p>"
            )

        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
            "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
            "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
            "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
            "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
            "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
            "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
            "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
            "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
            "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
            "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
            "xmlns:w15=\"http://schemas.microsoft.com/office/word/2012/wordml\" "
            "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
            "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
            "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
            "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
            "mc:Ignorable=\"w14 w15 wp14\">"
            "<w:body>"
            f"{p_xml}"
            "<w:sectPr><w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
            "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
            "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/></w:sectPr>"
            "</w:body></w:document>"
        )

    def write_run_config(self, model: str, max_turns: int) -> str:
        config = {
            "run_id": self.run_id,
            "task": self.task,
            "model": model,
            "max_turns": max_turns,
            "runner": "nooa-harvey-inline-loop",
        }
        config_path = self.results_dir() / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return str(config_path)

    def run_eval_command(self, judge_model: str | None, dual: bool) -> None:
        eval_args = [
            "uv",
            "run",
            "python",
            "-m",
            "evaluation.run_eval",
            "--run-id",
            self.run_id,
            "--task",
            self.task,
        ]
        if judge_model:
            eval_args.extend(["--judge-model", judge_model])
        if dual:
            eval_args.append("--dual")
        self._run_harvey_cli(eval_args)

    def run_report_command(self) -> None:
        self._run_harvey_cli(
            ["uv", "run", "python", "-m", "evaluation.report", "--run-id", self.run_id]
        )

    def describe_task(self) -> None:
        self._run_harvey_cli(["uv", "run", "python", "-m", "utils.describe_task", self.task])

    def _run_harvey_cli(self, command: list[str]) -> str:
        logger.info("Running Harvey CLI: %s", " ".join(command))
        print(f"\n$ {' '.join(command)}")
        started = time.perf_counter()
        result = subprocess.run(command, cwd=self.harvey_path, text=True, capture_output=True)
        elapsed = time.perf_counter() - started

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            logger.error("Harvey CLI failed after %.2fs: %s", elapsed, " ".join(command))
            raise RuntimeError(
                f"Command failed with exit code {result.returncode}: {' '.join(command)}"
            )

        logger.info("Harvey CLI completed in %.2fs: %s", elapsed, " ".join(command))
        return result.stdout

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=200)))
    async def solve_harvey_task(self) -> str:
        """Solve the assigned Harvey task end to end. Call task_context() first to read the instructions, work type, and required deliverables. Use glob() and read() to inspect documents/ (a .docx source may need read() which auto-extracts its text). Use write() to create each required deliverable under output/<filename>, and edit() or bash() as needed to refine files. Return a short summary of what was produced and which deliverables were written."""
        ...

    def _score_file_for_mode(self, dual: bool) -> str:
        return "scores_dual.json" if dual else "scores.json"

    def load_score_summary(self, dual: bool) -> dict[str, object] | None:
        score_file = self._score_file_for_mode(dual)
        score_path = self.results_dir() / score_file
        if not score_path.exists():
            return None

        payload = json.loads(score_path.read_text(encoding="utf-8"))
        n_criteria = payload.get("n_criteria")
        n_passed = payload.get("n_passed")
        n_failed = (
            n_criteria - n_passed if isinstance(n_criteria, int) and isinstance(n_passed, int) else None
        )
        pass_rate = (
            n_passed / n_criteria if isinstance(n_criteria, int) and n_criteria > 0 and isinstance(n_passed, int) else None
        )
        return {
            "score_file": str(score_path),
            "all_pass": payload.get("all_pass"),
            "score": payload.get("score"),
            "pass_rate": pass_rate,
            "passed": n_passed,
            "failed": n_failed,
            "total": n_criteria,
            "summary": payload.get("summary"),
        }

    async def summarize_run(self) -> str:
        """Summarize the Harvey benchmark run outcome in at most 5 bullets, including next actions."""
        ...


def load_system_prompt_override(path: str | None) -> str | None:
    """Load a GEPA-optimized system prompt candidate (see optimize_prompt.py) from a JSON file.

    Expects an object with a ``system_prompt`` string key. Task instructions come from
    the task itself (via task_context()) and are not overridable.
    """
    if not path:
        return None
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    system_prompt = payload.get("system_prompt")
    if not system_prompt:
        raise ValueError(f"No usable system_prompt found in {path}")
    return system_prompt


def _build_agent(
    harvey_path: Path,
    task: str,
    run_id: str,
    model: str,
    system_prompt: str | None = None,
) -> HarveyBenchmarkAgent:
    logger.info("Building agent: task=%s model=%s run_id=%s", task, model, run_id)
    llm = get_llm_client(model)

    class ConcreteHarveyBenchmarkAgent(HarveyBenchmarkAgent, llm=llm):
        pass

    if system_prompt:
        ConcreteHarveyBenchmarkAgent.__doc__ = system_prompt
        logger.info("Applied GEPA-optimized system prompt override")

    agent = ConcreteHarveyBenchmarkAgent()
    agent.harvey_path = harvey_path
    agent.task = task
    agent.run_id = run_id
    agent._register_metrics_hook()
    return agent


def describe_task(args: argparse.Namespace) -> None:
    harvey_path = resolve_harvey_path(args.harvey_path)
    agent = _build_agent(harvey_path=harvey_path, task=args.task, run_id="describe", model=DEFAULT_MODEL)
    agent.describe_task()


def run_task(args: argparse.Namespace) -> None:
    harvey_path = resolve_harvey_path(args.harvey_path)
    logger.info("Resolved Harvey Labs path: %s", harvey_path)
    run_id = args.run_id or default_run_id(task=args.task, model=args.model)
    system_prompt = load_system_prompt_override(args.prompts)
    agent = _build_agent(
        harvey_path=harvey_path, task=args.task, run_id=run_id, model=args.model, system_prompt=system_prompt
    )

    agent.write_run_config(model=args.model, max_turns=args.max_turns)
    agent.prepare_workspace()

    logger.info("Starting solve_harvey_task() agent loop (max_turns=%d)...", args.max_turns)
    started = time.perf_counter()
    summary = asyncio.run(agent.solve_harvey_task())
    logger.info(
        "solve_harvey_task() finished in %.2fs | %s",
        time.perf_counter() - started,
        agent._context_stats_line(),
    )

    transcript_path = agent.results_dir() / "transcript.txt"
    transcript_path.write_text(summary, encoding="utf-8")
    logger.info("Wrote transcript to %s", transcript_path)

    if args.eval:
        logger.info("Running evaluation...")
        agent.run_eval_command(judge_model=args.judge_model, dual=args.dual)

    if args.report:
        logger.info("Generating report...")
        agent.run_report_command()

    print(f"Detected run ID: {run_id}")

    if args.summary:
        logger.info("Requesting run summary from agent...")
        run_summary = asyncio.run(agent.summarize_run())
        print("\nRun summary:")
        print(run_summary)


def solve_task(args: argparse.Namespace) -> None:
    harvey_path = resolve_harvey_path(args.harvey_path)
    logger.info("Resolved Harvey Labs path: %s", harvey_path)
    run_id = args.run_id or default_run_id(task=args.task, model=args.model)
    system_prompt = load_system_prompt_override(args.prompts)
    agent = _build_agent(
        harvey_path=harvey_path, task=args.task, run_id=run_id, model=args.model, system_prompt=system_prompt
    )

    agent.write_run_config(model=args.model, max_turns=args.max_turns)
    logger.info("Wrote run config (model=%s, max_turns=%d)", args.model, args.max_turns)
    agent.prepare_workspace()

    logger.info("Starting solve_harvey_task() agent loop (max_turns=%d)...", args.max_turns)
    started = time.perf_counter()
    summary = asyncio.run(agent.solve_harvey_task())
    logger.info(
        "solve_harvey_task() finished in %.2fs | %s",
        time.perf_counter() - started,
        agent._context_stats_line(),
    )

    transcript_path = agent.results_dir() / "transcript.txt"
    transcript_path.write_text(summary, encoding="utf-8")
    logger.info("Wrote transcript to %s", transcript_path)

    logger.info("Running evaluation...")
    agent.run_eval_command(judge_model=args.judge_model, dual=args.dual)
    logger.info("Generating report...")
    agent.run_report_command()

    score_summary = agent.load_score_summary(dual=args.dual)
    logger.info("Loaded score summary")

    print(f"Detected run ID: {run_id}")

    if score_summary:
        print("\nScore summary:")
        print(f"  score_file: {score_summary.get('score_file')}")
        print(f"  all_pass:   {score_summary.get('all_pass')}")
        print(f"  score:      {score_summary.get('score')}")
        print(f"  pass_rate:  {score_summary.get('pass_rate')}")
        print(
            "  criteria:   "
            f"{score_summary.get('passed')}/{score_summary.get('total')} passed, "
            f"{score_summary.get('failed')} failed"
        )
    else:
        print("\nScore summary unavailable: expected score artifact was not found.")

    if args.summary:
        run_summary = asyncio.run(agent.summarize_run())
        print("\nRun summary:")
        print(run_summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple CLI wrapper for running the Harvey Labs benchmark tutorial flow."
    )
    parser.add_argument(
        "--harvey-path",
        help="Path to a local clone of harvey-labs. Defaults to HARVEY_LABS_PATH or ../harvey-labs.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging (shows tool call arguments/results, timings, etc.).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", help="Describe a benchmark task.")
    describe.add_argument("--task", default=DEFAULT_TASK)
    describe.set_defaults(func=describe_task)

    run = subparsers.add_parser("run", help="Run a benchmark task, with optional eval/report.")
    run.add_argument("--task", default=DEFAULT_TASK)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--max-turns", type=int, default=200)
    run.add_argument("--run-id", help="Optional explicit run id.")
    run.add_argument("--eval", action="store_true", help="Run evaluation after harness run.")
    run.add_argument("--report", action="store_true", help="Generate report.html after run/eval.")
    run.add_argument("--judge-model", help="Optional judge model override for evaluation.")
    run.add_argument("--dual", action="store_true", help="Enable dual-judge evaluation mode.")
    run.add_argument(
        "--summary",
        action="store_true",
        help="Ask the OO agent to summarize the run after execution.",
    )
    run.add_argument(
        "--prompts",
        help="Path to a GEPA-optimized prompt JSON (see optimize_prompt.py) with a system_prompt override.",
    )
    run.set_defaults(func=run_task)

    solve = subparsers.add_parser(
        "solve",
        help="Solve a benchmark task end-to-end (run + eval + report + score summary).",
    )
    solve.add_argument("--task", default=DEFAULT_TASK)
    solve.add_argument("--model", default=DEFAULT_MODEL)
    solve.add_argument("--max-turns", type=int, default=200)
    solve.add_argument("--run-id", help="Optional explicit run id.")
    solve.add_argument("--judge-model", help="Optional judge model override for evaluation.")
    solve.add_argument("--dual", action="store_true", help="Enable dual-judge evaluation mode.")
    solve.add_argument(
        "--summary",
        action="store_true",
        help="Ask the OO agent to summarize the solved run.",
    )
    solve.add_argument(
        "--prompts",
        help="Path to a GEPA-optimized prompt JSON (see optimize_prompt.py) with a system_prompt override.",
    )
    solve.set_defaults(func=solve_task)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
