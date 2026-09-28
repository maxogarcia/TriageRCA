"""CLI entry point: run a config over a directory of RCAEval cases, one JSON record per case
written to runs/<config_name>/<case_id>.json (hard rule #5).

Usage:
    python -m src.run --config configs/dev.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.data.rcaeval_adapter import load_case
from src.llm import gemini, openai_compat
from src.pipeline import run_case
from src.rag.retriever import StubRetriever
from src.tools.log_search import SPEC as LOG_SEARCH_SPEC
from src.tools.log_search import make_log_search
from src.tools.metric_query import SPEC as METRIC_QUERY_SPEC
from src.tools.metric_query import make_metric_query
from src.tools.registry import ToolRegistry
from src.tools.runbook_retrieval import SPEC as RUNBOOK_RETRIEVAL_SPEC
from src.tools.runbook_retrieval import make_runbook_retrieval
from src.tools.trace_query import SPEC as TRACE_QUERY_SPEC
from src.tools.trace_query import make_trace_query


def build_llm(model_cfg: dict, cache_dir: str):
    if model_cfg["provider"] == "gemini":
        return gemini.build(
            model=model_cfg["name"],
            cache_dir=cache_dir,
            min_interval_s=model_cfg.get("min_interval_s", 0.0),
        )
    if model_cfg["provider"] == "openai_compat":
        return openai_compat.build(
            model=model_cfg["name"],
            base_url=model_cfg["base_url"],
            cache_dir=cache_dir,
            min_interval_s=model_cfg.get("min_interval_s", 0.0),
        )
    raise ValueError(f"Unknown provider: {model_cfg['provider']!r}")


def build_tools(case) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(METRIC_QUERY_SPEC, make_metric_query(case))
    registry.register(LOG_SEARCH_SPEC, make_log_search(case))
    registry.register(TRACE_QUERY_SPEC, make_trace_query(case))
    registry.register(RUNBOOK_RETRIEVAL_SPEC, make_runbook_retrieval(StubRetriever()))
    return registry


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    models = yaml.safe_load(Path("configs/models.yaml").read_text())
    model_cfg = models[cfg["model"]]

    llm = build_llm(model_cfg, cache_dir=cfg["cache_dir"])

    data_dir = Path(cfg["data_dir"])
    case_dirs = sorted(p for p in data_dir.iterdir() if (p / "inject_time.txt").is_file())[: cfg["cases_limit"]]
    if not case_dirs:
        raise SystemExit(
            f"No case directories (containing inject_time.txt) directly under {data_dir}. "
            "Download them with: python scripts/download_data.py"
        )

    out_dir = Path(cfg["runs_dir"]) / cfg["config_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for case_dir in case_dirs:
        case, _label = load_case(case_dir)
        tools = build_tools(case)
        record = run_case(case, cfg["config_name"], llm, tools, max_tool_calls=cfg["max_tool_calls"])
        out_path = out_dir / f"{case.case_id}.json"
        out_path.write_text(record.model_dump_json(indent=2))
        print(f"{case.case_id}: {'ok' if record.report else 'FAILED'} -> {out_path}")


if __name__ == "__main__":
    main()
