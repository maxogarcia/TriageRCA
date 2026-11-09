"""CLI entry point: run a config over a directory of RCAEval cases, one JSON record per case
written to runs/<config_name>/<case_id>.json (hard rule #5).

Usage:
    python -m src.run --config configs/dev.yaml
    python -m src.run --config configs/dev.yaml --config-name react --cases-limit 5
    python -m src.run --config configs/dev.yaml --model open_weight_model --runs-dir runs/qwen   # local Qwen
    python -m src.run --config configs/dev.yaml --cases "re2ob_*_cpu_1" "re2ob_*_loss_1" --cases-limit 10
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.agents.critic import AlwaysAcceptCritic, Critic
from src.agents.critic.llm_critic import LLMCritic
from src.data.rcaeval_adapter import load_case
from src.data.labels import DEFAULT_MANIFEST, ManifestLabels
from src.llm import gemini, openai_compat
from src.pipeline import Pricing, run_case
from src.rag.retriever import Retriever, StubRetriever
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


def build_retriever(name: str) -> Retriever:
    if name == "stub":
        return StubRetriever()
    if name == "chroma":
        # Imported lazily: loads sentence-transformers + ChromaDB, and needs the index built
        # first (python -m src.rag.embed, from data/runbooks/).
        from src.rag.chroma_retriever import ChromaRetriever

        return ChromaRetriever()
    raise ValueError(f"Unknown retriever: {name!r} (expected 'stub' or 'chroma')")


def build_critic(cfg: dict, models: dict) -> Critic:
    name = cfg.get("critic", "stub")
    if name == "stub":
        return AlwaysAcceptCritic()
    if name == "llm":
        # Fixed critic model, independent of --model: the same critic judges every pipeline model.
        critic_llm = build_llm(models[cfg.get("critic_model", "open_weight_model")], cache_dir=cfg["cache_dir"])
        return LLMCritic(
            llm=critic_llm,
            evidence_mode=cfg.get("critic_evidence", "full"),
            comparisons=cfg.get("critic_comparisons", True),
        )
    raise ValueError(f"Unknown critic: {name!r} (expected 'stub' or 'llm')")


def build_tools(case, retriever: Retriever) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(METRIC_QUERY_SPEC, make_metric_query(case))
    registry.register(LOG_SEARCH_SPEC, make_log_search(case))
    registry.register(TRACE_QUERY_SPEC, make_trace_query(case))
    registry.register(RUNBOOK_RETRIEVAL_SPEC, make_runbook_retrieval(retriever))
    return registry


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="manifest defining eligible cases and their ground-truth labels")
    parser.add_argument("--config-name", help="override config_name (retrieval_only, react, triagerca, ...)")
    parser.add_argument("--cases-limit", type=int, help="override cases_limit")
    parser.add_argument("--model", help="override model, a key in configs/models.yaml (e.g. open_weight_model)")
    parser.add_argument("--cases", nargs="+", help="case IDs or globs to run, e.g. re2ob_emailservice_*_1")
    parser.add_argument("--critic", help="override critic: stub (always accepts) or llm")
    parser.add_argument("--runs-dir", help="override runs_dir, e.g. runs/qwen so runs don't overwrite another model's")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.config_name:
        cfg["config_name"] = args.config_name
    if args.model:
        cfg["model"] = args.model
    if args.runs_dir:
        cfg["runs_dir"] = args.runs_dir
    if args.critic:
        cfg["critic"] = args.critic
    if args.cases_limit is not None:
        cfg["cases_limit"] = args.cases_limit
    models = yaml.safe_load(Path("configs/models.yaml").read_text())
    model_cfg = models[cfg["model"]]

    labels = ManifestLabels(args.manifest)
    llm = build_llm(model_cfg, cache_dir=cfg["cache_dir"])
    pricing = Pricing(
        input_per_1m=model_cfg.get("price_per_1m_input_tokens", 0.0),
        output_per_1m=model_cfg.get("price_per_1m_output_tokens", 0.0),
    )
    retriever = build_retriever(cfg.get("retriever", "stub"))
    critic = build_critic(cfg, models)

    data_dir = Path(cfg["data_dir"])
    case_dirs = sorted(p for p in data_dir.iterdir() if (p / "inject_time.txt").is_file())
    # The manifest defines experiment membership, not just the available labels.
    # Filter before --cases and cases_limit so unrelated downloads cannot be selected.
    case_dirs = [p for p in case_dirs if p.name in labels.rows.index]
    if args.cases:
        case_dirs = [p for p in case_dirs if any(fnmatch(p.name, pattern) for pattern in args.cases)]
    case_dirs = case_dirs[: cfg["cases_limit"]]
    if not case_dirs:
        raise SystemExit(
            f"No cases matched the manifest, --cases filter, and cases_limit under {data_dir}. "
            "Check data_dir and your selection settings. Download the selected manifest with: "
            f"python scripts/download_data.py --manifest {str(args.manifest)!r} "
            f"--out {str(data_dir)!r}"
        )

    out_dir = Path(cfg["runs_dir"]) / cfg["config_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for case_dir in case_dirs:
        case, _label = load_case(case_dir, label_provider=labels)
        tools = build_tools(case, retriever)
        record = run_case(
            case,
            cfg["config_name"],
            llm,
            tools,
            critic=critic,
            max_tool_calls=cfg["max_tool_calls"],
            max_iterations=cfg.get("max_iterations", 3),
            pricing=pricing,
        )
        out_path = out_dir / f"{case.case_id}.json"
        out_path.write_text(record.model_dump_json(indent=2))
        status = "ok" if record.report else "FAILED"
        print(f"{case.case_id}: {status} (${record.cost_usd:.4f}, {len(record.llm_calls)} calls) -> {out_path}")
        for err in record.errors:
            print(f"  error: {err}")


if __name__ == "__main__":
    main()
