"""Run only the critic on a saved run record and print every claim check.

Usage: python -m scripts.critic_debug runs/qwen_llm/triagerca/<case>.json
       [--evidence full|cited] [--no-comparisons] [--no-onset-check]
Fast loop for prompt work: no Planner/Executor calls, just one critic call (cached if unchanged).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.agents.critic.llm_critic import LLMCritic
from src.evidence import EvidenceLedger
from src.run import build_llm
from src.schemas import Hypothesis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_json", type=Path)
    parser.add_argument("--evidence", default="full", choices=["full", "cited"])
    parser.add_argument("--no-comparisons", action="store_true", help="leave out the cross-service metric table")
    parser.add_argument("--no-onset-check", action="store_true", help="turn off the code check on the onset shift ranking")
    parser.add_argument("--critic-model", default="open_weight_model")
    args = parser.parse_args()

    record = json.loads(args.run_json.read_text())
    ledger = EvidenceLedger()
    for item in record["evidence_ledger"]:
        ledger.add(item["source"], item["args"], item["content"])  # same order -> same IDs
    hypothesis = Hypothesis(**record["report"])

    models = yaml.safe_load(Path("configs/models.yaml").read_text())
    critic = LLMCritic(
        llm=build_llm(models[args.critic_model], cache_dir=".cache/llm"),
        evidence_mode=args.evidence,
        comparisons=not args.no_comparisons,
        onset_check=not args.no_onset_check,
    )
    verdict = critic.review(hypothesis, ledger)

    for i, c in enumerate(critic.last_claims, 1):
        root = " ROOT" if c.is_root_cause else ""
        print(f"{i}. [{c.label}{root}] ids={c.evidence_ids}")
        print(f"   claim:      {c.claim}")
        print(f"   comparison: {c.comparison or '(none)'}")
        print(f"   reason:     {c.reason}")
    print(f"\nverdict: {verdict.decision}")
    for m in verdict.missing_evidence:
        print(f"  check: {m}")
    for call in critic.calls:
        tokens = f"{call.input_tokens} in / {call.output_tokens} out"
        print(f"critic call: {tokens}, {call.latency_s:.1f}s, cached={call.cached}")


if __name__ == "__main__":
    main()
