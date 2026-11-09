"""Run only the critic over many saved run records and summarize the verdicts.

Usage:
    python -m scripts.critic_batch runs/qwen_stub/triagerca --cases <id or glob> ... \
        [--critic-model open_weight_model] [--evidence full|cited] [--no-comparisons] [--no-onset-check] [--tag NAME]

Reviews each saved first hypothesis with one critic setting and prints one line per case. The
true root cause and fault are read from the case ID (re2ob_<service>_<fault>_<n>) for this
summary only; the critic never sees them. Full claim checks are saved to
runs/critic_eval/<tag>.json so settings can be compared later.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path

import yaml

from src.agents.critic.llm_critic import LLMCritic
from src.evidence import EvidenceLedger
from src.run import build_llm
from src.schemas import Hypothesis


def truth_from_case_id(case_id: str) -> tuple[str, str]:
    """re2ob_checkoutservice_cpu_1 -> ("checkoutservice", "cpu")."""
    parts = case_id.split("_")
    return parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path, help="folder of saved run JSONs, e.g. runs/qwen_stub/triagerca")
    parser.add_argument("--cases", nargs="+", default=["*"], help="case IDs or globs")
    parser.add_argument("--critic-model", default="open_weight_model")
    parser.add_argument("--evidence", default="full", choices=["full", "cited"])
    parser.add_argument("--no-comparisons", action="store_true")
    parser.add_argument("--no-onset-check", action="store_true")
    parser.add_argument("--tag", help="name for the saved results (default: built from the settings)")
    args = parser.parse_args()

    comparisons = not args.no_comparisons
    onset_check = not args.no_onset_check
    tag = args.tag or f"{args.critic_model}_{args.evidence}_{'table' if comparisons else 'notable'}{'_onset' if onset_check else ''}"
    models = yaml.safe_load(Path("configs/models.yaml").read_text())
    critic_llm = build_llm(models[args.critic_model], cache_dir=".cache/llm")

    paths = sorted(p for p in args.runs_dir.glob("*.json") if any(fnmatch(p.stem, c) for c in args.cases))
    if not paths:
        raise SystemExit(f"No run JSONs in {args.runs_dir} match {args.cases}")

    print(f"setting: {tag}  ({len(paths)} cases)\n")
    print(f"{'case':40} {'hyp top':24} {'top ok':6} {'indicator':32} {'root':12} {'ind.':12} verdict")
    results, decisions = [], Counter()
    start = time.monotonic()
    for path in paths:
        record = json.loads(path.read_text())
        if not record.get("report"):
            print(f"{record['case_id']:40} (no report, skipped)")
            continue
        ledger = EvidenceLedger()
        for item in record["evidence_ledger"]:
            ledger.add(item["source"], item["args"], item["content"])
        hypothesis = Hypothesis(**record["report"])

        critic = LLMCritic(llm=critic_llm, evidence_mode=args.evidence, comparisons=comparisons, onset_check=onset_check)
        verdict = critic.review(hypothesis, ledger)
        decisions[verdict.decision] += 1

        true_svc, true_fault = truth_from_case_id(record["case_id"])
        top = hypothesis.ranked_services[0] if hypothesis.ranked_services else ""
        top_ok = top == true_svc
        claims = critic.last_claims
        root_label = claims[0].label if claims else "-"
        ind_label = claims[1].label if len(claims) > 1 else "-"
        print(
            f"{record['case_id']:40} {top:24} {'yes' if top_ok else 'NO':6} {hypothesis.indicator[:32]:32} "
            f"{root_label:12} {ind_label:12} {verdict.decision}"
        )
        results.append(
            {
                "case_id": record["case_id"],
                "true_service": true_svc,
                "true_fault": true_fault,
                "hyp_top": top,
                "hyp_indicator": hypothesis.indicator,
                "top_ok": top_ok,
                "verdict": verdict.model_dump(),
                "claims": [c.model_dump() for c in claims],
                "critic_calls": [
                    {"input_tokens": c.input_tokens, "output_tokens": c.output_tokens, "latency_s": c.latency_s}
                    for c in critic.calls
                ],
            }
        )

    # Did the critic accept correct hypotheses and push back on wrong ones?
    ok = [r for r in results if r["top_ok"]]
    wrong = [r for r in results if not r["top_ok"]]
    accepted_ok = sum(r["verdict"]["decision"] == "accept" for r in ok)
    pushed_wrong = sum(r["verdict"]["decision"] != "accept" for r in wrong)
    elapsed = time.monotonic() - start
    print(f"\nverdicts: {dict(decisions)}   total {elapsed:.0f}s")
    print(f"correct top service accepted:      {accepted_ok}/{len(ok)}")
    print(f"wrong top service pushed back on:  {pushed_wrong}/{len(wrong)}")

    out = Path("runs/critic_eval") / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"setting": tag, "results": results}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
