#!/usr/bin/env python3
"""Grounding judge for TriageRCA run outputs (Grounding Metric Spec v1.1, sections 2, 3, 5, 6, 7).

Standalone: talks to Ollama over HTTP and reads run JSONs + the manifest + case parquet headers.
It does not import src/ except in --check-query mode, which verifies the rebuilt incident query
against the pipeline's own code once.

Per report:
  Step A (decompose): the explanation becomes atomic claims typed root-cause / observational / causal.
                      The structured fields (ranked_services[0], fault_category, indicator) become
                      fixed root-cause claims without an LLM call.
  Step B (label):     ONE call per claim: case input (rebuilt incident query) + full evidence ledger
                      + the claim. The model fills claim_says -> evidence_says -> match (yes/partly/no/
                      absent); code maps match to Supported / Partially Supported / Unsupported.
  The judge never sees ground truth, critic verdicts, hypotheses, or the report's citations.

Grounding rate = (S + 0.5 * P) / claims. PRIMARY (grounding_expl) uses the explanation claims only;
the 3 fixed root-cause claims are still judged and kept in the full score (grounding), which has a
known S->P bias on them (calibration, day 7.1). Parse failures are counted, not dropped.
Primary score uses verified labels: a yes/partly whose copied evidence quote is not found in the
cited items counts as Unsupported (quote_ok=0). grounding_raw keeps the judge's labels as given.

Examples:
  python scripts/judge_grounding.py --out runs/ablation --check-query
  python scripts/judge_grounding.py --out runs/ablation --rows hybrid --limit 3
  python scripts/judge_grounding.py --out runs/ablation runs/baselines --export-calibration
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "data/manifests/sample_manifest.csv"
# Copy of src/data/labels.py SYSTEMS (verified by --check-query).
SYSTEMS = {"ob": "online_boutique", "ss": "sock_shop", "tt": "train_ticket"}
SKIP_DIRS = {"llm_cache", "grounding", "judge_cache"}

LABELS = ["Supported", "Partially Supported", "Unsupported"]
WEIGHT = {"Supported": 1.0, "Partially Supported": 0.5, "Unsupported": 0.0}
CLAIM_TYPES = ["root-cause", "causal", "observational"]
CITE_RE = re.compile(r"\s*[\(\[]\s*E\d+(?:\s*[,;/]\s*E\d+)*\s*[\)\]]")

PROMPT_VERSION = "v3-per-claim"  # bump on any prompt/schema change; freeze before held-out
SCORING_VERSION = "expl-primary-v1"  # primary = explanation claims only; full score kept as grounding
NUM_PREDICT = 1024
CHARS_PER_TOKEN = 3.2  # conservative estimate for log/JSON-heavy text

# ---------------------------------------------------------------- prompts

DECOMPOSE_SYSTEM = """You split the explanation part of an incident diagnosis into atomic claims for an evaluation.

An atomic claim is ONE factual assertion about this specific incident, rewritten as a standalone sentence (name the service, metric, or resource instead of using pronouns). Extract a claim when the text:
- attributes a behavior or value to a specific service, metric, log, trace, or resource (e.g. "the frontend 5xx error rate rose to 12%");
- asserts a causal or temporal relationship (e.g. "the cartservice failure preceded the checkout timeouts");
- states a root cause or contributing factor;
- references a threshold, baseline, or normal level.

Do NOT extract:
- hedged or speculative statements ("it is possible that...", "may be");
- restatements of the incident report or task;
- general knowledge not specific to this incident ("microservices can fail due to...");
- recommendations or next steps;
- any claim equivalent to one in the ALREADY EXTRACTED list.

Copy numbers, names and units exactly as written. Do not add facts that are not in the text. Ignore evidence ID markers such as (E001).

Give each claim a type:
- root-cause: says what the root cause or a contributing factor is;
- causal: says one thing caused, led to, or preceded another;
- observational: reports an observed behavior or value, or compares one to a baseline.

Return JSON: {"claims": [{"text": "...", "type": "root-cause|causal|observational"}]}. Return {"claims": []} if there are none."""

DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {
        "type": "object",
        "properties": {"text": {"type": "string"}, "type": {"type": "string", "enum": CLAIM_TYPES}},
        "required": ["text", "type"]}}},
    "required": ["claims"],
}

LABEL_SYSTEM = """You check ONE claim from an incident diagnosis against the evidence the diagnosing system actually saw. The evidence is a closed set: use only it. Do not use outside knowledge to fill gaps, and do not treat a claim as true because it sounds plausible.

Work in this order:
1. claim_says: restate exactly what the claim asserts (service, metric, value, time, or relationship).
2. evidence_says: copy the most relevant evidence line(s) word for word, or write "absent" if nothing in the evidence is about this.
3. evidence_ids: the IDs of the items you copied from (such as "E003", or "INPUT" for the case input).
4. match: compare claim_says with evidence_says:
   - "yes": the evidence directly states what the claim asserts; the claim follows from the evidence alone.
   - "partly": the evidence has related information, but the claim needs inference, extrapolation, or domain knowledge not in the evidence.
   - "no": the evidence contradicts the claim (a different number, a different service, a different metric, or the opposite direction).
   - "absent": nothing in the evidence is about what the claim asserts.

Rules:
- Compare numbers, service names, metric names and times exactly. A value counts only for the service it is listed under.
- For a claim that names the root-cause service, the fault category, or the anomaly indicator: "yes" if the evidence clearly singles out what the claim names (for example, its anomaly is the largest by a wide margin, starts first, or error logs name it); "no" if the evidence clearly singles out something else, or the named metric does not appear; "partly" only if the evidence is mixed.
- A generic caveat in a tool output (such as "not causal proof") does not by itself lower the match.

Return JSON with keys claim_says, evidence_says, evidence_ids, match."""

MATCHES = ["yes", "partly", "no", "absent"]
MATCH_TO_LABEL = {"yes": "Supported", "partly": "Partially Supported", "no": "Unsupported", "absent": "Unsupported"}
LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_says": {"type": "string"},
        "evidence_says": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "match": {"type": "string", "enum": MATCHES},
    },
    "required": ["claim_says", "evidence_says", "evidence_ids", "match"],
}

# ---------------------------------------------------------------- inputs


def repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def discover(out_dirs: list[str], rows: set[str] | None) -> list[tuple[str, Path]]:
    """Run JSONs at <out>/<row>/<sub>/<case>.json; row = first directory under <out>."""
    found = []
    for out in out_dirs:
        base = Path(out)
        for p in sorted(base.glob("*/*/*.json")):
            rel = p.relative_to(base)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            row = rel.parts[0]
            if rows and row not in rows:
                continue
            found.append((row, p))
    return found


def load_manifest(path: Path) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {r["case"]: r for r in csv.DictReader(f)}


_query_cache: dict[str, str] = {}


def incident_query(m: dict) -> str:
    """Rebuild src/data/rcaeval_adapter.py::_template_incident_query for one manifest row."""
    cid = m["case"]
    if cid not in _query_cache:
        import pandas as pd

        case_dir = repo_path(m["case_dir"])
        cols = pd.read_parquet(case_dir / "metrics.parquet").columns
        services = sorted({c.rsplit("_", 1)[0] for c in cols if c != "time"})
        inject_time = float((case_dir / "inject_time.txt").read_text().strip())
        system = SYSTEMS[m["system"]]
        _query_cache[cid] = (
            f"Users are reporting degraded behavior in the {system.replace('_', ' ')} system starting around "
            f"unix time {inject_time:.0f}. Services involved: {', '.join(services)}. "
            f"Please investigate and identify the likely root cause."
        )
    return _query_cache[cid]


def structured_claims(report: dict) -> list[str]:
    out = []
    rs = report.get("ranked_services") or []
    if rs:
        out.append(f"{rs[0]} is the root-cause service of this incident.")
    if report.get("fault_category"):
        out.append(f"The fault category is {report['fault_category']}.")
    if report.get("indicator"):
        out.append(f"The anomaly indicator is {report['indicator']}.")
    return out


def format_evidence(query: str, ledger: list[dict]) -> str:
    parts = [f"### INPUT | case input (incident description)\n{query}"]
    for item in ledger:
        args = json.dumps(item.get("args"), sort_keys=True, default=str)
        parts.append(f"### {item.get('id')} | source: {item.get('source')} | args: {args}\n{item.get('content')}")
    return "\n\n".join(parts)


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def quote_ok(evidence_says: str, cited: list[str], texts: dict[str, str]) -> int | str:
    """1 if every copied fragment (>= 12 chars) appears in the cited items (all items if none cited)."""
    if _norm(evidence_says) in ("", "absent"):
        return ""
    pool = [texts[i] for i in cited if i in texts] or list(texts.values())
    hay = _norm(" ".join(pool))
    frags = [f for f in (_norm(x).strip(" .;-") for x in re.split(r"\n|\.\.\.|…", evidence_says)) if len(f) >= 12]
    return int(all(f in hay for f in frags)) if frags else ""


# ---------------------------------------------------------------- ollama


class OllamaError(RuntimeError):
    pass


def chat(args, system: str, user: str, schema: dict) -> tuple[dict, bool]:
    body = {
        "model": args.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "format": schema,
        "options": {"temperature": 0, "seed": 0, "num_ctx": args.num_ctx, "num_predict": NUM_PREDICT},
    }
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    cache_file = args.cache_dir / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text()), True
    req = urllib.request.Request(
        args.ollama_url.rstrip("/") + "/api/chat",
        data=json.dumps({**body, "keep_alive": args.keep_alive}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            resp = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise OllamaError(str(e)) from e
    rec = {
        "content": resp.get("message", {}).get("content", ""),
        "prompt_eval_count": resp.get("prompt_eval_count"),
        "eval_count": resp.get("eval_count"),
        "seconds": round(resp.get("total_duration", 0) / 1e9, 2),
    }
    cache_file.write_text(json.dumps(rec))
    return rec, False


def parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def truncated(rec: dict, num_ctx: int) -> bool:
    n = rec.get("prompt_eval_count") or 0
    return n >= num_ctx - NUM_PREDICT - 16


# ---------------------------------------------------------------- judging


def judge_case(args, row: str, path: Path, run: dict, m: dict) -> tuple[dict, list[dict]]:
    case_id = run.get("case_id", path.stem)
    base = {"row": row, "case_id": case_id, "system": m.get("system", ""), "fault": m.get("fault", ""),
            "run_path": str(path)}
    case = {**base, "prompt_version": PROMPT_VERSION, "status": "ok", "n_parse_fail": 0, "n_claims": 0, "n_S": 0, "n_P": 0, "n_U": 0, "grounding": "", "grounding_raw": "",
            "n_claims_expl": 0, "grounding_expl": "", "grounding_expl_raw": "", "scoring_version": SCORING_VERSION, "n_evidence": 0, "bad_cites": 0, "bad_quotes": 0, "prompt_tokens_B": "", "judge_s": 0.0, "cached": True}
    report = run.get("report")
    if not report:
        case["status"] = "no_report"
        return case, []
    ledger = run.get("evidence_ledger") or []
    case["n_evidence"] = len(ledger)

    # Step A
    fixed = structured_claims(report)
    claims = [{"text": t, "type": "root-cause", "origin": "structured"} for t in fixed]
    explanation = CITE_RE.sub("", report.get("explanation") or "").strip()
    if explanation:
        user_a = "ALREADY EXTRACTED:\n" + "\n".join(f"- {t}" for t in fixed) + f"\n\nEXPLANATION:\n{explanation}"
        rec, cached = chat(args, DECOMPOSE_SYSTEM, user_a, DECOMPOSE_SCHEMA)
        case["judge_s"] += rec["seconds"] if not cached else 0
        case["cached"] &= cached
        parsed = parse_json(rec["content"])
        if not isinstance(parsed, dict) or not isinstance(parsed.get("claims"), list):
            case["status"] = "parse_fail_decompose"
            return case, []
        for c in parsed["claims"]:
            if isinstance(c, dict) and str(c.get("text", "")).strip():
                ctype = c.get("type") if c.get("type") in CLAIM_TYPES else "observational"
                claims.append({"text": c["text"].strip(), "type": ctype, "origin": "explanation"})
    if not claims:
        case["status"] = "no_claims"
        return case, []
    case["n_claims"] = len(claims)
    case["n_claims_expl"] = sum(c["origin"] == "explanation" for c in claims)

    # Step B: one call per claim; evidence first so Ollama can reuse the shared prompt prefix
    evidence = format_evidence(incident_query(m), ledger)
    prefix = f"EVIDENCE:\n\n{evidence}\n\nCLAIM TO CHECK:\n"
    est = (len(LABEL_SYSTEM) + len(prefix) + max(len(c["text"]) for c in claims)) / CHARS_PER_TOKEN
    case["prompt_tokens_B"] = f"~{int(est)}"
    if est + NUM_PREDICT > args.num_ctx:
        case["status"] = "too_long"
        return case, []
    valid_ids = {str(i.get("id")) for i in ledger} | {"INPUT"}
    texts = {str(i.get("id")): f"{json.dumps(i.get('args'), sort_keys=True, default=str)} {i.get('content')}"
             for i in ledger}
    texts["INPUT"] = incident_query(m)
    claim_rows, score, score_raw, failed = [], 0.0, 0.0, 0
    expl_score, expl_raw = 0.0, 0.0
    for i, c in enumerate(claims, 1):
        rec, cached = chat(args, LABEL_SYSTEM, prefix + c["text"], LABEL_SCHEMA)
        case["judge_s"] += rec["seconds"] if not cached else 0
        case["cached"] &= cached
        if truncated(rec, args.num_ctx):
            case["status"] = "truncated"
            return case, []
        p = parse_json(rec["content"])
        row = {**base, "prompt_version": PROMPT_VERSION, "claim_id": i, "claim_type": c["type"],
               "origin": c["origin"], "claim": c["text"]}
        if not isinstance(p, dict) or p.get("match") not in MATCHES:
            failed += 1
            claim_rows.append({**row, "match": "PARSE_FAIL", "label": "PARSE_FAIL", "evidence": "",
                               "verified_label": "PARSE_FAIL", "bad_cite": 0, "quote_ok": "", "claim_says": "", "evidence_says": rec["content"][:300]})
            continue
        label = MATCH_TO_LABEL[p["match"]]
        ev = [str(e) for e in (p.get("evidence_ids") or [])]
        bad = p["match"] in ("yes", "partly") and (not ev or any(e not in valid_ids for e in ev))
        case["bad_cites"] += bad
        q = quote_ok(p.get("evidence_says", ""), ev, texts)
        case["bad_quotes"] += q == 0
        # Verified label (primary): support must rest on a quote that exists in the cited evidence.
        vlabel = "Unsupported" if q == 0 else label
        case[{"Supported": "n_S", "Partially Supported": "n_P", "Unsupported": "n_U"}[vlabel]] += 1
        score += WEIGHT[vlabel]
        score_raw += WEIGHT[label]
        if c["origin"] == "explanation":
            expl_score += WEIGHT[vlabel]
            expl_raw += WEIGHT[label]
        claim_rows.append({**row, "match": p["match"], "label": label, "verified_label": vlabel,
                           "evidence": " ".join(ev),
                           "bad_cite": int(bad), "quote_ok": q, "claim_says": p.get("claim_says", ""),
                           "evidence_says": p.get("evidence_says", "")})
    if failed:
        case["status"] = f"parse_fail_label"
        case["n_parse_fail"] = failed
        return case, claim_rows
    case["grounding"] = round(score / len(claims), 4)
    case["grounding_raw"] = round(score_raw / len(claims), 4)
    if case["n_claims_expl"]:
        case["grounding_expl"] = round(expl_score / case["n_claims_expl"], 4)
        case["grounding_expl_raw"] = round(expl_raw / case["n_claims_expl"], 4)
    return case, claim_rows


# ---------------------------------------------------------------- outputs


def stats(xs: list[float]) -> dict:
    if not xs:
        return {"mean": "", "median": "", "std": ""}
    return {"mean": round(statistics.mean(xs), 4), "median": round(statistics.median(xs), 4),
            "std": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(cases: list[dict], claims: list[dict]) -> tuple[list[dict], list[dict]]:
    by = defaultdict(list)
    for c in cases:
        by[(c["row"], c["system"])].append(c)
        by[(c["row"], "ALL")].append(c)
    summary = []
    for (row, system), cs in sorted(by.items()):
        scored = [c["grounding"] for c in cs if c["status"] == "ok"]
        raw = [c["grounding_raw"] for c in cs if c["status"] == "ok"]
        ex = [c["grounding_expl"] for c in cs if c["status"] == "ok" and c["grounding_expl"] != ""]
        ex_raw = [c["grounding_expl_raw"] for c in cs if c["status"] == "ok" and c["grounding_expl_raw"] != ""]
        n_cl = sum(c["n_claims"] for c in cs if c["status"] == "ok")
        fails = Counter(c["status"] for c in cs if c["status"] != "ok")
        summary.append({"row": row, "system": system, "n_cases": len(cs),
                        "n_scored_expl": len(ex), **{f"{k}_expl": v for k, v in stats(ex).items()},
                        "mean_expl_raw": stats(ex_raw)["mean"], "n_scored": len(scored),
                        **stats(scored), "mean_raw": stats(raw)["mean"],
                        "bad_quote_rate": round(sum(c["bad_quotes"] for c in cs if c["status"] == "ok") / n_cl, 4)
                        if n_cl else "",
                        "failures": " ".join(f"{k}={v}" for k, v in sorted(fails.items())),
                        "bad_cites": sum(c["bad_cites"] for c in cs),
                        "bad_quotes": sum(c["bad_quotes"] for c in cs)})
    by_t = defaultdict(list)
    for cl in claims:
        if cl["label"] in WEIGHT:
            by_t[(cl["row"], cl["claim_type"])].append((cl["verified_label"], cl["label"]))
    types = []
    for (row, t), pairs in sorted(by_t.items()):
        n = Counter(v for v, _ in pairs)
        types.append({"row": row, "claim_type": t, "n_claims": len(pairs), "n_S": n["Supported"],
                      "n_P": n["Partially Supported"], "n_U": n["Unsupported"],
                      "claim_grounding": round(sum(WEIGHT[v] for v, _ in pairs) / len(pairs), 4),
                      "claim_grounding_raw": round(sum(WEIGHT[r] for _, r in pairs) / len(pairs), 4)})
    return summary, types


def export_calibration(save_dir: Path, claims: list[dict], n_cases: int = 10, per_case: int = 3, seed: int = 0):
    """Spec 7.1: 30 claims (10 cases x 3), spread over rows and labels. Blind file hides judge labels."""
    rng = random.Random(seed)
    by_case = defaultdict(list)
    for cl in claims:
        if cl["label"] in WEIGHT:
            by_case[(cl["row"], cl["case_id"])].append(cl)
    keys = sorted(by_case)
    rng.shuffle(keys)
    keys.sort(key=lambda k: len({c["label"] for c in by_case[k]}), reverse=True)  # label-diverse first
    picked, rows_seen = [], Counter()
    for k in keys:  # round-robin over rows
        if len(picked) >= n_cases:
            break
        if rows_seen[k[0]] <= min(rows_seen.values(), default=0) or len(rows_seen) < len({x[0] for x in keys}):
            picked.append(k)
            rows_seen[k[0]] += 1
    blind, key = [], []
    for k in picked:
        cl = by_case[k][:]
        rng.shuffle(cl)
        chosen, seen = [], set()
        for c in cl:  # one claim per label first, then fill
            if c["label"] not in seen:
                chosen.append(c)
                seen.add(c["label"])
        chosen += [c for c in cl if c not in chosen]
        for c in chosen[:per_case]:
            common = {"row": c["row"], "case_id": c["case_id"], "claim_id": c["claim_id"],
                      "claim_type": c["claim_type"], "claim": c["claim"], "run_path": c["run_path"]}
            blind.append({**common, "human_label": "", "human_evidence": ""})
            key.append({**common, "judge_label": c["label"], "judge_verified_label": c["verified_label"],
                        "judge_match": c["match"], "judge_quote_ok": c["quote_ok"],
                        "judge_evidence": c["evidence"]})
    write_csv(save_dir / "calibration_blind.csv", blind)
    write_csv(save_dir / "calibration_key.csv", key)
    return len(blind)


# ---------------------------------------------------------------- check-query


def check_query(case_ids: list[str], manifest: dict) -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from RCAEval.utility import read_metrics
    from src.data.labels import SYSTEMS as SRC_SYSTEMS
    from src.data.rcaeval_adapter import _template_incident_query

    bad = 0 if SRC_SYSTEMS == SYSTEMS else 1
    if bad:
        print(f"SYSTEMS differ: src={SRC_SYSTEMS} judge={SYSTEMS}")
    for cid in case_ids:
        m = manifest[cid]
        case_dir = repo_path(m["case_dir"])
        metrics = read_metrics(str(case_dir))
        services = sorted({c.rsplit("_", 1)[0] for c in metrics.columns if c != "time"})
        it = float((case_dir / "inject_time.txt").read_text().strip())
        ref = _template_incident_query(services, it, SRC_SYSTEMS[m["system"]])
        if incident_query(m) != ref:
            bad += 1
            print(f"MISMATCH {cid}\n  src:   {ref[:300]}\n  judge: {incident_query(m)[:300]}")
    print(f"check-query: {len(case_ids)} cases, {bad} mismatches")
    return bad


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", nargs="+", required=True, help="run output dir(s); reads <out>/<row>/*/*.json")
    ap.add_argument("--rows", nargs="*", help="only these rows (first dir under --out)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="repo-relative or absolute")
    ap.add_argument("--model", default="gemma3-12b-16k")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--keep-alive", default="30m", help="keeps the judge model loaded between calls")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--save-dir", default="results/grounding", help="CSV output dir (relative to cwd)")
    ap.add_argument("--limit", type=int, help="judge only the first N runs (smoke test)")
    ap.add_argument("--check-query", action="store_true", help="verify rebuilt incident query vs src, then exit")
    ap.add_argument("--export-calibration", action="store_true", help="also write the 30-claim calibration CSVs")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    args.cache_dir = save_dir / "judge_cache"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(repo_path(args.manifest))

    runs = discover(args.out, set(args.rows) if args.rows else None)
    loaded = []
    for row, p in runs:
        try:
            run = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(run, dict) and "report" in run and "evidence_ledger" in run:
            loaded.append((row, p, run))
    if args.limit:
        loaded = loaded[: args.limit]
    rows_found = Counter(r for r, _, _ in loaded)
    print(f"found {len(loaded)} runs: " + ", ".join(f"{r}={n}" for r, n in sorted(rows_found.items())))
    missing = sorted({run.get("case_id", p.stem) for _, p, run in loaded} - set(manifest))
    if missing:
        sys.exit(f"{len(missing)} case_ids not in manifest, e.g. {missing[:3]}")

    if args.check_query:
        ids = sorted({run.get("case_id", p.stem) for _, p, run in loaded})
        sys.exit(1 if check_query(ids, manifest) else 0)

    cases, claims, consecutive_err = [], [], 0
    t0 = time.time()
    for i, (row, p, run) in enumerate(loaded, 1):
        m = manifest[run.get("case_id", p.stem)]
        try:
            case, cl = judge_case(args, row, p, run, m)
            consecutive_err = 0
        except OllamaError as e:
            consecutive_err += 1
            case = {"row": row, "case_id": run.get("case_id", p.stem), "system": m.get("system", ""),
                    "fault": m.get("fault", ""), "run_path": str(p), "prompt_version": PROMPT_VERSION,
                    "status": "ollama_error", "n_parse_fail": 0, "n_claims": 0,
                    "n_S": 0, "n_P": 0, "n_U": 0, "grounding": "", "grounding_raw": "", "n_claims_expl": 0, "grounding_expl": "", "grounding_expl_raw": "", "scoring_version": SCORING_VERSION, "n_evidence": 0, "bad_cites": 0,
                    "bad_quotes": 0, "prompt_tokens_B": "", "judge_s": 0.0, "cached": False}
            cl = []
            print(f"  ollama error: {e}")
            if consecutive_err >= 3:
                sys.exit("3 Ollama errors in a row; is `ollama serve` up and the model created?")
        cases.append(case)
        claims += cl
        tag = "cached" if case["cached"] else f"{case['judge_s']:.0f}s"
        g = case["grounding"] if case["grounding"] != "" else "-"
        print(f"[{i}/{len(loaded)}] {row} {case['case_id']} {case['status']} expl={case['grounding_expl'] if case['grounding_expl'] != '' else '-'} full={g} "
              f"claims={case['n_claims']} ({tag})")

    summary, types = summarize(cases, claims)
    write_csv(save_dir / "grounding_cases.csv", cases)
    write_csv(save_dir / "grounding_claims.csv", claims)
    write_csv(save_dir / "grounding_summary.csv", summary)
    write_csv(save_dir / "grounding_by_claim_type.csv", types)
    print(f"\ndone in {time.time() - t0:.0f}s -> {save_dir}/")
    for s in summary:
        if s["system"] == "ALL":
            print(f"  {s['row']:<24} PRIMARY expl n={s['n_scored_expl']}/{s['n_cases']} mean={s['mean_expl']} "
                  f"median={s['median_expl']} std={s['std_expl']} raw={s['mean_expl_raw']}")
            print(f"  {'':<24} full    n={s['n_scored']}/{s['n_cases']} mean={s['mean']} "
                  f"median={s['median']} std={s['std']} raw={s['mean_raw']} "
                  f"bad_quote_rate={s['bad_quote_rate']} {s['failures']}")
    if args.export_calibration:
        n = export_calibration(save_dir, claims)
        print(f"calibration: {n} claims -> calibration_blind.csv (+ calibration_key.csv)")


if __name__ == "__main__":
    main()
