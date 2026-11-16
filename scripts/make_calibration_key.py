#!/usr/bin/env python3
"""Build a calibration key for an already-labeled CSV from the judge's grounding_claims.csv.

judge_grounding.py --export-calibration samples its own claims, so it cannot produce a key for a
labeling batch made elsewhere (e.g. export_rootcause_blind.py). This picks the judge's labels for
exactly the claims in --labeled and writes them in calibration_key.csv format, for
calibration_agreement.py. Checks that the claim text matches, so a renumbering cannot slip through.

Read-only on its inputs.

  python scripts/make_calibration_key.py --claims results/grounding_onset/v3.2/grounding_claims.csv \\
      --labeled results/grounding_onset/calibration_labeled_rootcause.csv \\
      --out results/grounding_onset/v3.2/calibration_key_rootcause.csv
"""
import argparse
import csv
import sys
from pathlib import Path

KEY = ("row", "case_id", "claim_id")


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--claims", required=True, help="grounding_claims.csv from judge_grounding.py")
ap.add_argument("--labeled", required=True, help="human-labeled CSV (or the blind CSV)")
ap.add_argument("--out", required=True)
ap.add_argument("--skip-unjudged", action="store_true",
                help="drop labeled claims the judge has no row for (e.g. too_long cases) and also write the "
                     "matching labeled subset next to --out as <out stem>_labeled.csv")
args = ap.parse_args()

out = Path(args.out)
if out.resolve() in {Path(args.claims).resolve(), Path(args.labeled).resolve()}:
    sys.exit("--out would overwrite an input")
judge = {tuple(r[k] for k in KEY): r for r in read(args.claims)}
rows, problems, kept, skipped = [], [], [], []
labeled = read(args.labeled)
for h in labeled:
    k = tuple(h[c] for c in KEY)
    j = judge.get(k)
    if j is None:
        (skipped if args.skip_unjudged else problems).append(f"not judged: {k}")
        continue
    if j["label"] not in ("Supported", "Partially Supported", "Unsupported"):
        (skipped if args.skip_unjudged else problems).append(f"judge label {j['label']}: {k}")
        continue
    if j["claim"].strip() != h["claim"].strip():
        problems.append(f"claim text differs for {k}: judge={j['claim']!r} labeled={h['claim']!r}")
        continue
    rows.append({**{c: h[c] for c in KEY}, "claim_type": h["claim_type"], "claim": h["claim"],
                 "run_path": h.get("run_path", ""), "judge_label": j["label"],
                 "judge_verified_label": j["verified_label"], "judge_match": j["match"],
                 "judge_quote_ok": j["quote_ok"], "judge_evidence": j["evidence"],
                 "prompt_version": j.get("prompt_version", "")})
    kept.append(h)
if problems:
    sys.exit("\n".join(problems[:10]) + f"\n{len(problems)} problem(s); nothing written")
versions = {r["prompt_version"] for r in rows}
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
for s in skipped:
    print(f"skipped {s}")
if args.skip_unjudged:
    sub = out.with_name(out.stem + "_labeled.csv")
    with open(sub, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(labeled[0].keys()))
        w.writeheader()
        w.writerows(kept)
    print(f"labeled subset ({len(kept)} of {len(labeled)}) -> {sub}")
print(f"{len(rows)} claims, prompt_version={','.join(sorted(versions))} -> {out}")
