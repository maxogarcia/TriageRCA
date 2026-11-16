#!/usr/bin/env python3
"""Human vs. judge agreement on the grounding calibration set (Grounding Metric Spec v1.1, 7.1).

Joins the human-labeled CSV (from the review page) with calibration_key.csv on (row, case_id, claim_id)
and reports, for the judge's verified labels (primary) and raw labels:
  - Cohen's kappa over the 3 labels (Supported / Partially Supported / Unsupported); target >= 0.7
  - percent agreement, label counts per side, 3x3 confusion matrix
Plus agreement by claim type, and writes every claim (with both labels) to calibration_agreement.csv.

Read-only on its inputs. Run it only after the human labels are final for this pass.

Note: kappa is unstable when one label dominates (e.g. mostly Supported) and with few claims;
read it together with percent agreement and the confusion matrix.

Example:
  python scripts/calibration_agreement.py --labeled results/grounding_dev/calibration_labeled.csv \
      --key results/grounding_dev/calibration_key.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["Supported", "Partially Supported", "Unsupported"]
SHORT = {"Supported": "S", "Partially Supported": "P", "Unsupported": "U"}
KEY = ("row", "case_id", "claim_id")


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in LABELS) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def fmt(k: float | None) -> str:
    return "n/a (all one label)" if k is None else f"{k:.3f}"


def report(name: str, pairs: list[tuple[str, str]]):
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    ch, cj = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    print(f"\n== {name}: kappa={fmt(kappa(pairs))}  agreement={agree}/{n} ({agree / n:.0%})")
    print("   counts  human: " + " ".join(f"{SHORT[l]}={ch[l]}" for l in LABELS)
          + "   judge: " + " ".join(f"{SHORT[l]}={cj[l]}" for l in LABELS))
    print("   confusion (rows=human, cols=judge)   S  P  U")
    cm = Counter(pairs)
    for h in LABELS:
        print(f"   {h:<35}" + "".join(f"{cm[(h, j)]:>3}" for j in LABELS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeled", required=True, help="human-labeled CSV from the review page")
    ap.add_argument("--key", required=True, help="calibration_key.csv from judge_grounding.py")
    ap.add_argument("--out", help="per-claim CSV (default: calibration_agreement.csv next to --labeled)")
    args = ap.parse_args()

    human = read(Path(args.labeled))
    key = {tuple(r[k] for k in KEY): r for r in read(Path(args.key))}
    bad = [r for r in human if r.get("human_label") not in LABELS]
    if bad:
        sys.exit(f"{len(bad)} claims have no valid human_label, e.g. {[r['claim_id'] for r in bad[:3]]}")
    missing = [tuple(r[k] for k in KEY) for r in human if tuple(r[k] for k in KEY) not in key]
    if missing:
        sys.exit(f"{len(missing)} labeled claims not in key, e.g. {missing[:2]}")
    if len(key) != len(human):
        print(f"note: key has {len(key)} claims, labeled file has {len(human)}")

    rows = []
    for r in human:
        k = key[tuple(r[c] for c in KEY)]
        rows.append({**{c: r[c] for c in KEY}, "claim_type": r["claim_type"], "claim": r["claim"],
                     "human_label": r["human_label"], "human_evidence": r.get("human_evidence", ""),
                     "human_note": r.get("human_note", ""),
                     "judge_verified_label": k["judge_verified_label"], "judge_label": k["judge_label"],
                     "judge_match": k.get("judge_match", ""), "judge_quote_ok": k.get("judge_quote_ok", ""),
                     "judge_evidence": k.get("judge_evidence", ""),
                     "agree_verified": int(r["human_label"] == k["judge_verified_label"]),
                     "agree_raw": int(r["human_label"] == k["judge_label"])})

    print(f"{len(rows)} claims, {len({(r['row'], r['case_id']) for r in rows})} cases (target kappa >= 0.7)")
    report("verified (primary)", [(r["human_label"], r["judge_verified_label"]) for r in rows])
    report("raw", [(r["human_label"], r["judge_label"]) for r in rows])

    by_t = defaultdict(list)
    for r in rows:
        by_t[r["claim_type"]].append(r)
    print("\n== by claim type (verified)")
    for t, rs in sorted(by_t.items()):
        a = sum(r["agree_verified"] for r in rs)
        print(f"   {t:<14} agreement={a}/{len(rs)}  kappa={fmt(kappa([(r['human_label'], r['judge_verified_label']) for r in rs]))}")

    dis = [r for r in rows if not r["agree_verified"]]
    print(f"\n== disagreements (verified): {len(dis)}")
    for r in dis:
        print(f"   {r['case_id']} #{r['claim_id']} [{r['claim_type']}] human={SHORT[r['human_label']]} "
              f"judge={SHORT[r['judge_verified_label']]} (raw {SHORT.get(r['judge_label'], '?')}, "
              f"quote_ok={r['judge_quote_ok'] or '-'})")

    out = Path(args.out) if args.out else Path(args.labeled).with_name("calibration_agreement.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-claim table -> {out}")


if __name__ == "__main__":
    main()
