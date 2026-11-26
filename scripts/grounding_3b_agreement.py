#!/usr/bin/env python3
"""Phase 3b agreement (Addendum A13.3): judge vs blind human labels on 40 held-out explanation claims.

Read-only. No LLM calls. Two steps, run in this order:

  1) --receipt   checks every label file against blind_3b.csv and writes
                 <labels>/labels_3b_sha256.txt (sha256sum format). Does NOT read the map or the key.
                 Refuses to overwrite an existing receipt. Send the printed hashes for the fix plan.
  2) (default)   gate, then analysis, into a NEW (or empty) --out folder:
                 a) sha256 of DO_NOT_OPEN_map_3b.csv / DO_NOT_OPEN_key_3b.csv / blind / page / strata
                    vs the fix plan AND vs prep_3b_record.txt (bytes are hashed, never parsed or printed)
                 b) every label file vs the receipt; Aeon's file vs the hash recorded Nov 18
                 c) every label file: same 40 claim codes, case codes and claim text as blind_3b.csv,
                    one valid label per claim, no blanks
                 d) only then the map and key are parsed, checked, and the A13.3 numbers computed.

A13.3 numbers: judge (verified = primary, raw alongside) vs each labeler and vs the human majority:
Cohen kappa on 3 labels, Cohen kappa Supported vs not, raw agreement; human pairwise kappa and Fleiss
kappa; confusion matrices (human x judge verified / raw / raw match); false support (majority
Unsupported -> judge Supported) as a share of majority-Unsupported claims; over-credit (P -> S) and
under-credit (S -> not S); descriptive breakdowns by quote check, row, AC@1, system. CIs: 95% percentile
bootstrap resampling CASES (2 claims of a case stay together), B = 10,000, seed 0, one shared set of
resamples for every statistic. Resamples where a statistic is undefined (e.g. kappa with one class,
ratio with an empty denominator) are left out and counted (column n_undefined).

Pass rule (A13.3, fixed Nov 18): validated secondary metric only if (1) judge verified vs majority,
Supported vs not, kappa 95% CI lower bound > 0.4 AND (2) false support <= 20% with >= 8
majority-Unsupported claims. With fewer than 3 labelers there is no majority: the rule is not met.

Example (repo root):
  python scripts/grounding_3b_agreement.py --receipt
  python scripts/grounding_3b_agreement.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("ERROR: numpy is required (activate the project venv)")

# ---- fixed values from claude/fix-plan-critic-review.md, Phase 3b Results (recorded Nov 18) ----
EXPECTED = {
    "map": "ffcf22e711b42d9914ef28d3a978dcbd94b15d19560d174215e5912a0da5fdde",
    "key": "a5ab2bb3efaa4b557c70ca122a998fc810668f53a02702b0a5ed3d20ef01e4c5",
    "blind_prefix": "1d5465d4641d",
    "page_prefix": "4b232f215cd1",
    "strata_prefix": "8880e2df83c4",
    "prep_script_prefix": "93de05f63ff4",
    "labels_aeon": "7412c9b299d8f45a7bce011ff8e6fa21e9415be9b51503e28d61bc88f09b582f",
    "n_fallback": 1,
}
ALL_LABELERS = ["aeon", "daniel", "max"]  # A13.3
LABELS = ["Supported", "Partially Supported", "Unsupported"]
SHORT = {"Supported": "S", "Partially Supported": "P", "Unsupported": "U"}
ROWS = ["hybrid", "llm_only", "no_critic", "react", "retrieval_only"]
N_CASES, N_CLAIMS, CLAIMS_PER_CASE = 20, 40, 2
B, SEED = 10_000, 0
PASS_KAPPA_LO, PASS_FS_MAX, PASS_MIN_U = 0.4, 0.20, 8

F_MAP, F_KEY, F_BLIND = "DO_NOT_OPEN_map_3b.csv", "DO_NOT_OPEN_key_3b.csv", "blind_3b.csv"
F_PAGE, F_STRATA, F_RECORD = "labeling_page_3b.html", "strata_counts_3b.csv", "prep_3b_record.txt"
F_RECEIPT = "labels_3b_sha256.txt"
LABEL_COLS = ["labeler", "case_code", "claim_code", "claim", "human_label", "human_evidence", "human_note"]
MAP_COLS = ["case_code", "claim_code", "row", "case_id", "system", "ac1_correct", "fallback", "claim_id"]
KEY_COLS = ["claim_code", "judge_label", "judge_verified_label", "judge_match", "judge_quote_ok"]


def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path, need: list[str], what: str) -> list[dict]:
    if not path.exists():
        die(f"missing {what}: {path}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        cols, rows = r.fieldnames or [], list(r)
    miss = [c for c in need if c not in cols]
    if miss:
        die(f"{what} ({path.name}) is missing columns {miss}")
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0].keys()) if rows else ["empty"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ gate

def parse_record(path: Path) -> dict:
    if not path.exists():
        die(f"missing {path}")
    rec = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^sha256 ([0-9a-f]{64})\s+(\S+)$", line.strip())
        if m:
            rec[m.group(2)] = m.group(1)
        m = re.match(r"^script sha256 ([0-9a-f]{64})", line.strip())
        if m:
            rec["__script__"] = m.group(1)
    return rec


def hash_gate(d: Path, exp: dict) -> list[str]:
    """Hash (bytes only) the prep outputs; compare with the fix plan and the prep record. Stops on mismatch."""
    rec = parse_record(d / F_RECORD)
    bad, lines = [], []
    for name in (F_MAP, F_KEY, F_BLIND, F_PAGE, F_STRATA):
        if not (d / name).exists():
            bad.append(f"{name}: file missing")
            continue
        h = sha256(d / name)
        if rec.get(name) != h:
            bad.append(f"{name}: sha256 differs from {F_RECORD}")
        want = {F_MAP: exp["map"], F_KEY: exp["key"]}.get(name)
        pre = {F_BLIND: exp["blind_prefix"], F_PAGE: exp["page_prefix"], F_STRATA: exp["strata_prefix"]}.get(name)
        if want is not None and h != want:
            bad.append(f"{name}: sha256 differs from the fix plan")
        if pre is not None and not h.startswith(pre):
            bad.append(f"{name}: sha256 prefix differs from the fix plan ({pre})")
        lines.append(f"sha256 {h}  {name}")
    if not rec.get("__script__", "").startswith(exp["prep_script_prefix"]):
        bad.append(f"{F_RECORD}: prep script sha256 does not start with {exp['prep_script_prefix']}")
    if bad:
        die("hash gate failed; map and key NOT opened:\n  " + "\n  ".join(bad))
    return lines


def load_blind(d: Path) -> list[dict]:
    rows = read_csv(d / F_BLIND, ["case_code", "claim_code", "claim"], "blind file")
    codes = [r["claim_code"] for r in rows]
    if len(rows) != N_CLAIMS or len(set(codes)) != N_CLAIMS:
        die(f"{F_BLIND}: expected {N_CLAIMS} distinct claim codes, got {len(set(codes))} of {len(rows)} rows")
    cc = Counter(r["case_code"] for r in rows)
    if len(cc) != N_CASES or set(cc.values()) != {CLAIMS_PER_CASE}:
        die(f"{F_BLIND}: expected {N_CASES} cases x {CLAIMS_PER_CASE} claims")
    for r in rows:
        if not r["claim_code"].startswith(r["case_code"]):
            die(f"{F_BLIND}: claim code {r['claim_code']} does not belong to case {r['case_code']}")
    return rows


def label_path(labels_dir: Path, name: str, overrides: dict) -> Path:
    return Path(overrides[name]) if name in overrides else labels_dir / f"labels_3b_{name}.csv"


def load_labels(path: Path, name: str, blind: list[dict]) -> dict:
    """-> claim_code -> label. Stops on any difference from the blind file (no case IDs exist here)."""
    rows = read_csv(path, LABEL_COLS, f"label file of {name}")
    who = {r["labeler"].strip().lower() for r in rows}
    if len(who) != 1:
        die(f"{path.name}: labeler column has {len(who)} different values")
    typed = who.pop()
    if not typed.startswith(name):  # page writes the typed name, e.g. "max" or "max_garcia"
        die(f"{path.name}: labeler column '{typed}' does not match labeler '{name}'")
    by = {}
    for r in rows:
        k = r["claim_code"]
        if k in by:
            die(f"{path.name}: claim {k} appears twice")
        by[k] = r
    bl = {r["claim_code"]: r for r in blind}
    if set(by) != set(bl):
        die(f"{path.name}: claim codes differ from {F_BLIND} "
            f"(missing {sorted(set(bl) - set(by))}, extra {sorted(set(by) - set(bl))})")
    out, blanks, badlab = {}, [], []
    for k, b in bl.items():
        r = by[k]
        if r["case_code"] != b["case_code"]:
            die(f"{path.name}: claim {k} has case code {r['case_code']}, blind file says {b['case_code']}")
        if r["claim"].replace("\r\n", "\n").strip() != b["claim"].replace("\r\n", "\n").strip():
            die(f"{path.name}: claim {k} text differs from {F_BLIND}")
        lab = r["human_label"].strip()
        if not lab:
            blanks.append(k)
        elif lab not in LABELS:
            badlab.append(k)
        out[k] = lab
    if badlab:
        die(f"{path.name}: invalid label value for {badlab}")
    if blanks:
        die(f"{path.name}: {len(blanks)} claim(s) have no label ({blanks}). A13.3 has no rule for "
            "blanks: stop here and write one before any number is computed")
    return out


def labeler_set(arg: str) -> list[str]:
    names = [x.strip().lower() for x in arg.split(",") if x.strip()]
    if len(set(names)) != len(names) or not set(names) <= set(ALL_LABELERS) or len(names) < 2:
        die(f"--labelers must be 2 or 3 distinct names from {ALL_LABELERS}")
    if "aeon" not in names:
        die("--labelers must include aeon")
    return [n for n in ALL_LABELERS if n in names]


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            die(f"--label expects name=path, got {it!r}")
        n, p = it.split("=", 1)
        out[n.strip().lower()] = p
    return out


def receipt(d: Path, labels_dir: Path, names: list[str], overrides: dict, exp: dict) -> None:
    rp = labels_dir / F_RECEIPT
    if rp.exists():
        die(f"{rp} already exists; a receipt is written once. Delete it only if you are sure no map/key was opened")
    blind = load_blind(d)
    lines = []
    for n in names:
        p = label_path(labels_dir, n, overrides)
        load_labels(p, n, blind)
        h = sha256(p)
        if n == "aeon" and h != exp["labels_aeon"]:
            die(f"{p.name}: sha256 differs from the hash recorded Nov 18 ({exp['labels_aeon'][:12]}...)")
        lines.append(f"{h}  {p.name}")
    rp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"label files ok vs {F_BLIND} ({N_CLAIMS} claims each, no blanks); map and key not read")
    print("\n".join(lines))
    print(f"wrote {rp}  (send these hashes for the fix plan, then run without --receipt)")


def check_receipt(labels_dir: Path, names: list[str], overrides: dict, exp: dict) -> list[str]:
    rp = labels_dir / F_RECEIPT
    if not rp.exists():
        die(f"missing {rp}: run with --receipt first")
    rec = {}
    for line in rp.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+(\S.*)$", line.strip())
        if m:
            rec[m.group(2)] = m.group(1)
    out = []
    for n in names:
        p = label_path(labels_dir, n, overrides)
        if not p.exists():
            die(f"missing label file of {n}: {p}")
        h = sha256(p)
        if rec.get(p.name) != h:
            die(f"{p.name}: sha256 differs from {F_RECEIPT} (file changed after the receipt, or not in it)")
        if n == "aeon" and h != exp["labels_aeon"]:
            die(f"{p.name}: sha256 differs from the hash recorded Nov 18")
        out.append(f"sha256 {h}  {p.name}")
    return out


# ------------------------------------------------------------------ map / key

def load_map_key(d: Path, blind: list[dict], exp: dict) -> tuple[dict, dict]:
    mrows = read_csv(d / F_MAP, MAP_COLS, "map")
    krows = read_csv(d / F_KEY, KEY_COLS, "key")
    bl = {r["claim_code"]: r for r in blind}
    m = {r["claim_code"]: r for r in mrows}
    k = {r["claim_code"]: r for r in krows}
    if len(m) != len(mrows) or set(m) != set(bl):
        die("map claim codes differ from the blind file")
    if len(k) != len(krows) or set(k) != set(bl):
        die("key claim codes differ from the blind file")
    case_of = defaultdict(set)
    for c, r in m.items():
        if r["case_code"] != bl[c]["case_code"]:
            die(f"map: claim {c} is under case {r['case_code']}, blind file says {bl[c]['case_code']}")
        if r["row"] not in ROWS:
            die(f"map: claim {c} has row {r['row']!r}")
        case_of[r["case_code"]].add((r["row"], r["case_id"], r["system"], r["ac1_correct"], r["fallback"]))
    if any(len(v) != 1 for v in case_of.values()):
        die("map: the two claims of a case disagree on row / case_id / system / AC@1")
    cases = {cc: next(iter(v)) for cc, v in case_of.items()}
    if len({v[1] for v in cases.values()}) != N_CASES:
        die("map: an incident appears more than once")
    per_row = Counter(v[0] for v in cases.values())
    if any(per_row[r] != 4 for r in ROWS):
        die(f"map: expected 4 cases per row, got {dict(per_row)}")
    n_fb = sum(v[4] == "True" for v in cases.values())
    if n_fb != exp["n_fallback"]:
        die(f"map: {n_fb} fallback cases, the prep record says {exp['n_fallback']}")
    for c, r in k.items():
        if r["judge_verified_label"] not in LABELS:
            die(f"key: claim {c} verified label {r['judge_verified_label']!r} is not one of the 3 labels")
        if r["judge_label"] not in LABELS:
            die(f"key: claim {c} raw label {r['judge_label']!r} is not one of the 3 labels")
    return m, k


def quote_group(v: str) -> str:
    v = (v or "").strip().lower()
    if v in ("0", "0.0", "false"):
        return "quote_ok=0"
    if v in ("1", "1.0", "true"):
        return "quote_ok=1"
    return "quote_ok=missing"


# ------------------------------------------------------------------ statistics

class Boot:
    """Case-cluster bootstrap weights shared by every statistic."""

    def __init__(self, claim_case_idx: np.ndarray, n_cases: int, b: int, seed: int):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_cases, size=(b, n_cases))
        cw = np.zeros((b, n_cases))
        for j in range(n_cases):
            cw[:, j] = (idx == j).sum(axis=1)
        self.w = np.vstack([np.ones((1, len(claim_case_idx))), cw[:, claim_case_idx]])  # row 0 = point


def onehot(codes: np.ndarray, k: int) -> np.ndarray:
    return (codes[:, None] == np.arange(k)[None, :]).astype(float)


def kappa_from_conf(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = c.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        po = np.trace(c, axis1=1, axis2=2) / n
        pe = (c.sum(axis=2) * c.sum(axis=1)).sum(axis=1) / n ** 2
        kap = np.where(np.isclose(1 - pe, 0), np.nan, (po - pe) / (1 - pe))
    return kap, po


def summarize(arr: np.ndarray) -> dict:
    point, boot = arr[0], arr[1:]
    ok = boot[~np.isnan(boot)]
    lo, hi = (np.percentile(ok, 2.5), np.percentile(ok, 97.5)) if len(ok) else (np.nan, np.nan)
    return {"point": point, "lo": lo, "hi": hi, "n_undefined": int(np.isnan(boot).sum())}


def pair_stats(bt: Boot, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict:
    """a, b: codes 0=S, 1=P, 2=U. Returns kappa3, kappa_bin, agree3, agree_bin arrays (B+1)."""
    w = bt.w * mask[None, :]
    c3 = np.einsum("bn,ni,nj->bij", w, onehot(a, 3), onehot(b, 3))
    ab, bb = (a != 0).astype(int), (b != 0).astype(int)
    c2 = np.einsum("bn,ni,nj->bij", w, onehot(ab, 2), onehot(bb, 2))
    k3, a3 = kappa_from_conf(c3)
    k2, a2 = kappa_from_conf(c2)
    return {"kappa_3label": k3, "kappa_s_vs_not": k2, "agree_3label": a3, "agree_s_vs_not": a2}


def fleiss(bt: Boot, codes: np.ndarray, k: int) -> np.ndarray:
    """codes: (n_claims, n_raters). Weighted Fleiss kappa."""
    nr = codes.shape[1]
    cnt = np.stack([(codes == j).sum(axis=1) for j in range(k)], axis=1).astype(float)
    p_i = ((cnt ** 2).sum(axis=1) - nr) / (nr * (nr - 1))
    w = bt.w
    sw = w.sum(axis=1)
    pbar = (w * p_i[None, :]).sum(axis=1) / sw
    pj = (w @ cnt) / (nr * sw[:, None])
    pe = (pj ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isclose(1 - pe, 0), np.nan, (pbar - pe) / (1 - pe))


def ratio(bt: Boot, num: np.ndarray, den: np.ndarray) -> tuple[np.ndarray, int, int]:
    n = bt.w @ (num & den).astype(float)
    d = bt.w @ den.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(d == 0, np.nan, n / d)
    return r, int((num & den).sum()), int(den.sum())


def fmt(x, nd=3) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ------------------------------------------------------------------ main analysis

def analyse(d: Path, out: Path, names: list[str], labels: dict, blind: list[dict], m: dict, k: dict,
            hash_lines: list[str], test_note: str) -> None:
    codes = [r["claim_code"] for r in blind]
    case_codes = sorted({r["case_code"] for r in blind})
    cidx = np.array([case_codes.index(m[c]["case_code"]) for c in codes])
    bt = Boot(cidx, len(case_codes), B, SEED)
    enc = {lab: i for i, lab in enumerate(LABELS)}
    H = {n: np.array([enc[labels[n][c]] for c in codes]) for n in names}
    J = {"verified": np.array([enc[k[c]["judge_verified_label"]] for c in codes]),
         "raw": np.array([enc[k[c]["judge_label"]] for c in codes])}
    full = np.ones(len(codes), bool)
    have_major = len(names) == 3
    if have_major:
        stack = np.stack([H[n] for n in names], axis=1)
        maj = np.full(len(codes), -1)
        for i in range(len(codes)):
            v, c = Counter(stack[i]).most_common(1)[0]
            if c >= 2:
                maj[i] = v
        has_maj = maj >= 0
    refs = [(n, H[n], full) for n in names] + ([("majority", maj, has_maj)] if have_major else [])

    agree_rows = []
    raw = {}  # unrounded point / CI for the pass rule (the CSV and summary show 3 decimals)

    def add(comp, ref, judge, scope_n, stats: dict):
        for metric, arr in stats.items():
            s = summarize(arr)
            raw[(comp, judge, ref, metric)] = s
            agree_rows.append({"comparison": comp, "a": judge, "b": ref, "metric": metric, "n_claims": scope_n,
                               "point": fmt(s["point"]), "ci_lo": fmt(s["lo"]), "ci_hi": fmt(s["hi"]),
                               "n_undefined": s["n_undefined"]})

    for jn, jv in J.items():
        for rn, rv, mask in refs:
            add("judge_vs_human", rn, f"judge_{jn}", int(mask.sum()), pair_stats(bt, jv, np.where(mask, rv, 0), mask))
    for i, a in enumerate(names):
        for b2 in names[i + 1:]:
            add("human_vs_human", b2, a, len(codes), pair_stats(bt, H[a], H[b2], full))
    if have_major:
        st = np.stack([H[n] for n in names], axis=1)
        add("humans_fleiss", "all 3", "humans", len(codes),
            {"fleiss_3label": fleiss(bt, st, 3), "fleiss_s_vs_not": fleiss(bt, (st != 0).astype(int), 2)})

    # error rates (false support = primary on majority + verified)
    rate_rows = []
    for jn, jv in J.items():
        for rn, rv, mask in refs:
            for rate, hum, jud in (("false_support", 2, "S"), ("over_credit", 1, "S"), ("under_credit", 0, "notS")):
                den = mask & (rv == hum)
                num = (jv == 0) if jud == "S" else (jv != 0)
                arr, nn, dd = ratio(bt, num, den)
                s = summarize(arr)
                raw[("rate", rate, f"judge_{jn}", rn)] = s
                rate_rows.append({"rate": rate, "judge": f"judge_{jn}", "reference": rn,
                                  "definition": {"false_support": "human U -> judge S",
                                                 "over_credit": "human P -> judge S",
                                                 "under_credit": "human S -> judge not S"}[rate],
                                  "count": nn, "denominator": dd, "point": fmt(s["point"]),
                                  "ci_lo": fmt(s["lo"]), "ci_hi": fmt(s["hi"]), "n_undefined": s["n_undefined"]})

    # confusion matrices
    conf_rows = []
    matches = sorted({k[c]["judge_match"] for c in codes})
    jcols = {"verified": [k[c]["judge_verified_label"] for c in codes],
             "raw": [k[c]["judge_label"] for c in codes],
             "raw_match": [k[c]["judge_match"] for c in codes]}
    for rn, rv, mask in refs:
        for jn, jl in jcols.items():
            cats = LABELS if jn != "raw_match" else matches
            for hl in range(3):
                row = {"human": rn, "judge": jn, "human_label": LABELS[hl]}
                for cat in cats:
                    row[f"judge={cat}"] = sum(1 for i in range(len(codes)) if mask[i] and rv[i] == hl and jl[i] == cat)
                conf_rows.append(row)

    # per-claim table
    claim_rows = []
    for i, c in enumerate(codes):
        r = {"claim_code": c, "case_code": m[c]["case_code"], "row": m[c]["row"], "case_id": m[c]["case_id"],
             "system": m[c]["system"], "ac1_correct": m[c]["ac1_correct"], "fallback": m[c]["fallback"],
             "claim_id": m[c]["claim_id"], "claim": next(b["claim"] for b in blind if b["claim_code"] == c)}
        for n in names:
            r[f"h_{n}"] = SHORT[labels[n][c]]
        r["majority"] = (SHORT[LABELS[maj[i]]] if maj[i] >= 0 else "none") if have_major else "n/a (<3 labelers)"
        r["judge_verified"] = SHORT[k[c]["judge_verified_label"]]
        r["judge_raw"] = SHORT[k[c]["judge_label"]]
        r["judge_match"] = k[c]["judge_match"]
        r["quote_check"] = quote_group(k[c]["judge_quote_ok"])
        claim_rows.append(r)

    # breakdowns (descriptive)
    br_rows = []
    groups = {"all": lambda c: "all", "quote_check": lambda c: quote_group(k[c]["judge_quote_ok"]),
              "row": lambda c: m[c]["row"], "ac1": lambda c: "correct" if m[c]["ac1_correct"] == "True" else "wrong",
              "system": lambda c: m[c]["system"]}
    for rn, rv, mask in refs:
        for gname, gf in groups.items():
            for g in sorted({gf(c) for c in codes}):
                sel = [i for i, c in enumerate(codes) if gf(c) == g and mask[i]]
                jv = J["verified"]
                br_rows.append({
                    "reference": rn, "by": gname, "group": g,
                    "claims": sum(1 for c in codes if gf(c) == g), "with_reference": len(sel),
                    "human_S": sum(rv[i] == 0 for i in sel), "human_P": sum(rv[i] == 1 for i in sel),
                    "human_U": sum(rv[i] == 2 for i in sel),
                    "judge_S": sum(jv[i] == 0 for i in sel), "judge_P": sum(jv[i] == 1 for i in sel),
                    "judge_U": sum(jv[i] == 2 for i in sel),
                    "agree_3label": sum(rv[i] == jv[i] for i in sel),
                    "agree_s_vs_not": sum((rv[i] == 0) == (jv[i] == 0) for i in sel),
                    "false_support": f"{sum(rv[i] == 2 and jv[i] == 0 for i in sel)}/{sum(rv[i] == 2 for i in sel)}"})

    # pass rule
    missing = [n for n in ALL_LABELERS if n not in names]
    if have_major:
        kb = next(r for r in agree_rows if r["comparison"] == "judge_vs_human" and r["a"] == "judge_verified"
                  and r["b"] == "majority" and r["metric"] == "kappa_s_vs_not")
        fs = next(r for r in rate_rows if r["rate"] == "false_support" and r["judge"] == "judge_verified"
                  and r["reference"] == "majority")
        lo = float(raw[("judge_vs_human", "judge_verified", "majority", "kappa_s_vs_not")]["lo"])
        c1 = (not np.isnan(lo)) and lo > PASS_KAPPA_LO
        n_u = fs["denominator"]
        fs_est = n_u >= PASS_MIN_U
        c2 = fs_est and float(raw[("rate", "false_support", "judge_verified", "majority")]["point"]) <= PASS_FS_MAX
        verdict = "VALIDATED SECONDARY METRIC" if (c1 and c2) else "SUPPLEMENTARY AUTOMATED MEASURE"
        rule = [f"(1) kappa S-vs-not, judge verified vs majority: {kb['point']} [{kb['ci_lo']}, {kb['ci_hi']}], "
                f"n = {kb['n_claims']}; CI lower bound > {PASS_KAPPA_LO}: {'MET' if c1 else 'NOT MET'}",
                f"(2) false support: {fs['count']}/{n_u} = {fs['point'] or 'n/a'}; majority-Unsupported claims "
                f"{n_u} (need >= {PASS_MIN_U}): " + (f"{'MET' if c2 else 'NOT MET'} (<= {PASS_FS_MAX:.0%})"
                                                     if fs_est else "NOT ESTIMABLE -> NOT MET")]
        n_split = int((~has_maj).sum())
    else:
        verdict = "SUPPLEMENTARY AUTOMATED MEASURE"
        rule = [f"fewer than 3 labelers (missing: {', '.join(missing)}): no majority, pass rule cannot be met (A13.3)"]
        n_split = None

    # write
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "agreement_3b.csv", agree_rows)
    write_csv(out / "error_rates_3b.csv", rate_rows)
    write_csv(out / "confusion_3b.csv", conf_rows, sorted({k2 for r in conf_rows for k2 in r},
                                                           key=lambda x: (not x.startswith("human") and x != "judge", x)))
    write_csv(out / "claims_3b.csv", claim_rows)
    write_csv(out / "breakdown_3b.csv", br_rows)

    def pick(comp, a, b, metric):
        r = next((r for r in agree_rows if r["comparison"] == comp and r["a"] == a and r["b"] == b
                  and r["metric"] == metric), None)
        return f"{r['point']} [{r['ci_lo']}, {r['ci_hi']}]" if r else "-"

    lines = [f"# Phase 3b agreement (A13.3), {len(names)} labelers: {', '.join(names)}"
             + (f"; MISSING: {', '.join(missing)}" if missing else ""), ""]
    if test_note:
        lines += [test_note, ""]
    lines += ["Inputs (sha256):"] + [f"- {h}" for h in hash_lines] + [
        f"- script {sha256(Path(__file__).resolve())}", "",
        f"Claims {len(codes)}, cases {len(case_codes)}; bootstrap B = {B}, seed {SEED}, resampling cases; "
        "unweighted over the sample as drawn (AC@1-wrong cases oversampled).", ""]
    if have_major:
        lines.append(f"Majority: {int(has_maj.sum())} claims with a majority, {n_split} three-way splits (left out of "
                     f"majority comparisons). Majority labels: "
                     + ", ".join(f"{SHORT[LABELS[j]]} {int((maj == j).sum())}" for j in range(3)))
        lines.append("")
    lines += ["| judge vs | n | kappa 3 labels | kappa S vs not | agree S vs not |", "|---|---:|---|---|---|"]
    for jn in ("verified", "raw"):
        for rn, _, mask in refs:
            lines.append(f"| {jn} vs {rn} | {int(mask.sum())} | {pick('judge_vs_human', f'judge_{jn}', rn, 'kappa_3label')}"
                         f" | {pick('judge_vs_human', f'judge_{jn}', rn, 'kappa_s_vs_not')}"
                         f" | {pick('judge_vs_human', f'judge_{jn}', rn, 'agree_s_vs_not')} |")
    lines += ["", "| humans | kappa 3 labels | kappa S vs not | agree 3 labels |", "|---|---|---|---|"]
    for i, a in enumerate(names):
        for b2 in names[i + 1:]:
            lines.append(f"| {a} vs {b2} | {pick('human_vs_human', a, b2, 'kappa_3label')} | "
                         f"{pick('human_vs_human', a, b2, 'kappa_s_vs_not')} | {pick('human_vs_human', a, b2, 'agree_3label')} |")
    if have_major:
        lines.append(f"| Fleiss (3) | {pick('humans_fleiss', 'humans', 'all 3', 'fleiss_3label')} | "
                     f"{pick('humans_fleiss', 'humans', 'all 3', 'fleiss_s_vs_not')} | |")
    else:
        lines.append("| Fleiss | not computed (fewer than 3 labelers) | | |")
    lines += ["", "| rate (judge verified) | reference | count / denom | point [95% CI] |", "|---|---|---|---|"]
    for r in rate_rows:
        if r["judge"] == "judge_verified":
            lines.append(f"| {r['rate']} ({r['definition']}) | {r['reference']} | {r['count']}/{r['denominator']} | "
                         f"{r['point'] or 'n/a'} [{r['ci_lo']}, {r['ci_hi']}] |")
    qref = "majority" if have_major else names[0]
    lines += ["", f"Quote-check breakdown vs {qref} (judge verified; descriptive):"]
    for r in br_rows:
        if r["reference"] == qref and r["by"] == "quote_check":
            lines.append(f"- {r['group']}: {r['with_reference']} claims, human S/P/U {r['human_S']}/{r['human_P']}/"
                         f"{r['human_U']}, judge S/P/U {r['judge_S']}/{r['judge_P']}/{r['judge_U']}, "
                         f"agree S-vs-not {r['agree_s_vs_not']}, false support {r['false_support']}")
    undefined = sum(int(r["n_undefined"]) for r in agree_rows + rate_rows)
    lines += ["", "## Pass rule (A13.3)"] + [f"- {x}" for x in rule] + [f"- **Decision: {verdict}**", ""]
    if undefined:
        lines.append(f"Note: some statistics were undefined in some resamples (total {undefined}; per row in "
                     "the n_undefined columns); CIs use the defined resamples.")
    (out / "summary_3b.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/grounding_3b", help="prep output folder")
    ap.add_argument("--labels-dir", default=None, help="default: <dir>/labels")
    ap.add_argument("--labelers", default="aeon,daniel,max", help="2 or 3 of aeon,daniel,max (aeon required)")
    ap.add_argument("--label", action="append", help="name=path if a file is not labels_3b_<name>.csv")
    ap.add_argument("--receipt", action="store_true", help="step 1: hash + check label files; map/key untouched")
    ap.add_argument("--out", default="results/grounding_3b_agreement")
    ap.add_argument("--expected", default=None, help=argparse.SUPPRESS)  # tests only
    a = ap.parse_args()

    d = Path(a.dir)
    labels_dir = Path(a.labels_dir) if a.labels_dir else d / "labels"
    names, overrides = labeler_set(a.labelers), parse_overrides(a.label)
    if set(overrides) - set(names):
        die(f"--label given for a labeler not in --labelers: {sorted(set(overrides) - set(names))}")
    exp, test_note = dict(EXPECTED), ""
    if a.expected:
        exp.update(json.loads(Path(a.expected).read_text()))
        test_note = f"NOTE: expected hashes from {a.expected} (test mode), not the fix plan values"
        print(test_note)
    missing = [n for n in ALL_LABELERS if n not in names]
    if missing:
        print(f"NOTE: labelers missing: {', '.join(missing)} (reported as missing; no majority, A13.3)")

    if a.receipt:
        receipt(d, labels_dir, names, overrides, exp)
        return

    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        die(f"{out} exists and is not empty; pick a new --out")
    hl = hash_gate(d, exp)                                   # a) map/key hashed, not opened
    hl += check_receipt(labels_dir, names, overrides, exp)   # b)
    blind = load_blind(d)
    labels = {n: load_labels(label_path(labels_dir, n, overrides), n, blind) for n in names}  # c)
    print(f"gate ok: {len(hl)} file hashes match; {len(names)} label files x {N_CLAIMS} claims match {F_BLIND}")
    m, k = load_map_key(d, blind, exp)                       # d) first time the map / key are parsed
    analyse(d, out, names, labels, blind, m, k, hl, test_note)


if __name__ == "__main__":
    main()
