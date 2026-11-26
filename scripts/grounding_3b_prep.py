#!/usr/bin/env python3
"""Phase 3b prep (Addendum A13.3): sample held-out explanation claims and build a BLIND labeling page.

Read-only. No LLM calls. Writes into a NEW (or empty) --out folder:
  labeling_page_3b.html    send this to the labelers (self-contained, works offline)
  blind_3b.csv             case_code, claim_code, claim (no case IDs, rows or judge labels)
  strata_counts_3b.csv     eligible / picked / fallback counts per cell (no case IDs)
  DO_NOT_OPEN_map_3b.csv   case codes -> row, case_id, system, AC@1, run path (ground truth!)
  DO_NOT_OPEN_key_3b.csv   judge labels for the sampled claims
  prep_3b_record.txt       seed, counts, sha256 of every output (re-checked in the agreement step)

Aeon labels too, so the console prints counts only: no case IDs, rows per code, or labels.
Do not open the DO_NOT_OPEN files until all 3 label files are in.

Sampling rules (A13.3, fixed before sampling):
  rows hybrid, llm_only, no_critic, react, retrieval_only; explanation claims only;
  (row, case) eligible if >= 2 explanation claims got a judge label (PARSE_FAIL = no label; the
  label VALUE is never read for sampling); 5 Phase 3 example incidents excluded;
  cells = row x system (ss, tt) x AC@1 (correct, wrong), 1 case per cell, scarcest cell first;
  each incident at most once; empty cell -> same row + system, other AC@1 (counted as fallback);
  2 claims per case; one random.Random(258) stream for picks, claims, page order.

Example (repo root):
  python scripts/grounding_3b_prep.py
  -> results/grounding_3b/labeling_page_3b.html
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_grounding import DEFAULT_MANIFEST, incident_query, load_manifest, repo_path  # noqa: E402

SEED = 258
ROWS = ["hybrid", "llm_only", "no_critic", "react", "retrieval_only"]
SYSTEMS = ["ss", "tt"]
SYS_ALIASES = {"ss": "ss", "tt": "tt", "sock_shop": "ss", "train_ticket": "tt", "ob": "ob",
               "online_boutique": "ob"}
CASES_PER_CELL = 1
CLAIMS_PER_CASE = 2
MIN_CLAIMS = 2
N_CASES_PER_ROW = 60
# Incidents whose claims + judge labels were shown in the Phase 3 wrong + high examples (A13.3).
EXCLUDED = ["re2tt_ts-auth-service_cpu_2", "re2ss_orders_delay_1", "re2tt_ts-route-service_delay_2",
            "re2tt_ts-order-service_disk_3", "re2ss_orders_disk_1"]
FAIL = "PARSE_FAIL"
LABELS = ["Supported", "Partially Supported", "Unsupported"]
CLAIM_COLS = ["row", "case_id", "claim_id", "claim_type", "origin", "claim", "run_path", "verified_label"]
KEY_COLS = ["label", "verified_label", "match", "quote_ok", "evidence"]  # copied to the key only

# Known numbers (fix plan Phase 1 / Phase 3); any mismatch stops before sampling.
EXPECTED = {
    "ac1/hybrid": 48, "ac1/llm_only": 42, "ac1/no_critic": 15, "ac1/react": 36, "ac1/retrieval_only": 5,
    "expl_claims/hybrid": 292, "expl_claims/react": 355, "expl_claims/retrieval_only": 268,
    "parse_fail_claims_expl/all": 34,
}

OUT_PAGE = "labeling_page_3b.html"
OUT_BLIND = "blind_3b.csv"
OUT_MAP = "DO_NOT_OPEN_map_3b.csv"
OUT_KEY = "DO_NOT_OPEN_key_3b.csv"
OUT_STRATA = "strata_counts_3b.csv"
OUT_RECORD = "prep_3b_record.txt"
PAGE_WARN_BYTES = 9_500_000  # Discord free upload limit is 10 MB

ROW_WORDS = re.compile(r"\b(no_critic|llm_only|retrieval_only|hybrid_7b|code_rule|hybrid|react)\b", re.I)
CASE_ID_PATTERN = re.compile(r"re2[-_]?(ob|ss|tt)[-_]", re.I)


def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def read_csv(path: Path, need: list[str]) -> list[dict]:
    if not path.exists():
        die(f"missing {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    miss = [c for c in need if c not in cols]
    if miss:
        die(f"{path} is missing columns {miss}")
    if not rows:
        die(f"{path} is empty")
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(p: str) -> Path:
    q = Path(p)
    if q.is_absolute() or q.exists():
        return q
    return repo_path(q)


# ------------------------------------------------------------------ loading

def load_master(path: Path) -> tuple[dict, dict]:
    """-> correct[(row, case)] (bool), system[case] ('ss'/'tt')."""
    correct, system = {}, {}
    for r in read_csv(path, ["row", "case_id", "system", "correct"]):
        k = (r["row"], r["case_id"])
        if k in correct:
            die(f"duplicate {k} in {path}")
        if r["correct"] not in ("True", "False"):
            die(f"{k}: correct={r['correct']!r} in {path}")
        s = SYS_ALIASES.get(r["system"].strip().lower())
        if s is None:
            die(f"{k}: unknown system {r['system']!r} in {path}")
        if system.setdefault(r["case_id"], s) != s:
            die(f"{r['case_id']}: system differs between rows in {path}")
        correct[k] = r["correct"] == "True"
    for row in ROWS:
        n = sum(1 for (rw, _) in correct if rw == row)
        if n != N_CASES_PER_ROW:
            die(f"{path}: row {row} has {n} cases, expected {N_CASES_PER_ROW}")
    return correct, system


def load_claims(path: Path) -> tuple[dict, dict]:
    """-> pub[(row, case)] = list of claim dicts WITHOUT judge values (+ 'judged' flag),
          judge[(row, case, claim_id)] = judge columns (used only to write the key)."""
    pub, judge, seen = defaultdict(list), {}, set()
    for r in read_csv(path, CLAIM_COLS):
        k = (r["row"], r["case_id"], r["claim_id"])
        if k in seen:
            die(f"duplicate claim {k} in {path}")
        seen.add(k)
        if r["origin"] not in ("structured", "explanation"):
            die(f"{k}: unexpected origin {r['origin']!r}")
        if r["verified_label"] not in LABELS + [FAIL]:
            die(f"{k}: unexpected verified_label value in {path}")
        pub[(r["row"], r["case_id"])].append({
            "row": r["row"], "case_id": r["case_id"], "claim_id": r["claim_id"],
            "claim_type": r["claim_type"], "origin": r["origin"], "claim": r["claim"],
            "run_path": r["run_path"], "judged": r["verified_label"] != FAIL})
        judge[k] = {c: r.get(c, "") for c in KEY_COLS}
    return pub, judge


def step0(correct: dict, pub: dict, expected: dict) -> None:
    got = {}
    for row in ROWS:
        got[f"ac1/{row}"] = sum(v for (rw, _), v in correct.items() if rw == row)
    allc = [c for cl in pub.values() for c in cl]
    for row in ("hybrid", "react", "retrieval_only"):
        got[f"expl_claims/{row}"] = sum(1 for c in allc if c["row"] == row and c["origin"] == "explanation")
    got["parse_fail_claims_expl/all"] = sum(1 for c in allc if c["origin"] == "explanation" and not c["judged"])
    bad = [f"{k}: expected {v}, got {got.get(k)}" for k, v in expected.items() if got.get(k) != v]
    orphans = [k for k in pub if k[0] in ROWS and k not in correct]
    if orphans:
        bad.append(f"{len(orphans)} (row, case) in the claims file but not in the master table")
    missing_ex = [c for c in EXCLUDED if not any(cid == c for (_, cid) in correct)]
    if missing_ex:
        bad.append(f"excluded incidents not found in master table: {missing_ex}")
    if bad:
        die("step 0 mismatch, nothing written:\n  " + "\n  ".join(bad))
    print(f"step 0: {len(expected) + 2}/{len(expected) + 2} checks ok")


# ------------------------------------------------------------------ sampling

def sample(correct: dict, system: dict, pub: dict, rng: random.Random):
    elig = {}
    for (row, cid), cl in pub.items():
        if row not in ROWS or cid in EXCLUDED:
            continue
        ex = sorted((c for c in cl if c["origin"] == "explanation" and c["judged"]),
                    key=lambda c: int(c["claim_id"]))
        if len(ex) >= MIN_CLAIMS:
            elig[(row, cid)] = ex
    cells = [(row, s, ac) for row in ROWS for s in SYSTEMS for ac in (True, False)]
    cand = {cell: sorted(cid for (rw, cid) in elig
                         if rw == cell[0] and system[cid] == cell[1] and correct[(rw, cid)] == cell[2])
            for cell in cells}
    order = sorted(cells, key=lambda c: (len(cand[c]), cells.index(c)))
    used, picks, empty = set(), [], []
    for cell in order:
        for _ in range(CASES_PER_CELL):
            avail = [c for c in cand[cell] if c not in used]
            if not avail:
                empty.append(cell)
                continue
            cid = rng.choice(avail)
            used.add(cid)
            picks.append({"cell": cell, "row": cell[0], "case_id": cid, "fallback": False})
    for cell in empty:
        other = (cell[0], cell[1], not cell[2])
        avail = [c for c in cand[other] if c not in used]
        if not avail:
            die(f"cannot fill a cell (row {cell[0]}, {cell[1]}, AC@1 {'correct' if cell[2] else 'wrong'}): "
                "no unused eligible case in the same row + system")
        cid = rng.choice(avail)
        used.add(cid)
        picks.append({"cell": cell, "row": cell[0], "case_id": cid, "fallback": True})
    for p in picks:
        p["claims"] = rng.sample(elig[(p["row"], p["case_id"])], CLAIMS_PER_CASE)
    rng.shuffle(picks)
    for i, p in enumerate(picks, 1):
        p["code"] = f"C{i:02d}"
        rng.shuffle(p["claims"])
        for j, c in enumerate(p["claims"]):
            c["code"] = f"{p['code']}{'abcdefghij'[j]}"
    strata = []
    for cell in cells:
        strata.append({"row": cell[0], "system": cell[1], "ac1": "correct" if cell[2] else "wrong",
                       "eligible_cases": len(cand[cell]),
                       "picked": sum(1 for p in picks if p["cell"] == cell and not p["fallback"]),
                       "filled_by_fallback": sum(1 for p in picks if p["cell"] == cell and p["fallback"])})
    return picks, strata, len(elig)


# ------------------------------------------------------------------ page

def esc(x) -> str:
    return html.escape(str(x), quote=True)


def content_text(c) -> str:
    return c if isinstance(c, str) else json.dumps(c, indent=2, sort_keys=True, default=str)


def forbidden_tokens(picks: list[dict], manifest: dict) -> list[tuple[str, str]]:
    """(token, kind) pairs that must not appear anywhere on the page or in the blind CSV."""
    toks = [("runs/heldout", "run path")]
    for p in picks:
        toks.append((p["case_id"], "a case ID"))
        cdir = str(manifest[p["case_id"]].get("case_dir", "")).strip().replace("\\", "/").rstrip("/")
        if cdir:
            toks.append((cdir, "a case data directory"))
            parts = [x for x in cdir.split("/") if x]
            if len(parts) >= 2:  # manifest layout is data/raw/rcaeval/<case_id>
                toks.append(("/".join(parts[-2:]), "a case data directory"))
    return toks


def leak_check(texts: dict[str, str], toks: list[tuple[str, str]]) -> tuple[list[str], Counter]:
    """texts: where -> raw text. Returns (hard leaks, row-word counts)."""
    leaks, rows = [], Counter()
    for where, t in texts.items():
        low = t.lower()
        for tok, kind in toks:
            if tok.lower() in low:
                leaks.append(f"{where}: contains {kind}")
        if CASE_ID_PATTERN.search(t):
            leaks.append(f"{where}: contains the case-ID pattern re2.._")
        for m in ROW_WORDS.finditer(t):
            rows[(where, m.group(1).lower())] += 1
    return sorted(set(leaks)), rows


def render_case(n: int, total: int, p: dict, run: dict, query: str) -> str:
    code = p["code"]
    ledger = run.get("evidence_ledger") or []
    ids = ["INPUT"] + [str(i.get("id")) for i in ledger]
    ev = [f'<div class="ev"><div class="evh"><button class="eid" data-id="INPUT">INPUT</button>'
          f'<span class="src">case input (incident description)</span></div><pre>{esc(query)}</pre></div>']
    for item in ledger:
        args = json.dumps(item.get("args"), sort_keys=True, default=str)
        ev.append(f'<div class="ev"><div class="evh"><button class="eid" data-id="{esc(item.get("id"))}">'
                  f'{esc(item.get("id"))}</button><span class="src">{esc(item.get("source"))}</span>'
                  f'<code class="args">{esc(args)}</code></div><pre>{esc(content_text(item.get("content")))}</pre></div>')
    cl = []
    for c in p["claims"]:
        k = c["code"]
        radios = "".join(f'<label class="lab"><input type="radio" name="{esc(k)}" value="{esc(lab)}"> {esc(lab)}</label>'
                         for lab in LABELS)
        cl.append(f'<div class="claim" data-key="{esc(k)}" data-ids="{esc(" ".join(ids))}">'
                  f'<div class="ct">claim {esc(k)}</div><div class="ctext">{esc(c["claim"])}</div>'
                  f'<div class="ctrl">{radios}</div>'
                  f'<div class="ctrl"><input class="evin" placeholder="evidence IDs, e.g. E003 E007 (click an ID on the left)">'
                  f'<span class="warn"></span></div>'
                  f'<div class="ctrl"><input class="note" placeholder="note (optional)"></div></div>')
    return (f'<section class="case" data-n="{n}"><header><h2>Case {esc(code)} ({n}/{total})</h2>'
            f'<div class="meta">{len(ledger)} evidence items &middot; {len(p["claims"])} claims</div></header>'
            f'<div class="cols"><div class="evcol"><h3>Evidence</h3>{"".join(ev)}</div>'
            f'<div class="clcol"><h3>Claims to label</h3>{"".join(cl)}'
            f'<div class="nav"><button class="prev">&larr; previous case</button>'
            f'<button class="next">next case &rarr;</button></div></div></div></section>')


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Grounding labeling, round 2 (blind)</title>
<style>
:root{--bg:#f6f6f3;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#ddddd6;--acc:#2f5d8a;--ok:#2e7d4f;--bad:#b23b3b}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}
.top{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.top h1{font-size:15px;margin:0 8px 0 0}.top select,.top button,.top input{font:inherit;padding:4px 8px}
#who.missing{outline:2px solid var(--bad)}#prog{color:var(--mute)}
#dl{background:var(--acc);color:#fff;border:0;border-radius:4px;padding:5px 12px;cursor:pointer}
details.rub{padding:0 16px;margin:8px 0}details.rub div{background:var(--card);border:1px solid var(--line);padding:8px 12px;border-radius:4px}
.case{display:none;padding:12px 16px}.case.on{display:block}.case h2{font-size:16px;margin:0}.meta{color:var(--mute);font-size:12px}
.cols{display:grid;grid-template-columns:minmax(0,3fr) minmax(320px,2fr);gap:16px;margin-top:10px}
.evcol{max-height:calc(100vh - 120px);overflow:auto;padding-right:6px}.clcol{position:sticky;top:56px;align-self:start;max-height:calc(100vh - 70px);overflow:auto}
h3{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute);margin:4px 0 8px}
.ev{background:var(--card);border:1px solid var(--line);border-radius:4px;margin-bottom:8px}
.evh{display:flex;gap:8px;align-items:baseline;padding:6px 8px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.eid{font:600 12px ui-monospace,monospace;background:#e8eef5;color:var(--acc);border:1px solid #c9d6e4;border-radius:3px;cursor:pointer;padding:1px 6px}
.src{font-weight:600}.args{font-size:11px;color:var(--mute);word-break:break-all}
pre{margin:0;padding:8px;white-space:pre-wrap;word-break:break-word;font:12px/1.4 ui-monospace,monospace}
.claim{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:4px;padding:8px 10px;margin-bottom:10px}
.claim.active{border-color:var(--acc);border-left-color:var(--acc)}.claim.done{border-left-color:var(--ok)}
.ct{font-size:12px;color:var(--mute)}.ctext{font-size:15px;margin:4px 0 8px}.ctrl{margin:4px 0}.lab{margin-right:10px;white-space:nowrap;cursor:pointer}
.evin,.note{width:100%;font:inherit;padding:4px 6px;border:1px solid var(--line);border-radius:3px}.warn{color:var(--bad);font-size:12px}
.nav{display:flex;justify-content:space-between;margin-top:8px}.nav button{font:inherit;padding:4px 10px}
</style></head><body>
<div class="top"><h1>Grounding labeling, round 2 (blind)</h1>
<input id="who" placeholder="your name (required)" size="16">
<select id="jump"></select><span id="prog"></span>
<button id="dl">Download CSV</button><span style="color:var(--mute);font-size:12px">saves labels_3b_&lt;name&gt;.csv &middot; progress autosaves in this browser</span></div>
<details class="rub" open><summary>How to label (read once)</summary><div>
Label alone and do not discuss labels with the others until all three files are in. Your first labels are final.<br>
Each case shows an incident description (INPUT) and the evidence a diagnosis system collected, then 2 claims from its explanation.
Judge each claim against the evidence on this page only (closed set: no outside knowledge, plausibility does not count).<br>
<b>Supported</b>: the evidence directly states what the claim asserts; it follows from the evidence alone.<br>
<b>Partially Supported</b>: the evidence has related information, but the claim needs inference, extrapolation, or domain knowledge not in the evidence.<br>
<b>Unsupported</b>: the evidence contradicts the claim (different number, service, metric, or direction), or nothing in the evidence is about it.<br>
Compare numbers, service names, metric names and times exactly; a value counts only for the service it is listed under.
Root-cause / fault-category / indicator claims: Supported if the evidence clearly singles out what the claim names; Unsupported if it clearly singles out something else or the named metric does not appear; Partially only if the evidence is mixed.
A generic caveat in a tool output (such as "not causal proof") does not by itself lower the label.<br>
Evidence IDs: the items your label rests on (INPUT for the case input). Leave empty for Unsupported-because-absent. Click an ID on the left to add it to the selected claim.<br>
When done: type your name at the top, press Download CSV, and send the file back.
</div></details>
__CASES__
<script id="blind" type="application/json">__BLIND__</script>
<script>
(function(){
var BLIND=JSON.parse(document.getElementById('blind').textContent), STORE='calib3b:__HASH__';
var state={}; try{state=JSON.parse(localStorage.getItem(STORE)||'{}')}catch(e){state={}}
if(!state._claims)state._claims={};
function save(){try{localStorage.setItem(STORE,JSON.stringify(state))}catch(e){}}
var who=document.getElementById('who'); who.value=state._who||'';
who.addEventListener('input',function(){state._who=who.value;who.classList.remove('missing');save()});
var cases=[].slice.call(document.querySelectorAll('.case')), jump=document.getElementById('jump'), active=null, cur=0;
cases.forEach(function(s,i){var o=document.createElement('option');o.value=i;o.textContent=s.querySelector('h2').textContent;jump.appendChild(o)});
function show(i){cur=Math.max(0,Math.min(cases.length-1,i));cases.forEach(function(s,j){s.classList.toggle('on',j===cur)});jump.value=cur;
  setActive(cases[cur].querySelector('.claim'));window.scrollTo(0,0);cases[cur].querySelector('.evcol').scrollTop=0}
function setActive(c){if(active)active.classList.remove('active');active=c;if(c)c.classList.add('active')}
function norm(s){return s.toUpperCase().split(/[\s,;\/]+/).filter(Boolean)}
function check(c){var ok=c.dataset.ids.split(' '),v=norm(c.querySelector('.evin').value),bad=v.filter(function(x){return ok.indexOf(x)<0});
  c.querySelector('.warn').textContent=bad.length?'not in this case: '+bad.join(' '):''}
function sync(c){var k=c.dataset.key,r=c.querySelector('input[type=radio]:checked');
  state._claims[k]={label:r?r.value:'',evidence:norm(c.querySelector('.evin').value).join(' '),note:c.querySelector('.note').value};
  c.classList.toggle('done',!!state._claims[k].label);check(c);save();prog()}
function prog(){var d=BLIND.filter(function(b){var s=state._claims[b.claim_code];return s&&s.label}).length;
  document.getElementById('prog').textContent=d+' / '+BLIND.length+' labeled'}
[].forEach.call(document.querySelectorAll('.claim'),function(c){var s=state._claims[c.dataset.key];
  if(s){if(s.label){var r=[].filter.call(c.querySelectorAll('input[type=radio]'),function(x){return x.value===s.label})[0];if(r)r.checked=true}
    c.querySelector('.evin').value=s.evidence||'';c.querySelector('.note').value=s.note||'';c.classList.toggle('done',!!s.label);check(c)}
  c.addEventListener('focusin',function(){setActive(c)});c.addEventListener('click',function(){setActive(c)});
  c.addEventListener('input',function(){sync(c)});c.addEventListener('change',function(){sync(c)})});
[].forEach.call(document.querySelectorAll('.eid'),function(b){b.addEventListener('click',function(){
  if(!active||!cases[cur].contains(active))return;var inp=active.querySelector('.evin'),v=norm(inp.value),id=b.dataset.id.toUpperCase();
  var i=v.indexOf(id);if(i<0)v.push(id);else v.splice(i,1);inp.value=v.join(' ');sync(active)})});
cases.forEach(function(s,i){s.querySelector('.prev').onclick=function(){show(i-1)};s.querySelector('.next').onclick=function(){show(i+1)}});
jump.onchange=function(){show(+jump.value)};
function q(v){v=String(v==null?'':v);return /[",\r\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v}
document.getElementById('dl').onclick=function(){
  var name=who.value.trim().toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_+|_+$/g,'');
  if(!name){who.classList.add('missing');who.focus();window.alert('Type your name at the top first.');return}
  var miss=BLIND.filter(function(b){var s=state._claims[b.claim_code];return !(s&&s.label)}).length;
  if(miss&&!window.confirm(miss+' claim(s) have no label yet. Download anyway?'))return;
  var cols=['labeler','case_code','claim_code','claim','human_label','human_evidence','human_note'],lines=[cols.join(',')];
  BLIND.forEach(function(b){var s=state._claims[b.claim_code]||{};
    var r={labeler:name,case_code:b.case_code,claim_code:b.claim_code,claim:b.claim,human_label:s.label||'',human_evidence:s.evidence||'',human_note:s.note||''};
    lines.push(cols.map(function(c){return q(r[c])}).join(','))});
  var a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\r\n')+'\r\n'],{type:'text/csv'}));
  a.download='labels_3b_'+name+'.csv';document.body.appendChild(a);a.click();setTimeout(function(){URL.revokeObjectURL(a.href);a.remove()},500)};
prog();show(0);
})();
</script></body></html>
"""


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default="results/critic_attribution/master_cases.csv")
    ap.add_argument("--claims", default="results/grounding_heldout/grounding_claims.csv")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="repo-relative or absolute")
    ap.add_argument("--out", default="results/grounding_3b")
    ap.add_argument("--expected", default=None, help=argparse.SUPPRESS)  # tests only
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        die(f"{out} exists and is not empty; pick a new --out")
    expected = EXPECTED
    if a.expected:
        expected = json.loads(Path(a.expected).read_text())
        print(f"NOTE: expected values from {a.expected} (test mode), not the held-out numbers")

    correct, system = load_master(Path(a.master))
    pub, judge = load_claims(Path(a.claims))
    manifest = load_manifest(repo_path(a.manifest))
    step0(correct, pub, expected)

    rng = random.Random(SEED)
    picks, strata, n_elig = sample(correct, system, pub, rng)

    # load runs, check them, build the page
    if not all(p["case_id"] in manifest for p in picks):
        die("a sampled case is not in the manifest")
    toks = forbidden_tokens(picks, manifest)
    sections, texts, map_rows, blind_rows, key_rows = [], {}, [], [], []
    for n, p in enumerate(picks, 1):
        paths = {c["run_path"] for c in p["claims"]}
        if len(paths) != 1:
            die(f"case {p['code']}: claims point to different run files")
        rp = resolve(paths.pop())
        try:
            run = json.loads(rp.read_text())
        except (OSError, json.JSONDecodeError) as e:
            die(f"cannot read the run JSON for case {p['code']} ({type(e).__name__})")
        if run.get("case_id") not in (None, p["case_id"]):
            die(f"case {p['code']}: run JSON case_id does not match the claims file")
        if not run.get("evidence_ledger"):
            die(f"case {p['code']}: run JSON has an empty evidence ledger")
        try:
            query = incident_query(manifest[p["case_id"]])
        except Exception as e:  # noqa: BLE001  (no traceback: it would print the case path)
            die(f"cannot rebuild the incident query for case {p['code']} ({type(e).__name__}); "
                "check that data/raw/rcaeval/ has metrics.parquet and inject_time.txt for every case")
        sections.append(render_case(n, len(picks), p, run, query))
        texts[f"case {p['code']} INPUT"] = query
        for item in run["evidence_ledger"]:
            texts[f"case {p['code']} item {item.get('id')}"] = (
                f"{item.get('id')} {item.get('source')} {json.dumps(item.get('args'), sort_keys=True, default=str)} "
                f"{content_text(item.get('content'))}")
        for c in p["claims"]:
            texts[f"claim {c['code']}"] = c["claim"]
            blind_rows.append({"case_code": p["code"], "claim_code": c["code"], "claim": c["claim"],
                               "human_label": "", "human_evidence": ""})
            map_rows.append({"case_code": p["code"], "claim_code": c["code"], "row": p["row"],
                             "case_id": p["case_id"], "system": system[p["case_id"]],
                             "ac1_correct": correct[(p["row"], p["case_id"])],
                             "cell_row": p["cell"][0], "cell_system": p["cell"][1],
                             "cell_ac1": "correct" if p["cell"][2] else "wrong", "fallback": p["fallback"],
                             "claim_id": c["claim_id"], "claim_type": c["claim_type"], "run_path": c["run_path"]})
            j = judge[(p["row"], p["case_id"], c["claim_id"])]
            key_rows.append({"claim_code": c["code"], **{f"judge_{k}": j[k] for k in KEY_COLS}})

    leaks, rowwords = leak_check(texts, toks)
    if leaks:
        die("blindness check failed, nothing written (values not printed, to keep you blind):\n  "
            + "\n  ".join(leaks[:20]))

    blind_json = json.dumps(blind_rows, ensure_ascii=False).replace("</", "<\\/")
    page_hash = hashlib.sha256(blind_json.encode()).hexdigest()[:12]
    head, tail = PAGE.split("__CASES__")
    page = (head + "\n".join(sections) + tail.replace("__BLIND__", blind_json).replace("__HASH__", page_hash))
    # final whole-page check (catches anything added by the template)
    leaks2, _ = leak_check({"page": page}, toks)
    if leaks2:
        die("blindness check on the finished page failed, nothing written")

    out.mkdir(parents=True, exist_ok=True)
    (out / OUT_PAGE).write_text(page, encoding="utf-8")
    write_csv(out / OUT_BLIND, blind_rows, ["case_code", "claim_code", "claim", "human_label", "human_evidence"])
    write_csv(out / OUT_MAP, map_rows, list(map_rows[0].keys()))
    write_csv(out / OUT_KEY, key_rows, list(key_rows[0].keys()))
    write_csv(out / OUT_STRATA, strata, list(strata[0].keys()))

    n_fb = sum(p["fallback"] for p in picks)
    rw = Counter(w for (_, w) in rowwords.elements())
    size = (out / OUT_PAGE).stat().st_size
    rec = [f"grounding_3b_prep.py (A13.3), seed {SEED}",
           f"script sha256 {sha256(Path(__file__).resolve())}",
           f"eligible (row, case) units: {n_elig}; cases sampled: {len(picks)}; claims: {len(blind_rows)}; "
           f"fallback cells: {n_fb}",
           f"row words found in evidence/claims (reported, not stopped): "
           + (", ".join(f"{k} x{v}" for k, v in sorted(rw.items())) or "none"),
           f"page size: {size:,} bytes"]
    for name in (OUT_PAGE, OUT_BLIND, OUT_MAP, OUT_KEY, OUT_STRATA):
        rec.append(f"sha256 {sha256(out / name)}  {name}")
    (out / OUT_RECORD).write_text("\n".join(rec) + "\n", encoding="utf-8")
    print("\n".join(rec))
    if n_fb:
        print(f"NOTE: {n_fb} cell(s) were filled by the fallback rule; see {OUT_STRATA}")
    if size > PAGE_WARN_BYTES:
        print(f"NOTE: the page is over 9.5 MB; Discord may refuse it, use Google Drive instead")
    print(f"\nwrote {out}/  (send only {OUT_PAGE}; do not open the DO_NOT_OPEN files)")


if __name__ == "__main__":
    main()
