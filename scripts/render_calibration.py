#!/usr/bin/env python3
"""Render calibration_blind.csv as one HTML page for blind human labeling (Grounding Metric Spec v1.1, 7.1).

Read-only: reads the blind CSV, the run JSONs it points to, the manifest and case headers (for the
incident query), and writes ONE new HTML file. It never edits its inputs and never reads
calibration_key.csv; it refuses a CSV that has any judge_* column.

The page shows what the judge saw and nothing more: per case, the incident query (INPUT) and every
evidence-ledger item (ID, source, args, content), then the claims to label. It does not show the
report explanation, citations, hypotheses, critic verdicts or ground truth.

In the page: pick Supported / Partially Supported / Unsupported per claim, type the evidence IDs
(click an evidence ID to add it to the active claim), optional note. Progress autosaves in the
browser. "Download CSV" saves calibration_labeled.csv (the blind columns, filled in, plus
human_note). Put that file next to the blind CSV for the agreement step.

Reuses incident_query / load_manifest / repo_path from judge_grounding.py (same folder), so the
INPUT text is the same rebuild that --check-query verified.

Example:
  python scripts/render_calibration.py --blind results/grounding_dev/calibration_blind.csv
  -> results/grounding_dev/calibration_review.html (open it in a browser)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_grounding import DEFAULT_MANIFEST, incident_query, load_manifest, repo_path  # noqa: E402

REQUIRED = ["row", "case_id", "claim_id", "claim_type", "claim", "run_path", "human_label", "human_evidence"]
LABELS = ["Supported", "Partially Supported", "Unsupported"]


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def resolve_run(run_path: str) -> Path:
    """run_path as the judge wrote it (relative to where it ran, normally the repo root)."""
    p = Path(run_path)
    if p.is_absolute() or p.exists():
        return p
    return repo_path(p)


def content_text(c) -> str:
    if isinstance(c, str):
        return c
    return json.dumps(c, indent=2, sort_keys=True, default=str)


def render_case(n: int, total: int, key: tuple, claims: list[dict], run: dict, query: str) -> str:
    row, case_id, run_path = key
    ledger = run.get("evidence_ledger") or []
    ids = ["INPUT"] + [str(i.get("id")) for i in ledger]
    ev = [f'<div class="ev" id="c{n}-INPUT"><div class="evh"><button class="eid" data-id="INPUT">INPUT</button>'
          f'<span class="src">case input (incident description)</span></div><pre>{esc(query)}</pre></div>']
    for item in ledger:
        iid = str(item.get("id"))
        args = json.dumps(item.get("args"), sort_keys=True, default=str)
        ev.append(f'<div class="ev" id="c{n}-{esc(iid)}"><div class="evh"><button class="eid" data-id="{esc(iid)}">{esc(iid)}</button>'
                  f'<span class="src">{esc(item.get("source"))}</span><code class="args">{esc(args)}</code></div>'
                  f'<pre>{esc(content_text(item.get("content")))}</pre></div>')
    cl = []
    for c in sorted(claims, key=lambda c: int(c["claim_id"])):
        k = f'{row}|{case_id}|{c["claim_id"]}'
        radios = "".join(
            f'<label class="lab"><input type="radio" name="{esc(k)}" value="{esc(lab)}"> {esc(lab)}</label>'
            for lab in LABELS)
        cl.append(f'<div class="claim" data-key="{esc(k)}" data-ids="{esc(" ".join(ids))}">'
                  f'<div class="ct"><span class="tag">{esc(c["claim_type"])}</span> claim {esc(c["claim_id"])}</div>'
                  f'<div class="ctext">{esc(c["claim"])}</div>'
                  f'<div class="ctrl">{radios}</div>'
                  f'<div class="ctrl"><input class="evin" placeholder="evidence IDs, e.g. E003 E007 (click an ID on the left)">'
                  f'<span class="warn"></span></div>'
                  f'<div class="ctrl"><input class="note" placeholder="note (optional)"></div></div>')
    return (f'<section class="case" id="case{n}" data-n="{n}">'
            f'<header><h2>Case {n}/{total}: {esc(case_id)}</h2><div class="meta">row: {esc(row)} &middot; '
            f'{len(ledger)} evidence items &middot; {len(claims)} claims &middot; <span class="rp">{esc(run_path)}</span></div></header>'
            f'<div class="cols"><div class="evcol"><h3>Evidence</h3>{"".join(ev)}</div>'
            f'<div class="clcol"><h3>Claims to label</h3>{"".join(cl)}'
            f'<div class="nav"><button class="prev">&larr; previous case</button><button class="next">next case &rarr;</button></div>'
            f'</div></div></section>')


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Grounding calibration review</title>
<style>
:root{--bg:#f6f6f3;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#ddddd6;--acc:#2f5d8a;--ok:#2e7d4f;--bad:#b23b3b}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}
.top{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.top h1{font-size:15px;margin:0 8px 0 0}.top select,.top button{font:inherit;padding:4px 8px}
#prog{color:var(--mute)}#dl{background:var(--acc);color:#fff;border:0;border-radius:4px;padding:5px 12px;cursor:pointer}
details.rub{padding:0 16px;margin:8px 0}details.rub div{background:var(--card);border:1px solid var(--line);padding:8px 12px;border-radius:4px}
.case{display:none;padding:12px 16px}.case.on{display:block}.case h2{font-size:16px;margin:0}.meta{color:var(--mute);font-size:12px}.rp{font-family:ui-monospace,monospace}
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
.ct{font-size:12px;color:var(--mute)}.tag{background:#eee;border-radius:3px;padding:0 5px;color:var(--ink)}
.ctext{font-size:15px;margin:4px 0 8px}.ctrl{margin:4px 0}.lab{margin-right:10px;white-space:nowrap;cursor:pointer}
.evin,.note{width:100%;font:inherit;padding:4px 6px;border:1px solid var(--line);border-radius:3px}.warn{color:var(--bad);font-size:12px}
.nav{display:flex;justify-content:space-between;margin-top:8px}.nav button{font:inherit;padding:4px 10px}
</style></head><body>
<div class="top"><h1>Grounding calibration (blind)</h1>
<select id="jump"></select><span id="prog"></span>
<button id="dl">Download CSV</button><span style="color:var(--mute);font-size:12px">saves __OUTNAME__ &middot; progress autosaves in this browser</span></div>
<details class="rub"><summary>Labeling rules (same definitions the judge uses)</summary><div>
Judge the claim against the evidence on this page only (closed set: no outside knowledge, plausibility does not count).<br>
<b>Supported</b>: the evidence directly states what the claim asserts; it follows from the evidence alone.<br>
<b>Partially Supported</b>: the evidence has related information, but the claim needs inference, extrapolation, or domain knowledge not in the evidence.<br>
<b>Unsupported</b>: the evidence contradicts the claim (different number, service, metric, or direction), or nothing in the evidence is about it.<br>
Compare numbers, service names, metric names and times exactly; a value counts only for the service it is listed under.
Root-cause / fault-category / indicator claims: Supported if the evidence clearly singles out what the claim names; Unsupported if it clearly singles out something else or the named metric does not appear; Partially only if the evidence is mixed.
A generic caveat in a tool output (such as "not causal proof") does not by itself lower the label.<br>
Evidence IDs: the items your label rests on (INPUT for the case input). Leave empty for Unsupported-because-absent.
</div></details>
__CASES__
<script id="blind" type="application/json">__BLIND__</script>
<script>
(function(){
var BLIND=JSON.parse(document.getElementById('blind').textContent), COLS=__COLS__, STORE='calib:__HASH__', OUT='__OUTNAME__';
var state={}; try{state=JSON.parse(localStorage.getItem(STORE)||'{}')}catch(e){state={}}
function save(){try{localStorage.setItem(STORE,JSON.stringify(state))}catch(e){}}
var cases=[].slice.call(document.querySelectorAll('.case')), jump=document.getElementById('jump'), active=null, cur=0;
cases.forEach(function(s,i){var o=document.createElement('option');o.value=i;o.textContent=s.querySelector('h2').textContent;jump.appendChild(o)});
function show(i){cur=Math.max(0,Math.min(cases.length-1,i));cases.forEach(function(s,j){s.classList.toggle('on',j===cur)});jump.value=cur;
  var f=cases[cur].querySelector('.claim');setActive(f);window.scrollTo(0,0);cases[cur].querySelector('.evcol').scrollTop=0}
function setActive(c){if(active)active.classList.remove('active');active=c;if(c)c.classList.add('active')}
function norm(s){return s.toUpperCase().split(/[\\s,;\\/]+/).filter(Boolean)}
function check(c){var ok=c.dataset.ids.split(' '),v=norm(c.querySelector('.evin').value),bad=v.filter(function(x){return ok.indexOf(x)<0});
  c.querySelector('.warn').textContent=bad.length?'not in this case: '+bad.join(' '):''}
function sync(c){var k=c.dataset.key,r=c.querySelector('input[type=radio]:checked');
  state[k]={label:r?r.value:'',evidence:norm(c.querySelector('.evin').value).join(' '),note:c.querySelector('.note').value};
  c.classList.toggle('done',!!state[k].label);check(c);save();prog()}
function prog(){var n=BLIND.length,d=BLIND.filter(function(b){var s=state[b.row+'|'+b.case_id+'|'+b.claim_id];return s&&s.label}).length;
  document.getElementById('prog').textContent=d+' / '+n+' labeled'}
[].forEach.call(document.querySelectorAll('.claim'),function(c){var s=state[c.dataset.key];
  if(s){if(s.label){var r=[].filter.call(c.querySelectorAll('input[type=radio]'),function(x){return x.value===s.label})[0];if(r)r.checked=true}
    c.querySelector('.evin').value=s.evidence||'';c.querySelector('.note').value=s.note||'';c.classList.toggle('done',!!s.label);check(c)}
  c.addEventListener('focusin',function(){setActive(c)});c.addEventListener('click',function(){setActive(c)});
  c.addEventListener('input',function(){sync(c)});c.addEventListener('change',function(){sync(c)})});
[].forEach.call(document.querySelectorAll('.eid'),function(b){b.addEventListener('click',function(){
  if(!active||!cases[cur].contains(active))return;var inp=active.querySelector('.evin'),v=norm(inp.value),id=b.dataset.id.toUpperCase();
  var i=v.indexOf(id);if(i<0)v.push(id);else v.splice(i,1);inp.value=v.join(' ');sync(active)})});
cases.forEach(function(s,i){s.querySelector('.prev').onclick=function(){show(i-1)};s.querySelector('.next').onclick=function(){show(i+1)}});
jump.onchange=function(){show(+jump.value)};
function q(v){v=String(v==null?'':v);return /[",\\r\\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v}
document.getElementById('dl').onclick=function(){
  var miss=BLIND.filter(function(b){var s=state[b.row+'|'+b.case_id+'|'+b.claim_id];return !(s&&s.label)}).length;
  if(miss&&!window.confirm(miss+' claim(s) have no label yet. Download anyway?'))return;
  var cols=COLS.concat(COLS.indexOf('human_note')<0?['human_note']:[]),lines=[cols.map(q).join(',')];
  BLIND.forEach(function(b){var s=state[b.row+'|'+b.case_id+'|'+b.claim_id]||{},r={};for(var k in b)r[k]=b[k];
    r.human_label=s.label||'';r.human_evidence=s.evidence||'';r.human_note=s.note||'';lines.push(cols.map(function(c){return q(r[c])}).join(','))});
  var a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\r\\n')+'\\r\\n'],{type:'text/csv'}));a.download=OUT;
  document.body.appendChild(a);a.click();setTimeout(function(){URL.revokeObjectURL(a.href);a.remove()},500)};
prog();show(0);
})();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blind", default="results/grounding/calibration_blind.csv", help="blind CSV from --export-calibration")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="repo-relative or absolute")
    ap.add_argument("--html", help="output page (default: calibration_review.html next to --blind)")
    ap.add_argument("--csv-name", default="calibration_labeled.csv", help="file name the page downloads")
    args = ap.parse_args()

    blind_path = Path(args.blind)
    raw = blind_path.read_bytes()
    with open(blind_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    leak = [c for c in cols if c.startswith("judge_")]
    if leak:
        sys.exit(f"{blind_path} has judge columns {leak}: use calibration_blind.csv, not the key")
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        sys.exit(f"{blind_path} is missing columns {missing}")
    if not rows:
        sys.exit(f"{blind_path} has no claims")
    if any((r.get("human_label") or "").strip() for r in rows):
        print("note: some human_label cells are already filled; the page starts empty and the download is a new file")

    manifest = load_manifest(repo_path(args.manifest))
    groups: OrderedDict[tuple, list[dict]] = OrderedDict()
    for r in rows:
        groups.setdefault((r["row"], r["case_id"], r["run_path"]), []).append(r)

    sections = []
    for n, (key, claims) in enumerate(groups.items(), 1):
        row, case_id, run_path = key
        p = resolve_run(run_path)
        try:
            run = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            sys.exit(f"cannot read run JSON for {row}/{case_id}: {p} ({e})")
        if case_id not in manifest:
            sys.exit(f"{case_id} not in manifest {args.manifest}")
        sections.append(render_case(n, len(groups), key, claims, run, incident_query(manifest[case_id])))

    out = Path(args.html) if args.html else blind_path.with_name("calibration_review.html")
    blind_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    head, tail = PAGE.split("__CASES__")
    tail_a, tail_b = tail.split("__BLIND__")
    fill = lambda s: (s.replace("__COLS__", json.dumps(cols))
                       .replace("__HASH__", hashlib.sha256(raw).hexdigest()[:12])
                       .replace("__OUTNAME__", esc(args.csv_name)))
    page = fill(head) + "\n".join(sections) + fill(tail_a) + blind_json + fill(tail_b)
    out.write_text(page, encoding="utf-8")
    print(f"{len(rows)} claims, {len(groups)} cases -> {out}")


if __name__ == "__main__":
    main()
