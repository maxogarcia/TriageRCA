#!/usr/bin/env python3
"""Critic eval on the held-out runs: how good is the critic's accept/reject decision?

Read-only. Reads run JSONs (<runs>/<row>/*/<case>.json, same layout score_ablation.py uses) and
writes CSVs + PNG charts to --out. Never edits inputs. No LLM calls.

Ground truth per round: the round's claimed root (hypotheses[i].ranked_services[0]) vs the true
root service parsed from the case id (re2ss_orders_cpu_1 -> orders, re2tt_ts-order-service_cpu_3 ->
ts-order-service). Match = exact after lowercasing and removing spaces (same rule as the AC@1 scorer).

Two views:
  A. Per-round decisions (critic as a classifier). Positive class = "claimed root is WRONG" (the
     critic should push back). Push back = reject OR need_more_evidence (both send the Executor back).
       TP caught wrong      wrong root, pushed back
       FN false accept      wrong root, accepted
       FP false reject      correct root, pushed back
       TN correct accept    correct root, accepted
     Reported for round 1 only (one decision per case, independent) and all rounds (rounds within a
     case are not independent, so CIs are a case-level bootstrap).
  B. Per-case outcome (round-1 answer vs final answer, final = report.ranked_services[0] = AC@1):
       kept_correct        r1 correct, accepted in r1
       recovered           r1 correct, pushed back, final still correct
       broke_correct       r1 correct, final wrong                       (critic hurt)
       caught_fixed        r1 wrong, final correct                       (critic helped)
       caught_not_fixed    r1 wrong, pushed back, final wrong
       missed              r1 wrong, accepted in r1                      (false accept)
       no_run              JSON missing / no report / no critic rounds   (counted wrong, not hidden)

Outputs (--out):
  critic_rounds.csv        one line per (row, case, round)
  critic_cases.csv         one line per (row, case) with outcome
  critic_metrics.csv       metrics per row x scope (round1 / all) x system (all / SS / TT), with CIs
  critic_outcomes.csv      outcome counts per row
  fig1_confusion.png, fig2_metrics.png, fig3_outcomes.png, fig4_by_system.png

Example (from repo root):
  python scripts/eval_critic.py --runs runs/heldout --rows hybrid llm_only \
      --out results/critic_eval --expect hybrid=48 llm_only=42
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PUSH_BACK = {"reject", "need_more_evidence"}
ACCEPT = {"accept"}
OUTCOMES = ["kept_correct", "recovered", "broke_correct", "caught_fixed",
            "caught_not_fixed", "missed", "no_run"]
OUTCOME_LABEL = {
    "kept_correct": "Correct, accepted round 1",
    "recovered": "Correct, pushed back, still correct",
    "broke_correct": "Correct -> wrong (critic hurt)",
    "caught_fixed": "Wrong -> fixed (critic helped)",
    "caught_not_fixed": "Wrong, pushed back, not fixed",
    "missed": "Wrong, accepted round 1 (missed)",
    "no_run": "No run / no report",
}
SYSTEM_NAME = {"ss": "SS", "tt": "TT", "ob": "OB"}


def norm(s) -> str:
    return "".join(str(s).split()).lower()


def true_root(case_id: str) -> tuple[str, str]:
    """re2ss_orders_cpu_1 -> ('SS', 'orders'). Service may contain dashes, not underscores."""
    head, _, rest = case_id.partition("_")
    parts = rest.rsplit("_", 2)
    if not head.startswith("re2") or len(parts) != 3 or not parts[0]:
        raise ValueError(f"cannot parse case id {case_id!r} (expected re2<sys>_<service>_<fault>_<n>)")
    return SYSTEM_NAME.get(head[3:], head[3:].upper()), parts[0]


def claimed_root(h) -> str | None:
    """Root service of one hypothesis. Hypotheses are report-shaped (ranked_services)."""
    if isinstance(h, dict):
        rs = h.get("ranked_services")
        if isinstance(rs, list):
            return str(rs[0]) if rs else None
        for k in ("root_cause_service", "root_service", "service"):
            if h.get(k):
                return str(h[k])
        raise KeyError(f"hypothesis has no ranked_services; keys = {sorted(h)}")
    if isinstance(h, str):
        return h
    raise TypeError(f"unexpected hypothesis type {type(h).__name__}")


def decision_of(v) -> str:
    d = v.get("decision") if isinstance(v, dict) else v
    return norm(d)


def load_row(runs: Path, row: str, case_ids: list[str] | None):
    folder = runs / row
    if not folder.is_dir():
        sys.exit(f"ERROR: no folder {folder}. Check --runs / --rows.")
    files = {p.stem: p for p in folder.glob("*/*.json")}
    if not files:
        sys.exit(f"ERROR: no run JSONs under {folder}/*/")
    ids = case_ids if case_ids is not None else sorted(files)
    rounds, cases, warnings = [], [], []
    for cid in ids:
        system, truth = true_root(cid)
        base = {"row": row, "case_id": cid, "system": system, "true_root": truth}
        p = files.get(cid)
        rec = json.loads(p.read_text()) if p else None
        report = (rec or {}).get("report") or {}
        hyps = (rec or {}).get("hypotheses") or []
        verds = (rec or {}).get("critic_verdicts") or []
        final = (report.get("ranked_services") or [None])[0] if report else None
        final_ok = final is not None and norm(final) == norm(truth)
        if not rec or not report or not verds:
            why = "missing json" if not rec else ("no report" if not report else "no critic verdicts")
            warnings.append(f"{row}/{cid}: {why} -> no_run")
            cases.append({**base, "n_rounds": len(verds), "r1_root": "", "r1_correct": "",
                          "r1_decision": "", "final_root": final or "", "final_correct": int(final_ok),
                          "last_decision": "", "outcome": "no_run"})
            continue
        if len(hyps) != len(verds):
            warnings.append(f"{row}/{cid}: {len(hyps)} hypotheses vs {len(verds)} verdicts; pairing first "
                            f"{min(len(hyps), len(verds))}")
        n = min(len(hyps), len(verds))
        for i in range(n):
            root = claimed_root(hyps[i])
            dec = decision_of(verds[i])
            if dec not in PUSH_BACK | ACCEPT:
                sys.exit(f"ERROR: {row}/{cid} round {i + 1}: unknown decision {dec!r}")
            ok = root is not None and norm(root) == norm(truth)
            pushed = dec in PUSH_BACK
            cell = ("TN_correct_accept" if ok and not pushed else "FP_false_reject" if ok
                    else "TP_caught_wrong" if pushed else "FN_false_accept")
            rounds.append({**base, "round": i + 1, "claimed_root": root or "", "claimed_correct": int(ok),
                           "decision": dec, "cell": cell})
        mine = [r for r in rounds if r["row"] == row and r["case_id"] == cid]
        if not mine:
            warnings.append(f"{row}/{cid}: 0 pairable rounds -> no_run")
            cases.append({**base, "n_rounds": 0, "r1_root": "", "r1_correct": "", "r1_decision": "",
                          "final_root": final or "", "final_correct": int(final_ok), "last_decision": "",
                          "outcome": "no_run"})
            continue
        r1 = mine[0]
        if r1["claimed_correct"]:
            out = ("kept_correct" if r1["decision"] in ACCEPT else
                   "recovered" if final_ok else "broke_correct")
        else:
            out = ("missed" if r1["decision"] in ACCEPT else
                   "caught_fixed" if final_ok else "caught_not_fixed")
        if r1["decision"] in ACCEPT and norm(final or "") != norm(r1["claimed_root"]):
            warnings.append(f"{row}/{cid}: accepted in round 1 but final root {final!r} != {r1['claimed_root']!r}")
        if mine[-1]["claimed_root"] and norm(final or "") != norm(mine[-1]["claimed_root"]):
            warnings.append(f"{row}/{cid}: final report root {final!r} != last hypothesis "
                            f"{mine[-1]['claimed_root']!r}")
        cases.append({**base, "n_rounds": len(mine), "r1_root": r1["claimed_root"],
                      "r1_correct": r1["claimed_correct"], "r1_decision": r1["decision"],
                      "final_root": final or "", "final_correct": int(final_ok),
                      "last_decision": mine[-1]["decision"], "outcome": out})
    return rounds, cases, warnings, ids


# ---------------------------------------------------------------- metrics

def counts(rs):
    c = Counter(r["cell"] for r in rs)
    return c["TP_caught_wrong"], c["FN_false_accept"], c["FP_false_reject"], c["TN_correct_accept"]


def rates(tp, fn, fp, tn):
    def div(a, b):
        return a / b if b else float("nan")
    catch = div(tp, tp + fn)          # recall on wrong roots
    keep = div(tn, tn + fp)           # specificity: correct roots accepted
    return {
        "accuracy": div(tp + tn, tp + fn + fp + tn),
        "balanced_accuracy": (catch + keep) / 2,
        "catch_rate_wrong": catch,
        "accept_rate_correct": keep,
        "pushback_precision": div(tp, tp + fp),     # when it pushes back, was the root really wrong?
        "accept_precision": div(tn, tn + fn),       # when it accepts, was the root really correct?
        "false_accept_rate": div(fn, tp + fn),
        "false_reject_rate": div(fp, tn + fp),
    }


METRIC_KEYS = list(rates(1, 1, 1, 1))


def bootstrap(rs, n_boot, seed):
    """95% percentile CI, resampling CASES (rounds within a case stay together)."""
    import numpy as np
    by_case = {}
    for r in rs:
        by_case.setdefault(r["case_id"], []).append(r)
    groups = [counts(g) for g in by_case.values()]
    if not groups:
        return {k: (float("nan"), float("nan")) for k in METRIC_KEYS}
    arr = np.array(groups)  # cases x 4
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    sums = arr[idx].sum(axis=1)
    vals = {k: [] for k in METRIC_KEYS}
    with np.errstate(invalid="ignore", divide="ignore"):
        for tp, fn, fp, tn in sums:
            for k, v in rates(tp, fn, fp, tn).items():
                vals[k].append(v)
    out = {}
    for k, v in vals.items():
        v = np.array(v, dtype=float)
        v = v[~np.isnan(v)]
        out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (float("nan"),) * 2
    return out


def metric_rows(rounds, rows, n_boot, seed):
    out = []
    for row in rows:
        for scope in ("round1", "all"):
            for system in ("all", "SS", "TT"):
                rs = [r for r in rounds if r["row"] == row
                      and (scope == "all" or r["round"] == 1)
                      and (system == "all" or r["system"] == system)]
                if not rs:
                    continue
                tp, fn, fp, tn = counts(rs)
                m = rates(tp, fn, fp, tn)
                ci = bootstrap(rs, n_boot, seed)
                line = {"row": row, "scope": scope, "system": system,
                        "n_cases": len({r["case_id"] for r in rs}), "n_decisions": len(rs),
                        "TP_caught_wrong": tp, "FN_false_accept": fn,
                        "FP_false_reject": fp, "TN_correct_accept": tn}
                for k in METRIC_KEYS:
                    line[k] = round(m[k], 4)
                    line[k + "_lo"] = round(ci[k][0], 4)
                    line[k + "_hi"] = round(ci[k][1], 4)
                out.append(line)
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- charts

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 (validated, fixed order)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
    })


def fig_confusion(metrics, rows, out: Path, scope: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, len(rows), figsize=(4.6 * len(rows), 4.3), squeeze=False)
    for ax, row in zip(axes[0], rows):
        m = next(x for x in metrics if x["row"] == row and x["scope"] == scope and x["system"] == "all")
        grid = [[m["TN_correct_accept"], m["FP_false_reject"]],
                [m["FN_false_accept"], m["TP_caught_wrong"]]]
        names = [["correct accept", "false reject"], ["false accept", "caught wrong"]]
        vmax = max(max(r) for r in grid) or 1
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax)
        tot = sum(map(sum, grid))
        for i in range(2):
            for j in range(2):
                v = grid[i][j]
                dark = v / vmax > 0.55
                ax.text(j, i - 0.08, f"{v}", ha="center", va="center", fontsize=18, fontweight="bold",
                        color="white" if dark else INK)
                ax.text(j, i + 0.22, f"{names[i][j]}\n{v / tot:.0%} of decisions" if tot else names[i][j],
                        ha="center", va="center", fontsize=8.5, color="white" if dark else INK2)
        ax.set_xticks([0, 1], ["accept", "push back\n(reject / need more)"])
        ax.set_yticks([0, 1], ["claimed root\ncorrect", "claimed root\nwrong"])
        ax.set_xlabel("critic decision")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f"{row}  (n={tot} decisions, {m['n_cases']} cases)\n"
                     f"accuracy {m['accuracy']:.0%}, balanced {m['balanced_accuracy']:.0%}",
                     loc="left", fontsize=10)
    label = "round 1 only" if scope == "round1" else "all rounds"
    fig.suptitle(f"Critic decisions vs ground truth ({label})", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_metrics(metrics, rows, out: Path):
    import matplotlib.pyplot as plt
    keys = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced\naccuracy"),
            ("catch_rate_wrong", "Catch rate\n(wrong roots\npushed back)"),
            ("accept_rate_correct", "Accept rate\n(correct roots\naccepted)"),
            ("pushback_precision", "Push-back\nprecision"), ("accept_precision", "Accept\nprecision")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    w = 0.8 / len(rows)
    for ax, scope, title in zip(axes, ("round1", "all"),
                                ("Round 1 only (1 decision per case)", "All rounds (CI: case bootstrap)")):
        for k, row in enumerate(rows):
            m = next(x for x in metrics if x["row"] == row and x["scope"] == scope and x["system"] == "all")
            xs = [i + (k - (len(rows) - 1) / 2) * w for i in range(len(keys))]
            ys = [m[key] for key, _ in keys]
            lo = [max(0, y - m[key + "_lo"]) for y, (key, _) in zip(ys, keys)]
            hi = [max(0, m[key + "_hi"] - y) for y, (key, _) in zip(ys, keys)]
            ax.bar(xs, ys, width=w - 0.04, color=SERIES[k], label=row, zorder=2)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
            for x, y in zip(xs, ys):
                if y == y:
                    ax.text(x, 0.02, f"{y:.0%}", ha="center", va="bottom", fontsize=7, rotation=90,
                            color=INK, fontweight="bold", zorder=4)
        ax.set_xticks(range(len(keys)), [lbl for _, lbl in keys], fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", color=GRID, zorder=0)
        ax.set_title(title, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=len(rows), bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("How good is the critic's accept / push-back decision? (95% CI)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_outcomes(cases, rows, out: Path):
    import matplotlib.pyplot as plt
    present = [o for o in OUTCOMES if o != "no_run" or any(c["outcome"] == "no_run" for c in cases)]
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(present) * len(rows) + 1.4))
    h = 0.8 / len(rows)
    for k, row in enumerate(rows):
        cnt = Counter(c["outcome"] for c in cases if c["row"] == row)
        n = sum(cnt.values())
        ys = [i + (k - (len(rows) - 1) / 2) * h for i in range(len(present))]
        vals = [cnt[o] for o in present]
        ax.barh(ys, vals, height=h - 0.04, color=SERIES[k], label=f"{row} (n={n})", zorder=2)
        for y, v in zip(ys, vals):
            ax.text(v + 0.3, y, f"{v}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(present)), [OUTCOME_LABEL[o] for o in present])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.set_xlabel("cases")
    ax.legend(loc="lower right")
    ax.set_title("What the critic did to each case (round-1 answer vs final answer)", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_by_system(metrics, rows, out: Path):
    import matplotlib.pyplot as plt
    keys = [("catch_rate_wrong", "Catch rate (wrong roots pushed back)"),
            ("accept_rate_correct", "Accept rate (correct roots accepted)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    w = 0.8 / len(rows)
    for ax, (key, title) in zip(axes, keys):
        for k, row in enumerate(rows):
            xs, ys, lo, hi = [], [], [], []
            for i, system in enumerate(("SS", "TT")):
                m = next((x for x in metrics if x["row"] == row and x["scope"] == "all"
                          and x["system"] == system), None)
                if m is None or m[key] != m[key]:
                    continue
                xs.append(i + (k - (len(rows) - 1) / 2) * w)
                ys.append(m[key])
                lo.append(max(0, m[key] - m[key + "_lo"]))
                hi.append(max(0, m[key + "_hi"] - m[key]))
            ax.bar(xs, ys, width=w - 0.04, color=SERIES[k], label=row, zorder=2)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
            for x, y in zip(xs, ys):
                ax.text(x, 0.02, f"{y:.0%}", ha="center", va="bottom", fontsize=7.5, rotation=90,
                        color=INK, fontweight="bold", zorder=4)
        ax.set_xticks([0, 1], ["Sock Shop", "Train Ticket"])
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", color=GRID, zorder=0)
        ax.set_title(title, loc="left", fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=len(rows), bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("Critic by system, all rounds (95% CI, case bootstrap)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/heldout", help="ablation output dir (repo-relative)")
    ap.add_argument("--rows", nargs="+", default=["hybrid", "llm_only"])
    ap.add_argument("--out", required=True, help="new output folder for CSVs + PNGs")
    ap.add_argument("--expect", nargs="*", default=[],
                    help="row=N final top-1 correct counts to check, e.g. hybrid=48 llm_only=42")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty --out")
    a = ap.parse_args()
    if len(a.rows) > len(SERIES):
        sys.exit(f"ERROR: {len(a.rows)} rows but only {len(SERIES)} chart colours; pass at most {len(SERIES)} rows")

    runs, out = Path(a.runs), Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        sys.exit(f"ERROR: {out} is not empty; pick a new --out or pass --overwrite")

    all_rounds, all_cases, warns, case_sets = [], [], [], {}
    for row in a.rows:
        try:
            r, c, w, ids = load_row(runs, row, None)
        except (KeyError, ValueError, TypeError) as e:
            sys.exit(f"ERROR in row {row}: {e}. Nothing written.")
        all_rounds += r
        all_cases += c
        warns += w
        case_sets[row] = set(ids)
    sets = list(case_sets.values())
    if any(s != sets[0] for s in sets):
        for row, s in case_sets.items():
            print(f"  {row}: {len(s)} cases")
        sys.exit("ERROR: rows have different case lists; the comparison would not be paired")

    # sanity: final top-1 correct must match the AC@1 numbers already reported
    bad = 0
    for row in a.rows:
        got = sum(c["final_correct"] for c in all_cases if c["row"] == row)
        print(f"{row}: {len(case_sets[row])} cases, final top-1 correct {got}")
    for e in a.expect:
        row, _, n = e.partition("=")
        got = sum(c["final_correct"] for c in all_cases if c["row"] == row)
        if row not in case_sets or got != int(n):
            print(f"EXPECT FAILED: {row} final correct {got if row in case_sets else 'n/a'} != {n}")
            bad += 1
    if bad:
        sys.exit("ERROR: final-correct check failed; truth matching differs from the scorer. Nothing written.")

    metrics = metric_rows(all_rounds, a.rows, a.boot, a.seed)
    outcomes = []
    for row in a.rows:
        cnt = Counter(c["outcome"] for c in all_cases if c["row"] == row)
        outcomes.append({"row": row, **{o: cnt[o] for o in OUTCOMES}, "n_cases": sum(cnt.values())})

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "critic_rounds.csv", all_rounds)
    write_csv(out / "critic_cases.csv", all_cases)
    write_csv(out / "critic_metrics.csv", metrics)
    write_csv(out / "critic_outcomes.csv", outcomes)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig_confusion(metrics, a.rows, out / "fig1_confusion.png", "all")
    fig_metrics(metrics, a.rows, out / "fig2_metrics.png")
    fig_outcomes(all_cases, a.rows, out / "fig3_outcomes.png")
    fig_by_system(metrics, a.rows, out / "fig4_by_system.png")

    print("\nscope  row          n_dec  acc   bal   catch accept  FA  FR")
    for m in metrics:
        if m["system"] == "all":
            print(f"{m['scope']:<6} {m['row']:<12} {m['n_decisions']:>5}  {m['accuracy']:.2f}  "
                  f"{m['balanced_accuracy']:.2f}  {m['catch_rate_wrong']:.2f}  {m['accept_rate_correct']:.2f}"
                  f"   {m['FN_false_accept']:>2}  {m['FP_false_reject']:>2}")
    print("\noutcomes: " + "; ".join(f"{o['row']}: " + ", ".join(f"{k}={o[k]}" for k in OUTCOMES if o[k])
                                    for o in outcomes))
    if warns:
        print(f"\n{len(warns)} warning(s):")
        for w in warns[:30]:
            print("  " + w)
        if len(warns) > 30:
            print(f"  ... {len(warns) - 30} more")
    print(f"\nwrote {out}/ (4 CSVs, 4 PNGs)")


if __name__ == "__main__":
    main()
