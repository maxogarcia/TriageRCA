"""
Retrieval evaluation for TriageRCA RAG pipeline.

Tests how well the vector search "librarian" finds the correct runbook
given an incident description.  Two test-case sources:

  1. RCAEval RE1 dataset — folder names encode (service, fault_injection)
     which we map to expected runbook fault IDs.
  2. Handcrafted queries with known ground-truth fault IDs.

Metrics:
  - Hit@k (Recall@k)   — is the correct runbook anywhere in the top k?
  - MRR                 — mean reciprocal rank of the first correct hit
  - Precision@k         — fraction of the top k that are correct
  - Breakdown by fault type (Resource / Network / Code-level / …)

Usage:
    cd ~/cmpe258
    python -m src.eval.retrieval_eval          # full evaluation
    python -m src.eval.retrieval_eval --quick   # handcrafted queries only
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `from src.rag.retrieve import …` works
# when invoked as `python -m src.eval.retrieval_eval` from ~/cmpe258.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rag.retrieve import retrieve  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# 1.  GROUND-TRUTH TEST CASES
# ═══════════════════════════════════════════════════════════════════════════

K_VALUES = [1, 3, 5]


@dataclass
class TestCase:
    """One retrieval test case."""
    query: str
    expected_fault_ids: list[str]   # any of these counts as a hit
    fault_type: str                 # Resource | Network | Code-level | …
    source: str                     # "rcaeval" or "handcrafted"
    label: str = ""                 # short human-readable tag


# ---------------------------------------------------------------------------
# 1a.  RCAEval RE1 ground-truth mapping
#
# RE1 folder names are  {service}_{injection}  where injection is one of:
#   cpu, mem, disk, delay, loss
#
# We map each (system, service, injection) → expected fault IDs.
#   - Specific synthesized runbook IDs (system-specific)
#   - Generic "real" runbook IDs (apply to any system)
# ---------------------------------------------------------------------------

# RE1 injection → fault type label
_INJECTION_TO_FAULT_TYPE = {
    "cpu":   "Resource",
    "mem":   "Resource",
    "disk":  "Resource",
    "delay": "Network",
    "loss":  "Network",
}

# System-specific expected fault IDs for each (system, service, injection).
# The values list both the specific synthesized ID and the generic real ID.
_OB_GROUND_TRUTH: dict[tuple[str, str], list[str]] = {
    # ---------- Resource ----------
    ("adservice", "cpu"):              ["OB-RES-CPU-001", "RES-CPU-001"],
    ("adservice", "mem"):              ["RES-MEM-001"],
    ("adservice", "disk"):             ["RES-DISK-001"],
    ("cartservice", "cpu"):            ["OB-RES-CPU-001", "RES-CPU-001"],
    ("cartservice", "mem"):            ["OB-RES-MEM-002", "RES-MEM-001"],
    ("cartservice", "disk"):           ["RES-DISK-001"],
    ("checkoutservice", "cpu"):        ["OB-RES-CPU-001", "RES-CPU-001"],
    ("checkoutservice", "mem"):        ["RES-MEM-001"],
    ("checkoutservice", "disk"):       ["RES-DISK-001"],
    ("currencyservice", "cpu"):        ["OB-RES-CPU-001", "RES-CPU-001"],
    ("currencyservice", "mem"):        ["RES-MEM-001"],
    ("currencyservice", "disk"):       ["RES-DISK-001"],
    ("productcatalogservice", "cpu"):  ["OB-RES-CPU-001", "RES-CPU-001"],
    ("productcatalogservice", "mem"):  ["RES-MEM-001"],
    ("productcatalogservice", "disk"): ["OB-RES-DISK-001", "RES-DISK-001"],
    # ---------- Network ----------
    ("adservice", "delay"):            ["NET-DELAY-001"],
    ("adservice", "loss"):             ["NET-LOSS-001"],
    ("cartservice", "delay"):          ["NET-DELAY-001"],
    ("cartservice", "loss"):           ["OB-NET-LOSS-001", "NET-LOSS-001"],
    ("checkoutservice", "delay"):      ["NET-DELAY-001"],
    ("checkoutservice", "loss"):       ["OB-NET-LOSS-002", "NET-LOSS-001"],
    ("currencyservice", "delay"):      ["NET-DELAY-001"],
    ("currencyservice", "loss"):       ["NET-LOSS-001"],
    ("productcatalogservice", "delay"):["OB-NET-DELAY-001", "NET-DELAY-001"],
    ("productcatalogservice", "loss"): ["NET-LOSS-001"],
}

_SS_GROUND_TRUTH: dict[tuple[str, str], list[str]] = {
    # ---------- Resource ----------
    ("carts", "cpu"):      ["SS-RES-CPU-001", "RES-CPU-001"],
    ("carts", "mem"):      ["SS-RES-MEM-002", "RES-MEM-001"],
    ("carts", "disk"):     ["RES-DISK-001"],
    ("catalogue", "cpu"):  ["SS-RES-CPU-002", "RES-CPU-001"],
    ("catalogue", "mem"):  ["RES-MEM-001"],
    ("catalogue", "disk"): ["RES-DISK-001"],
    ("orders", "cpu"):     ["SS-RES-CPU-001", "RES-CPU-001"],
    ("orders", "mem"):     ["SS-RES-MEM-001", "RES-MEM-001"],
    ("orders", "disk"):    ["RES-DISK-001"],
    ("payment", "cpu"):    ["SS-RES-CPU-001", "RES-CPU-001"],
    ("payment", "mem"):    ["RES-MEM-001"],
    ("payment", "disk"):   ["RES-DISK-001"],
    ("user", "cpu"):       ["SS-RES-CPU-001", "RES-CPU-001"],
    ("user", "mem"):       ["RES-MEM-001"],
    ("user", "disk"):      ["RES-DISK-001"],
    # ---------- Network ----------
    ("carts", "delay"):    ["SS-NET-MONGO-002", "NET-DELAY-001"],
    ("carts", "loss"):     ["SS-NET-MONGO-002", "NET-LOSS-001"],
    ("catalogue", "delay"):["NET-DELAY-001"],
    ("catalogue", "loss"): ["NET-LOSS-001"],
    ("orders", "delay"):   ["SS-NET-MONGO-001", "NET-DELAY-001"],
    ("orders", "loss"):    ["NET-LOSS-001"],
    ("payment", "delay"):  ["NET-DELAY-001"],
    ("payment", "loss"):   ["NET-LOSS-001"],
    ("user", "delay"):     ["NET-DELAY-001"],
    ("user", "loss"):      ["NET-LOSS-001"],
}

_TT_GROUND_TRUTH: dict[tuple[str, str], list[str]] = {
    # ---------- Resource ----------
    ("ts-auth-service", "cpu"):    ["TT-RES-F5-01", "TT-RES-F5-02", "RES-CPU-001"],
    ("ts-auth-service", "mem"):    ["RES-MEM-001"],
    ("ts-auth-service", "disk"):   ["RES-DISK-001"],
    ("ts-order-service", "cpu"):   ["TT-RES-F5-01", "TT-RES-F5-02", "RES-CPU-001"],
    ("ts-order-service", "mem"):   ["TT-RES-F3-01", "TT-RES-F3-02", "RES-MEM-001"],
    ("ts-order-service", "disk"):  ["RES-DISK-001"],
    ("ts-route-service", "cpu"):   ["TT-RES-F5-01", "TT-RES-F5-02", "RES-CPU-001"],
    ("ts-route-service", "mem"):   ["RES-MEM-001"],
    ("ts-route-service", "disk"):  ["RES-DISK-001"],
    ("ts-train-service", "cpu"):   ["TT-RES-F5-01", "TT-RES-F5-02", "RES-CPU-001"],
    ("ts-train-service", "mem"):   ["RES-MEM-001"],
    ("ts-train-service", "disk"):  ["RES-DISK-001"],
    ("ts-travel-service", "cpu"):  ["TT-RES-F5-01", "TT-RES-F5-02", "RES-CPU-001"],
    ("ts-travel-service", "mem"):  ["RES-MEM-001"],
    ("ts-travel-service", "disk"): ["RES-DISK-001"],
    # ---------- Network ----------
    ("ts-auth-service", "delay"):    ["NET-DELAY-001"],
    ("ts-auth-service", "loss"):     ["NET-LOSS-001"],
    ("ts-order-service", "delay"):   ["TT-NET-F7-01", "NET-DELAY-001"],
    ("ts-order-service", "loss"):    ["NET-LOSS-001"],
    ("ts-route-service", "delay"):   ["NET-DELAY-001"],
    ("ts-route-service", "loss"):    ["NET-LOSS-001"],
    ("ts-train-service", "delay"):   ["NET-DELAY-001"],
    ("ts-train-service", "loss"):    ["NET-LOSS-001"],
    ("ts-travel-service", "delay"):  ["TT-NET-F7-01", "NET-DELAY-001"],
    ("ts-travel-service", "loss"):   ["NET-LOSS-001"],
}

# Which system code maps to which ground-truth dict
_SYSTEM_MAP = {
    "RE1-OB": ("Online Boutique", _OB_GROUND_TRUTH),
    "RE1-SS": ("Sock Shop", _SS_GROUND_TRUTH),
    "RE1-TT": ("Train Ticket", _TT_GROUND_TRUTH),
}


def _build_rcaeval_query(system_name: str, service: str, injection: str) -> str:
    """
    Synthesize a natural-language incident query from the RE1 case metadata.
    This simulates what an on-call SRE would type.
    """
    injection_descriptions = {
        "cpu": f"{service} is consuming abnormally high CPU, response times degraded, possible runaway computation",
        "mem": f"{service} memory usage steadily increasing, pod at risk of OOMKill, potential memory leak",
        "disk": f"{service} disk I/O extremely high, read/write latency spiking, storage may be full",
        "delay": f"High latency observed on requests to {service}, upstream services timing out, possible network delay",
        "loss": f"Intermittent failures communicating with {service}, TCP retransmissions detected, possible packet loss",
    }
    desc = injection_descriptions.get(injection, f"{service} experiencing {injection} issues")
    return f"[{system_name}] {desc}"


def build_rcaeval_cases(re1_dir: str = "data/RE1") -> list[TestCase]:
    """
    Scan the RE1 directory structure and generate one TestCase per
    (system, service, injection) combination.
    """
    re1_root = Path(re1_dir)
    cases: list[TestCase] = []

    for system_code, (system_name, gt_dict) in _SYSTEM_MAP.items():
        system_dir = re1_root / system_code
        if not system_dir.is_dir():
            print(f"  Warning: {system_dir} not found, skipping")
            continue

        for case_dir in sorted(system_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            parts = case_dir.name.rsplit("_", 1)
            if len(parts) != 2:
                continue
            service, injection = parts

            fault_type = _INJECTION_TO_FAULT_TYPE.get(injection, "Unknown")
            expected = gt_dict.get((service, injection), [])

            if not expected:
                # Fall back to generic IDs if no specific mapping
                generic_map = {
                    "cpu": ["RES-CPU-001"],
                    "mem": ["RES-MEM-001"],
                    "disk": ["RES-DISK-001"],
                    "delay": ["NET-DELAY-001"],
                    "loss": ["NET-LOSS-001"],
                }
                expected = generic_map.get(injection, [])

            query = _build_rcaeval_query(system_name, service, injection)
            cases.append(TestCase(
                query=query,
                expected_fault_ids=expected,
                fault_type=fault_type,
                source="rcaeval",
                label=f"{system_code}/{case_dir.name}",
            ))

    return cases


# ---------------------------------------------------------------------------
# 1b.  Handcrafted queries with known ground truth
# ---------------------------------------------------------------------------

HANDCRAFTED_CASES: list[TestCase] = [
    # ── Resource faults ──
    TestCase(
        query="Frontend service in Online Boutique pegged at 100% CPU, all downstream RPCs timing out",
        expected_fault_ids=["OB-RES-CPU-001", "RES-CPU-001"],
        fault_type="Resource",
        source="handcrafted",
        label="OB frontend CPU hog",
    ),
    TestCase(
        query="recommendationservice memory keeps climbing, eventually OOMKilled by kubelet",
        expected_fault_ids=["OB-RES-MEM-001", "RES-MEM-001"],
        fault_type="Resource",
        source="handcrafted",
        label="OB recommendation memleak",
    ),
    TestCase(
        query="productcatalogservice latency spiked after disk utilization hit 95%",
        expected_fault_ids=["OB-RES-DISK-001", "RES-DISK-001"],
        fault_type="Resource",
        source="handcrafted",
        label="OB catalog disk stress",
    ),
    TestCase(
        query="checkoutservice ran out of sockets, CLOSE_WAIT connections piling up",
        expected_fault_ids=["OB-RES-SOCK-001", "RES-SOCK-001"],
        fault_type="Resource",
        source="handcrafted",
        label="OB checkout socket exhaustion",
    ),
    TestCase(
        query="orders service in Sock Shop using 100% CPU, all order submissions failing",
        expected_fault_ids=["SS-RES-CPU-001", "RES-CPU-001"],
        fault_type="Resource",
        source="handcrafted",
        label="SS orders CPU hog",
    ),
    TestCase(
        query="carts service memory steadily increasing in Sock Shop, pod restarting",
        expected_fault_ids=["SS-RES-MEM-002", "RES-MEM-001"],
        fault_type="Resource",
        source="handcrafted",
        label="SS carts memleak",
    ),
    TestCase(
        query="ts-order-service JVM heap exceeds container memory limit causing OOMKill",
        expected_fault_ids=["TT-RES-F3-01", "TT-RES-F3-02", "RES-MEM-001"],
        fault_type="Resource",
        source="handcrafted",
        label="TT order JVM heap overflow",
    ),
    TestCase(
        query="basic-info service thread pool full, all downstream services timing out in Train Ticket",
        expected_fault_ids=["TT-RES-F5-01", "TT-RES-F5-02", "RES-SOCK-001"],
        fault_type="Resource",
        source="handcrafted",
        label="TT thread pool exhaustion",
    ),

    # ── Network faults ──
    TestCase(
        query="huge latency between productcatalogservice and recommendationservice in Online Boutique",
        expected_fault_ids=["OB-NET-DELAY-001", "NET-DELAY-001"],
        fault_type="Network",
        source="handcrafted",
        label="OB catalog-recommendation delay",
    ),
    TestCase(
        query="packet loss between frontend and cartservice, users seeing intermittent empty carts",
        expected_fault_ids=["OB-NET-LOSS-001", "NET-LOSS-001"],
        fault_type="Network",
        source="handcrafted",
        label="OB frontend-cart packet loss",
    ),
    TestCase(
        query="checkout fan-out to payment, shipping, email all dropping packets in Online Boutique",
        expected_fault_ids=["OB-NET-LOSS-002", "NET-LOSS-001"],
        fault_type="Network",
        source="handcrafted",
        label="OB checkout fanout loss",
    ),
    TestCase(
        query="RabbitMQ AMQP connection refused, shipping queue stuck in Sock Shop",
        expected_fault_ids=["SS-NET-AMQP-001"],
        fault_type="Network",
        source="handcrafted",
        label="SS RabbitMQ failure",
    ),
    TestCase(
        query="orders service can't reach MongoDB, connection timeouts in Sock Shop",
        expected_fault_ids=["SS-NET-MONGO-001", "NET-DELAY-001"],
        fault_type="Network",
        source="handcrafted",
        label="SS orders-mongo timeout",
    ),
    TestCase(
        query="carts to MongoDB intermittent packet loss causing data inconsistency in Sock Shop",
        expected_fault_ids=["SS-NET-MONGO-002", "NET-LOSS-001"],
        fault_type="Network",
        source="handcrafted",
        label="SS carts-mongo packet loss",
    ),
    TestCase(
        query="external payment gateway timing out in Train Ticket system",
        expected_fault_ids=["TT-NET-F7-01", "NET-DELAY-001"],
        fault_type="Network",
        source="handcrafted",
        label="TT external payment timeout",
    ),
    TestCase(
        query="cancellation messages arriving out of order in Train Ticket due to network delays",
        expected_fault_ids=["TT-NET-F1-01", "TT-NET-F1-02"],
        fault_type="Network",
        source="handcrafted",
        label="TT cancellation message reorder",
    ),

    # ── Code-level faults ──
    TestCase(
        query="checkoutservice sending wrong amount to paymentservice, charges are incorrect",
        expected_fault_ids=["OB-CODE-PARAM-001", "CODE-PARAM-001"],
        fault_type="Code-level",
        source="handcrafted",
        label="OB wrong payment params",
    ),
    TestCase(
        query="checkout completes but shipping never happens in Online Boutique, ShipOrder not called",
        expected_fault_ids=["OB-CODE-FUNC-001", "CODE-FUNC-001"],
        fault_type="Code-level",
        source="handcrafted",
        label="OB missing shipping call",
    ),
    TestCase(
        query="no confirmation email sent after successful checkout in Online Boutique",
        expected_fault_ids=["OB-CODE-FUNC-002", "CODE-FUNC-001"],
        fault_type="Code-level",
        source="handcrafted",
        label="OB missing email call",
    ),
    TestCase(
        query="payment service returns incorrect authorization result, order goes through without payment",
        expected_fault_ids=["SS-CODE-RET-001", "CODE-RET-001"],
        fault_type="Code-level",
        source="handcrafted",
        label="SS payment wrong return",
    ),
    TestCase(
        query="orders placed but payment call is completely skipped in Sock Shop",
        expected_fault_ids=["SS-CODE-FUNC-001", "CODE-FUNC-001"],
        fault_type="Code-level",
        source="handcrafted",
        label="SS missing payment call",
    ),
    TestCase(
        query="customer cart not cleared after order completion in Sock Shop, items remain",
        expected_fault_ids=["SS-CODE-FUNC-002", "CODE-FUNC-001"],
        fault_type="Code-level",
        source="handcrafted",
        label="SS cart not cleared",
    ),
    TestCase(
        query="wrong contacts API called during reservation in Train Ticket",
        expected_fault_ids=["TT-CODE-F10-01", "CODE-PARAM-001"],
        fault_type="Code-level",
        source="handcrafted",
        label="TT wrong contacts API",
    ),
    TestCase(
        query="second class ticket prices are being calculated incorrectly in Train Ticket",
        expected_fault_ids=["TT-CODE-F14-01"],
        fault_type="Code-level",
        source="handcrafted",
        label="TT wrong price calc",
    ),
    TestCase(
        query="food order response is null, frontend chart crashes in Train Ticket",
        expected_fault_ids=["TT-CODE-F18-01", "CODE-RET-001"],
        fault_type="Code-level",
        source="handcrafted",
        label="TT null food response",
    ),
    TestCase(
        query="voucher query returning empty because of wrong SQL column name in Train Ticket",
        expected_fault_ids=["TT-CODE-F22-01"],
        fault_type="Code-level",
        source="handcrafted",
        label="TT wrong voucher column",
    ),
    TestCase(
        query="SQL error in voucher service triggers recursive retry storm flooding the database",
        expected_fault_ids=["TT-CODE-F6-01"],
        fault_type="Code-level",
        source="handcrafted",
        label="TT SQL retry storm",
    ),

    # ── Cross-system / Kubernetes ──
    TestCase(
        query="pod stuck in CrashLoopBackOff, keeps restarting every 30 seconds",
        expected_fault_ids=["GEN-K8S-CRASH-01", "K8S-CRASH-001"],
        fault_type="Resource",
        source="handcrafted",
        label="K8s CrashLoopBackOff",
    ),
    TestCase(
        query="service DNS resolution failing, can't resolve cluster.local addresses",
        expected_fault_ids=["GEN-K8S-DNS-01", "K8S-DNS-001"],
        fault_type="Network",
        source="handcrafted",
        label="K8s DNS failure",
    ),
    TestCase(
        query="one service went down and now everything downstream is failing too, cascading outage",
        expected_fault_ids=["GEN-CASCADE-01"],
        fault_type="Network",
        source="handcrafted",
        label="Cascading dependency failure",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# 2.  METRICS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CaseResult:
    """Result of one test case."""
    case: TestCase
    retrieved_fault_ids: list[str]
    retrieved_scores: list[float]
    hit_at: dict[int, bool] = field(default_factory=dict)   # k → hit?
    reciprocal_rank: float = 0.0
    precision_at: dict[int, float] = field(default_factory=dict)


def evaluate_case(case: TestCase, top_k: int = 5) -> CaseResult:
    """Run one test case through the retriever and compute per-case metrics."""
    results = retrieve(case.query, top_k=max(K_VALUES))

    retrieved_ids = [r["metadata"].get("fault_id", "") for r in results]
    retrieved_scores = [r["score"] for r in results]

    result = CaseResult(
        case=case,
        retrieved_fault_ids=retrieved_ids,
        retrieved_scores=retrieved_scores,
    )

    # Find rank of first correct hit (1-indexed)
    first_rank = None
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in case.expected_fault_ids:
            first_rank = rank
            break

    # Hit@k
    for k in K_VALUES:
        top_k_ids = retrieved_ids[:k]
        result.hit_at[k] = any(rid in case.expected_fault_ids for rid in top_k_ids)

    # Reciprocal rank
    result.reciprocal_rank = (1.0 / first_rank) if first_rank else 0.0

    # Precision@k
    for k in K_VALUES:
        top_k_ids = retrieved_ids[:k]
        n_correct = sum(1 for rid in top_k_ids if rid in case.expected_fault_ids)
        result.precision_at[k] = n_correct / k

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3.  AGGREGATION AND REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_results(results: list[CaseResult]) -> dict:
    """Compute aggregate metrics from a list of case results."""
    n = len(results)
    if n == 0:
        return {}

    agg = {
        "n": n,
        "mrr": _avg([r.reciprocal_rank for r in results]),
    }
    for k in K_VALUES:
        agg[f"hit@{k}"] = _avg([float(r.hit_at[k]) for r in results])
        agg[f"precision@{k}"] = _avg([r.precision_at[k] for r in results])

    return agg


def print_report(all_results: list[CaseResult]) -> None:
    """Print a formatted evaluation report."""
    W = 72  # report width

    print("\n" + "=" * W)
    print("  TriageRCA  —  Retrieval Evaluation Report")
    print("=" * W)

    # ── Overall ──
    overall = aggregate_results(all_results)
    print(f"\n{'OVERALL':^{W}}")
    print(f"  Total test cases:  {overall['n']}")
    print(f"  MRR:               {overall['mrr']:.4f}")
    for k in K_VALUES:
        print(f"  Hit@{k}:             {overall[f'hit@{k}']:.4f}")
    for k in K_VALUES:
        print(f"  Precision@{k}:       {overall[f'precision@{k}']:.4f}")

    # ── By source ──
    for source_label in ["rcaeval", "handcrafted"]:
        subset = [r for r in all_results if r.case.source == source_label]
        if not subset:
            continue
        agg = aggregate_results(subset)
        print(f"\n{'─' * W}")
        print(f"  Source: {source_label}  ({agg['n']} cases)")
        print(f"  MRR:               {agg['mrr']:.4f}")
        for k in K_VALUES:
            print(f"  Hit@{k}:             {agg[f'hit@{k}']:.4f}")
        for k in K_VALUES:
            print(f"  Precision@{k}:       {agg[f'precision@{k}']:.4f}")

    # ── By fault type ──
    fault_types = sorted(set(r.case.fault_type for r in all_results))
    print(f"\n{'─' * W}")
    print("  BREAKDOWN BY FAULT TYPE")
    for ft in fault_types:
        subset = [r for r in all_results if r.case.fault_type == ft]
        agg = aggregate_results(subset)
        print(f"\n  ▸ {ft}  ({agg['n']} cases)")
        print(f"    MRR:           {agg['mrr']:.4f}")
        for k in K_VALUES:
            print(f"    Hit@{k}:         {agg[f'hit@{k}']:.4f}")

    # ── By system (for rcaeval only) ──
    rcaeval = [r for r in all_results if r.case.source == "rcaeval"]
    if rcaeval:
        print(f"\n{'─' * W}")
        print("  BREAKDOWN BY SYSTEM (RCAEval only)")
        for sys_code in ["RE1-OB", "RE1-SS", "RE1-TT"]:
            subset = [r for r in rcaeval if r.case.label.startswith(sys_code)]
            if not subset:
                continue
            agg = aggregate_results(subset)
            sys_name = {"RE1-OB": "Online Boutique", "RE1-SS": "Sock Shop", "RE1-TT": "Train Ticket"}[sys_code]
            print(f"\n  ▸ {sys_name}  ({agg['n']} cases)")
            print(f"    MRR:           {agg['mrr']:.4f}")
            for k in K_VALUES:
                print(f"    Hit@{k}:         {agg[f'hit@{k}']:.4f}")

    # ── Failures (misses at k=5) ──
    misses = [r for r in all_results if not r.hit_at[max(K_VALUES)]]
    if misses:
        print(f"\n{'─' * W}")
        print(f"  MISSES (not found in top-{max(K_VALUES)}):  {len(misses)} cases")
        for r in misses:
            print(f"\n  ✗ {r.case.label}")
            print(f"    Query:    {r.case.query[:90]}…" if len(r.case.query) > 90 else f"    Query:    {r.case.query}")
            print(f"    Expected: {r.case.expected_fault_ids}")
            print(f"    Got:      {r.retrieved_fault_ids}")
            if r.retrieved_scores:
                print(f"    Scores:   {[round(s, 4) for s in r.retrieved_scores]}")
    else:
        print(f"\n  ✓ No misses — every query found the correct runbook in top-{max(K_VALUES)}!")

    print(f"\n{'=' * W}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TriageRCA Retrieval Evaluation")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only handcrafted queries (skip RCAEval dataset)",
    )
    parser.add_argument(
        "--re1-dir", default="data/RE1",
        help="Path to the RE1 dataset directory (default: data/RE1)",
    )
    args = parser.parse_args()

    cases: list[TestCase] = []

    # Handcrafted queries
    print(f"Loading {len(HANDCRAFTED_CASES)} handcrafted test cases...")
    cases.extend(HANDCRAFTED_CASES)

    # RCAEval queries
    if not args.quick:
        print(f"Scanning RE1 dataset at {args.re1_dir}...")
        rcaeval_cases = build_rcaeval_cases(args.re1_dir)
        print(f"Loaded {len(rcaeval_cases)} RCAEval test cases")
        cases.extend(rcaeval_cases)

    print(f"\nTotal: {len(cases)} test cases")
    print("Running retrieval evaluation...\n")

    # Evaluate
    all_results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        tag = f"[{i}/{len(cases)}]"
        print(f"  {tag:>10s}  {case.label or case.query[:60]}")
        result = evaluate_case(case)
        all_results.append(result)

    # Report
    print_report(all_results)


if __name__ == "__main__":
    main()
