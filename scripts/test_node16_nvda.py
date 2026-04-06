"""
Isolated test for Node 16 (SEC Fundamentals) — NVDA.

Runs ONLY Node 16 with a minimal state that supplies:
  - ticker: NVDA
  - related_companies: AMD and INTC as known NVDA peers (as Node 3 would provide)
  - final_signal: BUY  (simulates Node 12 output for divergence detection)

Prints a structured report verifying:
  1. Filing dates and quarters covered (no hardcoded fallbacks)
  2. Realistic management_sentiment from FinBERT (non-zero for 8-K)
  3. Revenue/margin trends derived from actual 10-Q text
  4. Peer events from real 8-K filings
  5. Divergence detection between SEC signal and Node 12's BUY
  6. data_quality is 'complete' or 'partial', never from the error path
"""

import sys
import json
import logging
from pathlib import Path

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.langgraph_nodes.node_16_sec_fundamentals import sec_fundamentals_node

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)

# ── Minimal state ─────────────────────────────────────────────────────────────
state = {
    "ticker": "NVDA",
    "related_companies": [
        {"ticker": "AMD",  "relationship": "COMPETITOR"},
        {"ticker": "INTC", "relationship": "COMPETITOR"},
    ],
    "final_signal": "BUY",
    "errors": [],
    "node_execution_times": {},
}

# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 70)
print("NODE 16 ISOLATED TEST — NVDA")
print("=" * 70)

result = sec_fundamentals_node(state)
fc = result.get("fundamental_context", {})

# ── Guard: was the error-path fallback triggered? ─────────────────────────────
errors = result.get("errors", [])
node_error = [e for e in errors if "Node 16" in e]
if node_error:
    print("\n[FAIL] Node 16 hit the exception path:")
    for e in node_error:
        print(f"  {e}")
    print("\nfundamental_context (error stub):")
    print(json.dumps(fc, indent=2, default=str))
    sys.exit(1)

print(f"\n[PASS] No exceptions — error path was NOT triggered\n")

# ── 1. Target summary ─────────────────────────────────────────────────────────
target = fc.get("target", {})
print("── TARGET ──────────────────────────────────────────────────────────────")
print(f"  ticker            : {target.get('ticker')}")
print(f"  revenue_trend     : {target.get('revenue_trend')}")
print(f"  margin_trend      : {target.get('margin_trend')}")
print(f"  management_sentiment : {target.get('management_sentiment')}")
print(f"  quarters_covered  : {target.get('quarters_covered')}")
print(f"  filing_dates      : {target.get('filing_dates')}")
print(f"  recent_events     : {target.get('recent_events')}")

# ── Check: hardcoded fallback indicators ─────────────────────────────────────
mgmt = target.get("management_sentiment")
if mgmt is None:
    print("  [WARN] management_sentiment is None — FinBERT scored 0 chunks "
          "(check SEC fetch or token truncation)")
elif mgmt == 0.0:
    print("  [WARN] management_sentiment is exactly 0.0 — possible silent "
          "FinBERT failure (tensor error still occurring)")
else:
    print(f"  [PASS] management_sentiment is non-zero: {mgmt:.4f}")

q = target.get("quarters_covered", 0)
if q == 0:
    print("  [WARN] quarters_covered = 0 — no 10-Q filings fetched")
else:
    print(f"  [PASS] quarters_covered = {q}")

# ── 2. Fundamental signal ─────────────────────────────────────────────────────
print("\n── SIGNAL ───────────────────────────────────────────────────────────────")
print(f"  fundamental_signal     : {fc.get('fundamental_signal')}")
print(f"  fundamental_confidence : {fc.get('fundamental_confidence')}")
print(f"  data_quality           : {fc.get('data_quality')}")

if fc.get("data_quality") == "unavailable" and not node_error:
    print("  [WARN] data_quality=unavailable even without a thrown exception "
          "— no filings were fetched at all")
else:
    print(f"  [PASS] data_quality = {fc.get('data_quality')}")

# ── 3. Peers ──────────────────────────────────────────────────────────────────
peers = fc.get("peers", {})
print(f"\n── PEERS ({len(peers)} fetched) ──────────────────────────────────────────")
if not peers:
    print("  [WARN] No peer data — CIK resolution or SEC fetch failed for all peers")
for pt, pd_ in peers.items():
    sent = pd_.get("event_sentiment")
    print(f"  {pt:6s}  relationship={pd_.get('relationship'):12s}  "
          f"events={pd_.get('recent_events')}  "
          f"sentiment={sent}  "
          f"filing_dates={pd_.get('filing_dates')}")
    if sent == 0.0:
        print(f"         [WARN] {pt} event_sentiment is exactly 0.0 — "
              "possible FinBERT silent failure")
    elif sent is None:
        print(f"         [INFO] {pt} sentiment is None — no 8-K text fetched")
    else:
        print(f"         [PASS] {pt} sentiment is non-zero: {sent:.4f}")

# ── 4. Divergences ────────────────────────────────────────────────────────────
divs = fc.get("divergences", [])
print(f"\n── DIVERGENCES ({len(divs)} found) ──────────────────────────────────────")
for i, d in enumerate(divs, 1):
    print(f"  {i}. {d}")
if not divs:
    print("  (none — fundamental signal agrees with Node 12 or no peers to compare)")

# ── 5. Execution time ─────────────────────────────────────────────────────────
elapsed = result.get("node_execution_times", {}).get("node_16")
print(f"\n── TIMING ───────────────────────────────────────────────────────────────")
print(f"  node_16 execution time : {elapsed}s")

print("\n" + "=" * 70)
print("FULL fundamental_context JSON")
print("=" * 70)
print(json.dumps(fc, indent=2, default=str))
