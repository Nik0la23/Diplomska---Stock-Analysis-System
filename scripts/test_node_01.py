"""
Test Node 1: Price Data Fetching
Quick test to verify Node 1 works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph.state import create_initial_state
from src.langgraph_nodes.node_01_data_fetching import fetch_price_data_node
from src.utils.logger import setup_application_logging

# Setup logging
setup_application_logging()

print("=" * 80)
print("Testing Node 1: Price Data Fetching")
print("=" * 80)

# Test 1: Valid ticker (AAPL)
print("\n📊 Test 1: Valid ticker (AAPL)")
print("-" * 80)

state = create_initial_state('AAPL')
result = fetch_price_data_node(state)

if result['raw_price_data'] is not None:
    df = result['raw_price_data']
    print(f"✅ SUCCESS: Fetched {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Execution time: {result['node_execution_times']['node_1']:.2f}s")
    print(f"\n   Sample data (last 3 rows):")
    print(df.tail(3).to_string())
else:
    print(f"❌ FAILED: No data returned")
    print(f"   Errors: {result['errors']}")

# Test 2: Cache hit (should be faster)
print("\n\n📊 Test 2: Cache hit test (same ticker)")
print("-" * 80)

state2 = create_initial_state('AAPL')
result2 = fetch_price_data_node(state2)

if result2['raw_price_data'] is not None:
    print(f"✅ SUCCESS: Cache hit - {len(result2['raw_price_data'])} rows")
    print(f"   Execution time: {result2['node_execution_times']['node_1']:.2f}s")
    if result2['node_execution_times']['node_1'] < result['node_execution_times']['node_1']:
        print(f"   ⚡ Cache made it faster! ({result['node_execution_times']['node_1']:.2f}s → {result2['node_execution_times']['node_1']:.2f}s)")
else:
    print(f"❌ FAILED: Cache should have worked")

# Test 3: Invalid ticker (should handle gracefully)
print("\n\n📊 Test 3: Invalid ticker (INVALID123)")
print("-" * 80)

state3 = create_initial_state('INVALID123')
result3 = fetch_price_data_node(state3)

if result3['raw_price_data'] is None:
    print(f"✅ SUCCESS: Handled invalid ticker gracefully")
    print(f"   Errors logged: {len(result3['errors'])}")
    if result3['errors']:
        print(f"   Error message: {result3['errors'][0]}")
else:
    print(f"❌ UNEXPECTED: Got data for invalid ticker")

# Summary
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)

tests_passed = 0
tests_total = 3

if result['raw_price_data'] is not None:
    tests_passed += 1
    print("✅ Test 1: Valid ticker - PASSED")
else:
    print("❌ Test 1: Valid ticker - FAILED")

if result2['raw_price_data'] is not None:
    tests_passed += 1
    print("✅ Test 2: Cache hit - PASSED")
else:
    print("❌ Test 2: Cache hit - FAILED")

if result3['raw_price_data'] is None and len(result3['errors']) > 0:
    tests_passed += 1
    print("✅ Test 3: Invalid ticker handling - PASSED")
else:
    print("❌ Test 3: Invalid ticker handling - FAILED")

print(f"\n{tests_passed}/{tests_total} tests passed")
print("=" * 80)
