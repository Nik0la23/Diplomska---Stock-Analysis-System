"""
Test Node 3: Related Companies Detection
Quick test to verify Node 3 works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph.state import create_initial_state
from src.langgraph_nodes.node_01_data_fetching import fetch_price_data_node
from src.langgraph_nodes.node_03_related_companies import detect_related_companies_node
from src.utils.logger import setup_application_logging

# Setup logging
setup_application_logging()

print("=" * 80)
print("Testing Node 3: Related Companies Detection")
print("=" * 80)

# Test 1: Technology stock (NVDA) with price data for correlation
print("\n📊 Test 1: Technology stock (NVDA) with correlation")
print("-" * 80)

# First fetch price data (Node 1)
state = create_initial_state('NVDA')
state = fetch_price_data_node(state)

# Then detect related companies (Node 3)
result = detect_related_companies_node(state)

if result['related_companies']:
    print(f"✅ SUCCESS: Found {len(result['related_companies'])} related companies")
    print(f"   Related companies: {', '.join(result['related_companies'])}")
    print(f"   Execution time: {result['node_execution_times']['node_3']:.2f}s")
    
    # Check that NVDA is not in the list
    if 'NVDA' not in result['related_companies']:
        print(f"   ✅ Correctly excluded target ticker (NVDA) from results")
    else:
        print(f"   ❌ Target ticker (NVDA) should not be in related companies!")
else:
    print(f"❌ FAILED: No related companies found")
    if result['errors']:
        print(f"   Errors: {result['errors']}")

# Test 2: Another major stock (AAPL)
print("\n\n📊 Test 2: Another major stock (AAPL)")
print("-" * 80)

state2 = create_initial_state('AAPL')
state2 = fetch_price_data_node(state2)
result2 = detect_related_companies_node(state2)

if result2['related_companies']:
    print(f"✅ SUCCESS: Found {len(result2['related_companies'])} related companies")
    print(f"   Related companies: {', '.join(result2['related_companies'])}")
    print(f"   Execution time: {result2['node_execution_times']['node_3']:.2f}s")
    
    if 'AAPL' not in result2['related_companies']:
        print(f"   ✅ Correctly excluded target ticker (AAPL) from results")
else:
    print(f"❌ FAILED: No related companies found")

# Test 3: Without price data (no correlation)
print("\n\n📊 Test 3: Without price data (TSLA, no correlation)")
print("-" * 80)

state3 = create_initial_state('TSLA')
# Don't run Node 1, so no price data available
result3 = detect_related_companies_node(state3)

if result3['related_companies']:
    print(f"✅ SUCCESS: Found {len(result3['related_companies'])} related companies")
    print(f"   Related companies: {', '.join(result3['related_companies'])}")
    print(f"   Execution time: {result3['node_execution_times']['node_3']:.2f}s")
    print(f"   Note: No correlation used (no price data)")
else:
    print(f"❌ FAILED: No related companies found")

# Test 4: Invalid ticker (should handle gracefully)
print("\n\n📊 Test 4: Invalid ticker (INVALID123)")
print("-" * 80)

state4 = create_initial_state('INVALID123')
result4 = detect_related_companies_node(state4)

if result4['related_companies'] == []:
    print(f"✅ SUCCESS: Handled invalid ticker gracefully (empty list)")
    print(f"   Execution time: {result4['node_execution_times']['node_3']:.2f}s")
else:
    print(f"❌ UNEXPECTED: Got related companies for invalid ticker")

# Summary
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)

tests_passed = 0
tests_total = 4

if result['related_companies'] and 'NVDA' not in result['related_companies']:
    tests_passed += 1
    print("✅ Test 1: NVDA with correlation - PASSED")
else:
    print("❌ Test 1: NVDA with correlation - FAILED")

if result2['related_companies'] and 'AAPL' not in result2['related_companies']:
    tests_passed += 1
    print("✅ Test 2: AAPL - PASSED")
else:
    print("❌ Test 2: AAPL - FAILED")

if result3['related_companies']:
    tests_passed += 1
    print("✅ Test 3: TSLA without price data - PASSED")
else:
    print("❌ Test 3: TSLA without price data - FAILED")

if result4['related_companies'] == []:
    tests_passed += 1
    print("✅ Test 4: Invalid ticker handling - PASSED")
else:
    print("❌ Test 4: Invalid ticker handling - FAILED")

print(f"\n{tests_passed}/{tests_total} tests passed")
print("=" * 80)
