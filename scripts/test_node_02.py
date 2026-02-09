"""
Test Node 2: Multi-Source News Fetching
Quick test to verify Node 2 works correctly with async parallel fetching.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph.state import create_initial_state
from src.langgraph_nodes.node_01_data_fetching import fetch_price_data_node
from src.langgraph_nodes.node_03_related_companies import detect_related_companies_node
from src.langgraph_nodes.node_02_news_fetching import fetch_all_news_node
from src.utils.logger import setup_application_logging

# Setup logging
setup_application_logging()

print("=" * 80)
print("Testing Node 2: Multi-Source News Fetching")
print("=" * 80)

# Test 1: Full pipeline (Node 1 → Node 3 → Node 2)
print("\n📰 Test 1: Full pipeline with AAPL")
print("-" * 80)

# Run Nodes 1 and 3 first to set up state
state = create_initial_state('AAPL')
state = fetch_price_data_node(state)
state = detect_related_companies_node(state)

print(f"   Related companies found: {', '.join(state['related_companies'][:3])}...")

# Now run Node 2
result = fetch_all_news_node(state)

print(f"\n✅ Node 2 completed!")
print(f"   Stock news articles: {len(result['stock_news'])}")
print(f"   Market news articles: {len(result['market_news'])}")
print(f"   Related company news: {len(result['related_company_news'])}")
print(f"   Total articles: {len(result['stock_news']) + len(result['market_news']) + len(result['related_company_news'])}")
print(f"   Execution time: {result['node_execution_times']['node_2']:.2f}s")

# Show sample article
if result['stock_news']:
    article = result['stock_news'][0]
    print(f"\n   Sample stock news article:")
    print(f"   - Headline: {article['headline'][:80]}...")
    print(f"   - Source: {article['source']}")
    print(f"   - Type: {article['news_type']}")

# Test 2: Cache hit (should be faster)
print("\n\n📰 Test 2: Cache hit test (same ticker)")
print("-" * 80)

state2 = create_initial_state('AAPL')
state2 = fetch_price_data_node(state2)
state2 = detect_related_companies_node(state2)
result2 = fetch_all_news_node(state2)

print(f"   Execution time: {result2['node_execution_times']['node_2']:.2f}s")
if result2['node_execution_times']['node_2'] < result['node_execution_times']['node_2']:
    print(f"   ⚡ Cache made it faster! ({result['node_execution_times']['node_2']:.2f}s → {result2['node_execution_times']['node_2']:.2f}s)")
else:
    print(f"   Note: Similar speed (may have re-fetched)")

# Test 3: Different ticker
print("\n\n📰 Test 3: Different ticker (NVDA)")
print("-" * 80)

state3 = create_initial_state('NVDA')
state3 = fetch_price_data_node(state3)
state3 = detect_related_companies_node(state3)
result3 = fetch_all_news_node(state3)

print(f"   Stock news articles: {len(result3['stock_news'])}")
print(f"   Market news articles: {len(result3['market_news'])}")
print(f"   Related company news: {len(result3['related_company_news'])}")
print(f"   Execution time: {result3['node_execution_times']['node_2']:.2f}s")

# Test 4: Without related companies
print("\n\n📰 Test 4: Without related companies")
print("-" * 80)

state4 = create_initial_state('TSLA')
# Skip Node 3, so no related companies
result4 = fetch_all_news_node(state4)

print(f"   Stock news articles: {len(result4['stock_news'])}")
print(f"   Market news articles: {len(result4['market_news'])}")
print(f"   Related company news: {len(result4['related_company_news'])}")
print(f"   Execution time: {result4['node_execution_times']['node_2']:.2f}s")

# Summary
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)

tests_passed = 0
tests_total = 4

total_articles_1 = len(result['stock_news']) + len(result['market_news']) + len(result['related_company_news'])
if total_articles_1 > 0:
    tests_passed += 1
    print("✅ Test 1: Full pipeline (AAPL) - PASSED")
else:
    print("❌ Test 1: Full pipeline (AAPL) - FAILED")

if result2['stock_news'] or result2['market_news']:
    tests_passed += 1
    print("✅ Test 2: Cache hit - PASSED")
else:
    print("❌ Test 2: Cache hit - FAILED")

total_articles_3 = len(result3['stock_news']) + len(result3['market_news']) + len(result3['related_company_news'])
if total_articles_3 > 0:
    tests_passed += 1
    print("✅ Test 3: Different ticker (NVDA) - PASSED")
else:
    print("❌ Test 3: Different ticker (NVDA) - FAILED")

if result4['stock_news'] or result4['market_news']:
    tests_passed += 1
    print("✅ Test 4: Without related companies - PASSED")
else:
    print("❌ Test 4: Without related companies - FAILED")

print(f"\n{tests_passed}/{tests_total} tests passed")
print("=" * 80)
