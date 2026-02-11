"""
Integration test for Node 9A with Nodes 1-3
Tests the complete flow: Price -> Related Companies -> News -> Content Analysis
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph.state import create_initial_state
from src.langgraph_nodes.node_01_data_fetching import fetch_price_data_node
from src.langgraph_nodes.node_03_related_companies import detect_related_companies_node
from src.langgraph_nodes.node_02_news_fetching import fetch_all_news_node
from src.langgraph_nodes.node_09a_content_analysis import content_analysis_node

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_full_integration():
    """Test Nodes 1 -> 3 -> 2 -> 9A integration"""
    
    print("=" * 80)
    print("INTEGRATION TEST: Nodes 1 -> 3 -> 2 -> 9A")
    print("=" * 80)
    print()
    
    # Test ticker
    ticker = 'NVDA'
    
    # ========================================================================
    # STEP 1: Initialize State
    # ========================================================================
    print(f"Step 1: Initializing state for {ticker}")
    state = create_initial_state(ticker)
    print(f"✓ State initialized")
    print()
    
    # ========================================================================
    # STEP 2: Run Node 1 (Price Data)
    # ========================================================================
    print("Step 2: Fetching price data (Node 1)...")
    state = fetch_price_data_node(state)
    
    if state['raw_price_data'] is not None:
        print(f"✓ Price data fetched: {len(state['raw_price_data'])} days")
    else:
        print("✗ Price data fetch failed")
        if state['errors']:
            print(f"  Errors: {state['errors']}")
    print()
    
    # ========================================================================
    # STEP 3: Run Node 3 (Related Companies)
    # ========================================================================
    print("Step 3: Detecting related companies (Node 3)...")
    state = detect_related_companies_node(state)
    
    if state['related_companies']:
        print(f"✓ Related companies found: {state['related_companies']}")
    else:
        print("✓ No related companies (acceptable)")
    print()
    
    # ========================================================================
    # STEP 4: Run Node 2 (News Fetching)
    # ========================================================================
    print("Step 4: Fetching news from multiple sources (Node 2)...")
    state = fetch_all_news_node(state)
    
    stock_news_count = len(state.get('stock_news', []))
    market_news_count = len(state.get('market_news', []))
    related_news_count = len(state.get('related_company_news', []))
    total_news = stock_news_count + market_news_count + related_news_count
    
    print(f"✓ News fetched:")
    print(f"  - Stock news: {stock_news_count}")
    print(f"  - Market news: {market_news_count}")
    print(f"  - Related news: {related_news_count}")
    print(f"  - Total: {total_news}")
    print()
    
    # ========================================================================
    # STEP 5: Run Node 9A (Content Analysis)
    # ========================================================================
    print("Step 5: Analyzing content and scoring articles (Node 9A)...")
    state = content_analysis_node(state)
    
    cleaned_stock = len(state.get('cleaned_stock_news', []))
    cleaned_market = len(state.get('cleaned_market_news', []))
    cleaned_related = len(state.get('cleaned_related_company_news', []))
    total_cleaned = cleaned_stock + cleaned_market + cleaned_related
    
    print(f"✓ Content analysis completed:")
    print(f"  - Cleaned stock news: {cleaned_stock}")
    print(f"  - Cleaned market news: {cleaned_market}")
    print(f"  - Cleaned related news: {cleaned_related}")
    print(f"  - Total processed: {total_cleaned}")
    print()
    
    # ========================================================================
    # STEP 6: Verify Content Analysis Summary
    # ========================================================================
    if 'content_analysis_summary' in state:
        summary = state['content_analysis_summary']
        print("Content Analysis Summary:")
        print(f"  - Total articles processed: {summary.get('total_articles_processed', 0)}")
        
        if 'average_scores' in summary:
            avg_scores = summary['average_scores']
            print(f"  - Average sensationalism: {avg_scores.get('sensationalism', 0):.3f}")
            print(f"  - Average urgency: {avg_scores.get('urgency', 0):.3f}")
            print(f"  - Average unverified claims: {avg_scores.get('unverified_claims', 0):.3f}")
            print(f"  - Average source credibility: {avg_scores.get('source_credibility', 0):.3f}")
            print(f"  - Average composite anomaly: {avg_scores.get('composite_anomaly', 0):.3f}")
        
        print(f"  - High-risk articles: {summary.get('high_risk_articles', 0)} ({summary.get('high_risk_percentage', 0):.1f}%)")
        
        if 'source_credibility_distribution' in summary:
            dist = summary['source_credibility_distribution']
            print(f"  - Source distribution: High={dist.get('high', 0)}, Medium={dist.get('medium', 0)}, Low={dist.get('low', 0)}")
        
        if 'top_keywords' in summary and summary['top_keywords']:
            print(f"  - Top keywords: {', '.join([f'{kw}({count})' for kw, count in summary['top_keywords'][:5]])}")
    print()
    
    # ========================================================================
    # STEP 7: Inspect Sample Article
    # ========================================================================
    if cleaned_stock > 0:
        print("Sample Cleaned Article (Stock News):")
        sample = state['cleaned_stock_news'][0]
        print(f"  Headline: {sample.get('headline', sample.get('title', 'N/A'))[:80]}")
        print(f"  Source: {sample.get('source', 'N/A')}")
        print(f"  Sensationalism: {sample.get('sensationalism_score', 'N/A')}")
        print(f"  Urgency: {sample.get('urgency_score', 'N/A')}")
        print(f"  Unverified Claims: {sample.get('unverified_claims_score', 'N/A')}")
        print(f"  Source Credibility: {sample.get('source_credibility_score', 'N/A')}")
        print(f"  Composite Anomaly: {sample.get('composite_anomaly_score', 'N/A')}")
        
        if 'content_tags' in sample:
            tags = sample['content_tags']
            print(f"  Topic: {tags.get('topic', 'N/A')}")
            print(f"  Temporal: {tags.get('temporal', 'N/A')}")
            if tags.get('keywords'):
                print(f"  Keywords: {', '.join(tags['keywords'][:5])}")
    print()
    
    # ========================================================================
    # STEP 8: Execution Times
    # ========================================================================
    print("Execution Times:")
    for node_name, exec_time in state.get('node_execution_times', {}).items():
        print(f"  - {node_name}: {exec_time:.2f}s")
    
    total_time = sum(state.get('node_execution_times', {}).values())
    print(f"  - Total: {total_time:.2f}s")
    print()
    
    # ========================================================================
    # STEP 9: Verify Integration Success
    # ========================================================================
    print("=" * 80)
    print("INTEGRATION VERIFICATION")
    print("=" * 80)
    
    checks = []
    
    # Check 1: Price data fetched
    checks.append(("Price data fetched", state['raw_price_data'] is not None))
    
    # Check 2: News fetched
    checks.append(("News articles fetched", total_news > 0))
    
    # Check 3: Content analysis completed
    checks.append(("Content analysis completed", 'content_analysis_summary' in state))
    
    # Check 4: Cleaned news lists populated
    checks.append(("Cleaned news lists populated", total_cleaned > 0))
    
    # Check 5: All articles have scores
    all_have_scores = True
    if cleaned_stock > 0:
        sample = state['cleaned_stock_news'][0]
        required_fields = ['sensationalism_score', 'urgency_score', 'unverified_claims_score',
                          'source_credibility_score', 'composite_anomaly_score', 'content_tags']
        all_have_scores = all(field in sample for field in required_fields)
    checks.append(("Articles have embedded scores", all_have_scores))
    
    # Check 6: Node 9A execution time tracked
    checks.append(("Node 9A execution tracked", 'node_9a' in state['node_execution_times']))
    
    # Check 7: No critical errors
    checks.append(("No critical errors", len(state.get('errors', [])) == 0))
    
    # Print results
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
    
    print()
    
    # Final verdict
    all_passed = all(passed for _, passed in checks)
    if all_passed:
        print("🎉 ALL INTEGRATION CHECKS PASSED!")
        print("Node 9A is correctly integrated with Nodes 1-3.")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("Review the failed checks above.")
    
    print("=" * 80)
    
    return state, all_passed


if __name__ == '__main__':
    try:
        state, success = test_full_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Integration test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
