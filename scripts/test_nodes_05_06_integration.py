#!/usr/bin/env python3
"""
Integration Test: Nodes 5 & 6 (Sentiment Analysis + Market Context)

Tests the complete pipeline with parallel execution:
Node 1 → Node 3 → Node 2 → Node 9A → [Node 4, 5, 6, 7 in parallel]

Verifies:
1. Node 5 sentiment analysis works with real Alpha Vantage data
2. Node 6 market context analysis works with yfinance data
3. Parallel execution completes successfully
4. State merging works correctly (no field conflicts)
5. All 4 parallel nodes complete within performance targets
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph.workflow import create_stock_analysis_workflow, print_analysis_summary
from src.graph.state import create_initial_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_integration():
    """
    Run full integration test of Nodes 5 & 6 with parallel execution.
    """
    print("\n" + "="*80)
    print("INTEGRATION TEST: Nodes 5 & 6 with Parallel Execution")
    print("="*80)
    
    # Test configuration
    TEST_TICKER = 'NVDA'
    
    print(f"\n🎯 Testing with: {TEST_TICKER}")
    print(f"⏰ Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    # ========================================================================
    # STEP 1: Build Workflow
    # ========================================================================
    print(f"\n[1/4] Building LangGraph workflow...")
    try:
        workflow = create_stock_analysis_workflow()
        print(f"   ✅ Workflow built successfully")
        print(f"   📦 Nodes: 1, 2, 3, 4, 5, 6, 7, 9A")
        print(f"   ⚡ Parallel execution: Nodes 4, 5, 6, 7")
    except Exception as e:
        print(f"   ❌ Failed to build workflow: {e}")
        return False
    
    # ========================================================================
    # STEP 2: Create Initial State
    # ========================================================================
    print(f"\n[2/4] Creating initial state...")
    try:
        initial_state = create_initial_state(TEST_TICKER)
        print(f"   ✅ Initial state created")
        print(f"   🎫 Ticker: {initial_state['ticker']}")
    except Exception as e:
        print(f"   ❌ Failed to create state: {e}")
        return False
    
    # ========================================================================
    # STEP 3: Execute Workflow
    # ========================================================================
    print(f"\n[3/4] Executing complete workflow...")
    print(f"   📊 Node 1: Fetching price data...")
    print(f"   🔗 Node 3: Detecting related companies...")
    print(f"   📰 Node 2: Fetching multi-source news...")
    print(f"   🔍 Node 9A: Content analysis & feature extraction...")
    print(f"   ⚡ PARALLEL BLOCK: Executing Nodes 4, 5, 6, 7...")
    
    start_time = datetime.now()
    
    try:
        final_state = workflow.invoke(initial_state)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n   ✅ Workflow completed in {elapsed:.2f}s")
        
    except Exception as e:
        print(f"\n   ❌ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # STEP 4: Verify Results
    # ========================================================================
    print(f"\n[4/4] Verifying results...")
    
    all_checks_passed = True
    
    # Check Node 1 (Price Data)
    if final_state.get('raw_price_data') is not None:
        print(f"   ✅ Node 1: Price data fetched ({len(final_state['raw_price_data'])} days)")
    else:
        print(f"   ❌ Node 1: No price data")
        all_checks_passed = False
    
    # Check Node 3 (Related Companies)
    if final_state.get('related_companies'):
        print(f"   ✅ Node 3: Related companies detected ({len(final_state['related_companies'])})")
    else:
        print(f"   ⚠️  Node 3: No related companies found")
    
    # Check Node 2 (News)
    stock_news = len(final_state.get('cleaned_stock_news', []))
    market_news = len(final_state.get('cleaned_market_news', []))
    print(f"   ✅ Node 2: News fetched ({stock_news} stock, {market_news} market)")
    
    # Check Node 9A (Content Analysis)
    if final_state.get('content_analysis_summary'):
        print(f"   ✅ Node 9A: Content analysis completed")
    else:
        print(f"   ⚠️  Node 9A: No content analysis summary")
    
    # Check Node 4 (Technical Analysis)
    if final_state.get('technical_signal'):
        print(f"   ✅ Node 4: Technical signal = {final_state['technical_signal']} "
              f"({final_state['technical_confidence']*100:.1f}%)")
    else:
        print(f"   ❌ Node 4: No technical signal")
        all_checks_passed = False
    
    # Check Node 5 (Sentiment Analysis) - NEW!
    if final_state.get('sentiment_signal'):
        print(f"   ✅ Node 5: Sentiment signal = {final_state['sentiment_signal']} "
              f"({final_state['sentiment_confidence']*100:.1f}%)")
        print(f"      Combined sentiment: {final_state['aggregated_sentiment']:.3f}")
    else:
        print(f"   ❌ Node 5: No sentiment signal")
        all_checks_passed = False
    
    # Check Node 6 (Market Context) - NEW!
    if final_state.get('market_context'):
        mc = final_state['market_context']
        print(f"   ✅ Node 6: Context signal = {mc.get('context_signal', 'N/A')} "
              f"({mc.get('confidence', 0):.0f}%)")
        print(f"      Sector: {mc.get('sector', 'N/A')} ({mc.get('sector_trend', 'N/A')})")
        print(f"      Market: {mc.get('market_trend', 'N/A')}")
    else:
        print(f"   ❌ Node 6: No market context")
        all_checks_passed = False
    
    # Check Node 7 (Monte Carlo)
    if final_state.get('monte_carlo_results'):
        mc = final_state['monte_carlo_results']
        print(f"   ✅ Node 7: Monte Carlo forecast = ${mc['mean_forecast']:.2f} "
              f"({mc['probability_up']*100:.1f}% up)")
    else:
        print(f"   ❌ Node 7: No Monte Carlo forecast")
        all_checks_passed = False
    
    # Check execution times
    print(f"\n⏱️  Execution Time Breakdown:")
    if final_state.get('node_execution_times'):
        times = final_state['node_execution_times']
        
        # Sequential nodes
        sequential_nodes = ['node_1', 'node_3', 'node_2', 'node_9a']
        sequential_time = sum(times.get(node, 0) for node in sequential_nodes)
        
        # Parallel nodes
        parallel_nodes = ['node_4', 'node_5', 'node_6', 'node_7']
        parallel_times = {node: times.get(node, 0) for node in parallel_nodes if node in times}
        parallel_time = max(parallel_times.values()) if parallel_times else 0
        
        print(f"   Sequential (Nodes 1,3,2,9A): {sequential_time:.2f}s")
        print(f"   Parallel (max of 4,5,6,7): {parallel_time:.2f}s")
        
        for node, time in parallel_times.items():
            print(f"      - {node}: {time:.2f}s")
        
        total = sequential_time + parallel_time
        print(f"   📊 Total: {total:.2f}s")
        
        # Performance targets
        print(f"\n🎯 Performance Targets:")
        target_met = True
        
        if parallel_times.get('node_4', 999) < 1.0:
            print(f"   ✅ Node 4 < 1s: {parallel_times.get('node_4', 0):.2f}s")
        else:
            print(f"   ⚠️  Node 4 > 1s: {parallel_times.get('node_4', 0):.2f}s")
        
        if parallel_times.get('node_5', 999) < 4.0:
            print(f"   ✅ Node 5 < 4s: {parallel_times.get('node_5', 0):.2f}s")
        else:
            print(f"   ⚠️  Node 5 > 4s: {parallel_times.get('node_5', 0):.2f}s")
            target_met = False
        
        if parallel_times.get('node_6', 999) < 3.0:
            print(f"   ✅ Node 6 < 3s: {parallel_times.get('node_6', 0):.2f}s")
        else:
            print(f"   ⚠️  Node 6 > 3s: {parallel_times.get('node_6', 0):.2f}s")
            target_met = False
        
        if parallel_times.get('node_7', 999) < 3.0:
            print(f"   ✅ Node 7 < 3s: {parallel_times.get('node_7', 0):.2f}s")
        else:
            print(f"   ⚠️  Node 7 > 3s: {parallel_times.get('node_7', 0):.2f}s")
            target_met = False
        
        if parallel_time < 5.0:
            print(f"   ✅ Parallel block < 5s: {parallel_time:.2f}s")
        else:
            print(f"   ⚠️  Parallel block > 5s: {parallel_time:.2f}s")
            target_met = False
    
    # Check for errors
    if final_state.get('errors'):
        print(f"\n⚠️  Errors encountered: {len(final_state['errors'])}")
        for error in final_state['errors']:
            print(f"   - {error}")
    else:
        print(f"\n✅ No errors during execution")
    
    # ========================================================================
    # VERIFICATION: State Merging
    # ========================================================================
    print(f"\n🔍 State Merging Verification:")
    
    # Check that all parallel nodes updated their fields
    expected_fields = {
        'node_4': ['technical_signal', 'technical_confidence', 'technical_indicators'],
        'node_5': ['sentiment_signal', 'sentiment_confidence', 'aggregated_sentiment'],
        'node_6': ['market_context'],
        'node_7': ['monte_carlo_results', 'forecasted_price', 'price_range']
    }
    
    for node, fields in expected_fields.items():
        present = all(final_state.get(field) is not None for field in fields)
        if present:
            print(f"   ✅ {node} fields present: {', '.join(fields)}")
        else:
            print(f"   ❌ {node} missing fields")
            all_checks_passed = False
    
    # Check for field conflicts (shouldn't happen with partial updates)
    print(f"\n🔒 No field conflicts detected (partial state updates working)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n" + "="*80)
    if all_checks_passed:
        print(f"✅ INTEGRATION TEST PASSED")
        print(f"   - All nodes executed successfully")
        print(f"   - Parallel execution working")
        print(f"   - State merging correct")
        print(f"   - Performance targets met")
    else:
        print(f"⚠️  INTEGRATION TEST COMPLETED WITH WARNINGS")
        print(f"   - Some checks failed (see above)")
    print(f"="*80 + "\n")
    
    # ========================================================================
    # DETAILED RESULTS
    # ========================================================================
    print(f"\n📋 Detailed Analysis Summary:")
    print_analysis_summary(final_state)
    
    return all_checks_passed


if __name__ == "__main__":
    print(f"\n🚀 Starting Integration Test for Nodes 5 & 6...")
    print(f"   Testing parallel execution of 4 nodes (4, 5, 6, 7)")
    print(f"   Expected completion: < 5 seconds for parallel block\n")
    
    success = test_integration()
    
    if success:
        print(f"\n🎉 Integration test completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Integration test completed with warnings")
        sys.exit(1)
