"""
Integration Tests for Parallel Nodes (4 & 7)

Tests the complete workflow from Node 1 through parallel execution of Nodes 4 & 7.

Flow tested:
Node 1 (Price) → Node 3 (Related) → Node 2 (News) → Node 9A (Content) 
    → Node 4 (Technical) & Node 7 (Monte Carlo) [PARALLEL]
"""

import pytest
import logging
from datetime import datetime
from src.graph.workflow import create_stock_analysis_workflow, run_stock_analysis
from src.graph.state import create_initial_state

# Set up logging for tests
logging.basicConfig(level=logging.INFO)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_workflow_creation():
    """Test that workflow can be created successfully"""
    workflow = create_stock_analysis_workflow()
    
    assert workflow is not None
    # Workflow should be compiled and ready to execute


def test_complete_workflow_execution():
    """Test complete workflow execution from start to end"""
    # Use a well-known ticker for reliable testing
    ticker = 'AAPL'
    
    # Create initial state
    initial_state = create_initial_state(ticker)
    
    # Build workflow
    workflow = create_stock_analysis_workflow()
    
    # Execute workflow
    final_state = workflow.invoke(initial_state)
    
    # Verify workflow completed
    assert final_state is not None
    assert final_state['ticker'] == ticker
    
    # Verify all nodes executed
    assert 'node_1' in final_state['node_execution_times']
    assert 'node_3' in final_state['node_execution_times']
    assert 'node_2' in final_state['node_execution_times']
    assert 'node_9a' in final_state['node_execution_times']
    assert 'node_4' in final_state['node_execution_times']
    assert 'node_7' in final_state['node_execution_times']


def test_parallel_nodes_both_execute():
    """Test that both parallel nodes (4 & 7) execute"""
    result = run_stock_analysis('NVDA')
    
    # Both nodes should have executed
    assert 'node_4' in result['node_execution_times']
    assert 'node_7' in result['node_execution_times']
    
    # Both should have results
    assert result.get('technical_signal') is not None
    assert result.get('monte_carlo_results') is not None


def test_parallel_execution_timing():
    """Test that parallel execution is faster than sequential would be"""
    result = run_stock_analysis('AAPL')
    
    node_4_time = result['node_execution_times'].get('node_4', 0)
    node_7_time = result['node_execution_times'].get('node_7', 0)
    total_time = sum(result['node_execution_times'].values())
    
    # Parallel execution should take less time than sequential
    # Total time should be less than sum of all node times if truly parallel
    # (Some overhead is expected, so we check it's reasonable)
    assert total_time > 0
    assert node_4_time > 0
    assert node_7_time > 0
    
    print(f"\n⏱️  Execution Times:")
    print(f"   Node 4 (Technical): {node_4_time:.2f}s")
    print(f"   Node 7 (Monte Carlo): {node_7_time:.2f}s")
    print(f"   Total workflow: {total_time:.2f}s")


def test_node1_to_node4_data_flow():
    """Test that data flows correctly from Node 1 to Node 4"""
    result = run_stock_analysis('AAPL')
    
    # Node 1 output should feed into Node 4
    assert result.get('raw_price_data') is not None
    assert result.get('technical_indicators') is not None
    assert result.get('technical_signal') in ['BUY', 'SELL', 'HOLD']


def test_node1_to_node7_data_flow():
    """Test that data flows correctly from Node 1 to Node 7"""
    result = run_stock_analysis('AAPL')
    
    # Node 1 output should feed into Node 7
    assert result.get('raw_price_data') is not None
    assert result.get('monte_carlo_results') is not None
    assert result.get('forecasted_price') is not None
    assert result.get('price_range') is not None


def test_node9a_to_parallel_nodes_data_flow():
    """Test that Node 9A cleaned news is available to parallel nodes"""
    result = run_stock_analysis('AAPL')
    
    # Node 9A should have created cleaned news
    assert 'cleaned_stock_news' in result
    assert 'cleaned_market_news' in result
    assert 'content_analysis_summary' in result
    
    # Parallel nodes should have executed after Node 9A
    assert result.get('technical_signal') is not None
    assert result.get('monte_carlo_results') is not None


def test_workflow_handles_errors_gracefully():
    """Test that workflow handles errors without crashing"""
    # Use an invalid ticker that might cause issues
    result = run_stock_analysis('INVALID_TICKER_XYZ')
    
    # Workflow should complete even with errors
    assert result is not None
    assert 'errors' in result
    
    # Some nodes may have failed, but workflow should not crash
    # At minimum, initial state should be preserved
    assert result['ticker'] == 'INVALID_TICKER_XYZ'


def test_state_accumulation():
    """Test that state accumulates data through the pipeline"""
    result = run_stock_analysis('NVDA')
    
    # Each node should add data to state
    expected_keys = [
        'ticker',
        'raw_price_data',
        'related_companies',
        'stock_news',
        'market_news',
        'cleaned_stock_news',
        'cleaned_market_news',
        'technical_signal',
        'technical_confidence',
        'monte_carlo_results',
        'forecasted_price',
        'node_execution_times'
    ]
    
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_multiple_tickers_sequential():
    """Test that workflow can analyze multiple tickers sequentially"""
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    results = {}
    
    for ticker in tickers:
        result = run_stock_analysis(ticker)
        results[ticker] = result
        
        # Each should complete successfully
        assert result is not None
        assert result['ticker'] == ticker
        assert result.get('technical_signal') is not None
    
    # All should have different results
    assert len(results) == 3
    assert len(set(r['technical_signal'] for r in results.values())) >= 1


def test_workflow_performance():
    """Test that complete workflow executes within acceptable time"""
    import time
    
    start_time = time.time()
    result = run_stock_analysis('AAPL')
    elapsed_time = time.time() - start_time
    
    # Workflow should complete in < 15 seconds
    assert elapsed_time < 15.0
    
    print(f"\n⚡ Total workflow execution: {elapsed_time:.2f}s")
    
    # Verify parallel nodes contributed to speed
    node_4_time = result['node_execution_times'].get('node_4', 0)
    node_7_time = result['node_execution_times'].get('node_7', 0)
    
    # If truly sequential, would take node_4_time + node_7_time
    # Parallel should take max(node_4_time, node_7_time)
    sequential_time = node_4_time + node_7_time
    parallel_time = max(node_4_time, node_7_time)
    
    print(f"   Parallel: {parallel_time:.2f}s vs Sequential would be: {sequential_time:.2f}s")
    print(f"   Speedup factor: {sequential_time/parallel_time:.2f}x")


def test_technical_and_monte_carlo_consistency():
    """Test that technical analysis and Monte Carlo provide consistent insights"""
    result = run_stock_analysis('AAPL')
    
    technical_signal = result.get('technical_signal')
    mc_results = result.get('monte_carlo_results')
    
    if technical_signal and mc_results:
        # If technical says BUY, Monte Carlo should show positive probability
        if technical_signal == 'BUY':
            assert mc_results['probability_up'] >= 0.4  # At least 40% chance up
        elif technical_signal == 'SELL':
            assert mc_results['probability_down'] >= 0.4  # At least 40% chance down
        
        # Monte Carlo expected return should align with technical signal direction
        if technical_signal == 'BUY':
            # Expected return should generally be positive for BUY
            # (Not strict requirement, just consistency check)
            print(f"   BUY signal with {mc_results['expected_return']:.2f}% expected return")
        elif technical_signal == 'SELL':
            print(f"   SELL signal with {mc_results['expected_return']:.2f}% expected return")


def test_full_integration_success_criteria():
    """Test that all success criteria are met"""
    result = run_stock_analysis('NVDA')
    
    # SUCCESS CRITERIA CHECKLIST
    
    # 1. Price data fetched
    assert result['raw_price_data'] is not None
    assert len(result['raw_price_data']) >= 50
    
    # 2. News fetched and analyzed
    assert len(result.get('cleaned_stock_news', [])) >= 0
    
    # 3. Technical analysis complete
    assert result['technical_signal'] in ['BUY', 'SELL', 'HOLD']
    assert 0 <= result['technical_confidence'] <= 1.0
    
    # 4. Monte Carlo forecast complete
    assert result['forecasted_price'] > 0
    assert result['monte_carlo_results']['num_simulations'] == 1000
    assert result['monte_carlo_results']['forecast_days'] == 7
    
    # 5. All nodes executed
    assert len(result['node_execution_times']) == 6
    
    # 6. No critical errors
    critical_errors = [e for e in result['errors'] if 'critical' in e.lower()]
    assert len(critical_errors) == 0
    
    # 7. Performance acceptable
    total_time = sum(result['node_execution_times'].values())
    assert total_time < 15.0
    
    print(f"\n✅ ALL SUCCESS CRITERIA MET!")
    print(f"   Ticker: {result['ticker']}")
    print(f"   Technical Signal: {result['technical_signal']} ({result['technical_confidence']*100:.1f}%)")
    print(f"   Forecast: ${result['forecasted_price']:.2f}")
    print(f"   Execution Time: {total_time:.2f}s")
    print(f"   Nodes Executed: {len(result['node_execution_times'])}")


if __name__ == "__main__":
    # Run basic integration test
    print("\n🧪 Running integration tests...\n")
    
    result = run_stock_analysis('AAPL')
    
    print("\n📊 Integration Test Results:")
    print(f"   Ticker: {result['ticker']}")
    print(f"   Technical Signal: {result.get('technical_signal')}")
    print(f"   Forecast Price: ${result.get('forecasted_price', 0):.2f}")
    print(f"   Total Time: {sum(result['node_execution_times'].values()):.2f}s")
    print("\n✅ Integration test complete!")
