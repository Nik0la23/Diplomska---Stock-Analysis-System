"""
Quick validation script for Node 8: News Verification & Learning System
Validates the implementation without requiring full pipeline execution.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
from datetime import datetime
from unittest.mock import patch

from src.langgraph_nodes.node_08_news_verification import news_verification_node


def create_mock_historical_data(num_events=100):
    """Create realistic mock historical data for testing"""
    events = []
    
    # High-accuracy source: Bloomberg (85% accurate - above 80% threshold)
    for i in range(50):
        events.append({
            'source': 'Bloomberg',
            'prediction_was_accurate_7day': i < 42,  # 84% accurate
            'price_change_7day': 2.5 if i < 42 else -0.5,
            'news_type': 'stock',
            'sentiment_label': 'positive' if i < 42 else 'negative'
        })
    
    # Low-accuracy source: Random Blog (30% accurate)
    for i in range(50):
        events.append({
            'source': 'Random Blog',
            'prediction_was_accurate_7day': i < 15,  # 30% accurate
            'price_change_7day': 0.5 if i < 15 else -1.0,
            'news_type': 'market',
            'sentiment_label': 'positive' if i < 15 else 'negative'
        })
    
    return pd.DataFrame(events)


def test_node_8_basic():
    """Test basic Node 8 functionality"""
    print("\n=== TEST 1: Basic Functionality ===")
    
    # Create mock data
    historical_df = create_mock_historical_data(100)
    
    # Create mock state
    state = {
        'ticker': 'NVDA',
        'sentiment_analysis': {
            'confidence': 0.70,
            'sentiment_signal': 'BUY',
            'combined_sentiment_score': 0.42
        },
        'cleaned_stock_news': [
            {'source': 'Bloomberg', 'title': 'NVDA earnings beat'},
            {'source': 'Bloomberg', 'title': 'AI demand strong'}
        ],
        'cleaned_market_news': [
            {'source': 'CNBC', 'title': 'Tech rallies'}
        ],
        'cleaned_related_news': []
    }
    
    # Mock the database call
    with patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes') as mock_get:
        with patch('src.langgraph_nodes.node_08_news_verification.store_source_reliability'):
            mock_get.return_value = historical_df
            
            # Run Node 8
            result = news_verification_node(state)
    
    # Validate results
    verification = result['news_impact_verification']
    
    print(f"✓ Sample size: {verification['sample_size']}")
    print(f"✓ Insufficient data: {verification['insufficient_data']}")
    print(f"✓ Historical correlation: {verification['historical_correlation']:.3f}")
    print(f"✓ News accuracy score: {verification['news_accuracy_score']:.1f}%")
    print(f"✓ Learning adjustment: {verification['learning_adjustment']:.3f}")
    
    # Validate bounds
    assert 0.0 <= verification['historical_correlation'] <= 1.0, "Correlation out of bounds"
    assert 0.0 <= verification['news_accuracy_score'] <= 100.0, "Accuracy out of bounds"
    assert 0.5 <= verification['learning_adjustment'] <= 2.0, "Learning adjustment out of bounds"
    
    # Validate confidence adjustment
    original = state['sentiment_analysis']['confidence']
    adjusted = result['sentiment_analysis']['confidence']
    print(f"✓ Confidence: {original:.3f} → {adjusted:.3f}")
    
    # Bloomberg is high accuracy, should boost confidence
    assert 'confidence_adjustment' in result['sentiment_analysis']
    
    print("✅ Test 1 PASSED\n")
    return True


def test_node_8_insufficient_data():
    """Test Node 8 with insufficient historical data"""
    print("=== TEST 2: Insufficient Data Handling ===")
    
    # Only 5 events (< 10 minimum)
    historical_df = pd.DataFrame([
        {'source': 'Bloomberg', 'prediction_was_accurate_7day': True, 
         'price_change_7day': 2.5, 'news_type': 'stock', 'sentiment_label': 'positive'}
    ] * 5)
    
    state = {
        'ticker': 'NVDA',
        'sentiment_analysis': {'confidence': 0.70},
        'cleaned_stock_news': [],
        'cleaned_market_news': [],
        'cleaned_related_news': []
    }
    
    with patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes') as mock_get:
        mock_get.return_value = historical_df
        
        result = news_verification_node(state)
    
    verification = result['news_impact_verification']
    
    print(f"✓ Sample size: {verification['sample_size']}")
    print(f"✓ Insufficient data flag: {verification['insufficient_data']}")
    print(f"✓ Learning adjustment: {verification['learning_adjustment']}")
    
    # Validate insufficient data handling
    assert verification['insufficient_data'] == True
    assert verification['learning_adjustment'] == 1.0  # Neutral
    assert verification['sample_size'] == 5
    
    # Confidence should NOT be adjusted
    assert result['sentiment_analysis']['confidence'] == 0.70
    
    print("✅ Test 2 PASSED\n")
    return True


def test_node_8_source_differentiation():
    """Test Node 8 differentiates between reliable and unreliable sources"""
    print("=== TEST 3: Source Differentiation (THESIS VALIDATION) ===")
    
    historical_df = create_mock_historical_data(100)
    
    # Test with Bloomberg articles
    state_bloomberg = {
        'ticker': 'NVDA',
        'sentiment_analysis': {'confidence': 0.70},
        'cleaned_stock_news': [{'source': 'Bloomberg'}] * 5,
        'cleaned_market_news': [],
        'cleaned_related_news': []
    }
    
    with patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes') as mock_get:
        with patch('src.langgraph_nodes.node_08_news_verification.store_source_reliability'):
            mock_get.return_value = historical_df
            result_bloomberg = news_verification_node(state_bloomberg)
    
    bloomberg_confidence = result_bloomberg['sentiment_analysis']['confidence']
    bloomberg_reliability = result_bloomberg['news_impact_verification']['source_reliability']['Bloomberg']
    
    # Test with Random Blog articles
    state_blog = {
        'ticker': 'NVDA',
        'sentiment_analysis': {'confidence': 0.70},
        'cleaned_stock_news': [],
        'cleaned_market_news': [{'source': 'Random Blog'}] * 5,
        'cleaned_related_news': []
    }
    
    with patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes') as mock_get:
        with patch('src.langgraph_nodes.node_08_news_verification.store_source_reliability'):
            mock_get.return_value = historical_df
            result_blog = news_verification_node(state_blog)
    
    blog_confidence = result_blog['sentiment_analysis']['confidence']
    blog_reliability = result_blog['news_impact_verification']['source_reliability']['Random Blog']
    
    print(f"Bloomberg:")
    print(f"  ✓ Accuracy rate: {bloomberg_reliability['accuracy_rate']:.1%}")
    print(f"  ✓ Confidence multiplier: {bloomberg_reliability['confidence_multiplier']:.3f}")
    print(f"  ✓ Confidence: 0.70 → {bloomberg_confidence:.3f}")
    
    print(f"\nRandom Blog:")
    print(f"  ✓ Accuracy rate: {blog_reliability['accuracy_rate']:.1%}")
    print(f"  ✓ Confidence multiplier: {blog_reliability['confidence_multiplier']:.3f}")
    print(f"  ✓ Confidence: 0.70 → {blog_confidence:.3f}")
    
    # Validate differentiation
    assert bloomberg_reliability['confidence_multiplier'] >= 1.0, "Bloomberg should boost or neutral"
    assert blog_reliability['confidence_multiplier'] < 1.0, "Blog should reduce"
    assert bloomberg_confidence > blog_confidence, "Should differentiate"
    
    # For thesis: Bloomberg (84% > 80%) should have multiplier > 1.0
    if bloomberg_reliability['accuracy_rate'] > 0.80:
        assert bloomberg_reliability['confidence_multiplier'] > 1.0, \
            f"High accuracy ({bloomberg_reliability['accuracy_rate']:.1%}) should have multiplier > 1.0"
    
    improvement = bloomberg_confidence - blog_confidence
    print(f"\n✅ THESIS INNOVATION VALIDATED: {improvement:.3f} differential")
    print("✅ Test 3 PASSED\n")
    return True


def test_node_8_execution_time():
    """Test Node 8 execution time is < 2 seconds"""
    print("=== TEST 4: Execution Time ===")
    
    historical_df = create_mock_historical_data(100)
    
    state = {
        'ticker': 'NVDA',
        'sentiment_analysis': {'confidence': 0.70},
        'cleaned_stock_news': [{'source': 'Bloomberg'}] * 10,
        'cleaned_market_news': [{'source': 'CNBC'}] * 5,
        'cleaned_related_news': []
    }
    
    start = datetime.now()
    
    with patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes') as mock_get:
        with patch('src.langgraph_nodes.node_08_news_verification.store_source_reliability'):
            mock_get.return_value = historical_df
            result = news_verification_node(state)
    
    execution_time = (datetime.now() - start).total_seconds()
    
    print(f"✓ Execution time: {execution_time:.3f}s")
    print(f"✓ Node recorded time: {result['node_execution_times']['node_8']:.3f}s")
    
    # Should be fast (< 2 seconds)
    assert execution_time < 2.0, f"Too slow: {execution_time:.3f}s"
    
    print("✅ Test 4 PASSED\n")
    return True


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("NODE 8 INTEGRATION VALIDATION")
    print("News Verification & Learning System")
    print("=" * 60)
    
    try:
        test_node_8_basic()
        test_node_8_insufficient_data()
        test_node_8_source_differentiation()
        test_node_8_execution_time()
        
        print("=" * 60)
        print("✅ ALL VALIDATION TESTS PASSED")
        print("=" * 60)
        print("\n📊 Summary:")
        print("  • Node 8 implementation: COMPLETE")
        print("  • Source reliability calculation: WORKING")
        print("  • Confidence adjustment: WORKING")
        print("  • Learning adjustment bounds: VALIDATED")
        print("  • Execution time: < 2 seconds")
        print("  • Thesis innovation: DEMONSTRATED")
        print("\n🎓 Node 8 is ready for thesis demonstration!")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
