"""
Tests for Background Script: News Outcomes Evaluator

Tests cover:
1. Helper functions (4 tests)
2. Single article evaluation (6 tests)
3. Batch evaluation (3 tests)
4. Edge cases (5 tests)
5. db_manager functions (4 tests)

Total: 22+ tests
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import background script functions
from scripts.update_news_outcomes import (
    determine_predicted_direction,
    determine_actual_direction,
    evaluate_single_article,
    run_evaluation
)

# Import db_manager functions being tested
from src.database.db_manager import (
    get_news_outcomes_pending,
    get_price_on_date,
    get_price_after_days,
    save_news_outcome
)


# ============================================================================
# TEST GROUP 1: HELPER FUNCTIONS
# ============================================================================

class TestHelperFunctions:
    """Test utility functions"""
    
    def test_determine_predicted_direction_alpha_vantage(self):
        """Test with Alpha Vantage label formats"""
        assert determine_predicted_direction('Bullish') == 'UP'
        assert determine_predicted_direction('Somewhat-Bullish') == 'UP'
        assert determine_predicted_direction('Bearish') == 'DOWN'
        assert determine_predicted_direction('Somewhat-Bearish') == 'DOWN'
        assert determine_predicted_direction('Neutral') == 'FLAT'
    
    def test_determine_predicted_direction_finbert(self):
        """Test with FinBERT label formats"""
        assert determine_predicted_direction('positive') == 'UP'
        assert determine_predicted_direction('negative') == 'DOWN'
        assert determine_predicted_direction('neutral') == 'FLAT'
    
    def test_determine_predicted_direction_edge_cases(self):
        """Test edge cases"""
        assert determine_predicted_direction(None) == 'FLAT'
        assert determine_predicted_direction('') == 'FLAT'
        assert determine_predicted_direction('unknown') == 'FLAT'
    
    def test_determine_actual_direction(self):
        """Test actual direction calculation"""
        # Clear movements
        assert determine_actual_direction(3.5) == 'UP'
        assert determine_actual_direction(-2.5) == 'DOWN'
        
        # Near threshold (0.5%)
        assert determine_actual_direction(0.6) == 'UP'
        assert determine_actual_direction(-0.6) == 'DOWN'
        assert determine_actual_direction(0.3) == 'FLAT'
        assert determine_actual_direction(-0.3) == 'FLAT'
        
        # Exact threshold
        assert determine_actual_direction(0.5) == 'FLAT'  # Not > 0.5
        assert determine_actual_direction(-0.5) == 'FLAT'  # Not < -0.5


# ============================================================================
# TEST GROUP 2: SINGLE ARTICLE EVALUATION
# ============================================================================

class TestSingleArticleEvaluation:
    """Test evaluate_single_article function"""
    
    @patch('scripts.update_news_outcomes.get_price_on_date')
    @patch('scripts.update_news_outcomes.get_price_after_days')
    def test_accurate_positive_prediction(self, mock_price_after, mock_price_on):
        """Test article with positive sentiment and price went up"""
        # Mock price data
        mock_price_on.return_value = 100.0  # Price at news
        mock_price_after.side_effect = [
            (101.0, '2025-09-16'),  # 1 day: +1%
            (103.0, '2025-09-18'),  # 3 days: +3%
            (105.0, '2025-09-22')   # 7 days: +5%
        ]
        
        article = {
            'id': 123,
            'ticker': 'NVDA',
            'published_at': '2025-09-15',
            'sentiment_label': 'positive'
        }
        
        outcome = evaluate_single_article(article)
        
        assert outcome is not None
        assert outcome['news_id'] == 123
        assert outcome['price_at_news'] == 100.0
        assert outcome['price_7day_later'] == 105.0
        assert outcome['price_change_7day'] == 5.0
        assert outcome['predicted_direction'] == 'UP'
        assert outcome['actual_direction'] == 'UP'
        assert outcome['prediction_was_accurate_7day'] == True
    
    @patch('scripts.update_news_outcomes.get_price_on_date')
    @patch('scripts.update_news_outcomes.get_price_after_days')
    def test_inaccurate_prediction(self, mock_price_after, mock_price_on):
        """Test article with positive sentiment but price went down"""
        mock_price_on.return_value = 100.0
        mock_price_after.side_effect = [
            (99.0, '2025-09-16'),   # 1 day: -1%
            (97.0, '2025-09-18'),   # 3 days: -3%
            (95.0, '2025-09-22')    # 7 days: -5%
        ]
        
        article = {
            'id': 124,
            'ticker': 'NVDA',
            'published_at': '2025-09-15',
            'sentiment_label': 'positive'
        }
        
        outcome = evaluate_single_article(article)
        
        assert outcome['predicted_direction'] == 'UP'
        assert outcome['actual_direction'] == 'DOWN'
        assert outcome['prediction_was_accurate_7day'] == False  # Incorrect
    
    @patch('scripts.update_news_outcomes.get_price_on_date')
    @patch('scripts.update_news_outcomes.get_price_after_days')
    def test_neutral_sentiment(self, mock_price_after, mock_price_on):
        """Test article with neutral sentiment"""
        mock_price_on.return_value = 100.0
        mock_price_after.side_effect = [
            (100.2, '2025-09-16'),  # Minimal change
            (100.3, '2025-09-18'),
            (100.1, '2025-09-22')
        ]
        
        article = {
            'id': 125,
            'ticker': 'NVDA',
            'published_at': '2025-09-15',
            'sentiment_label': 'neutral'
        }
        
        outcome = evaluate_single_article(article)
        
        assert outcome['predicted_direction'] == 'FLAT'
        assert outcome['actual_direction'] == 'FLAT'
        assert outcome['prediction_was_accurate_7day'] == True
    
    @patch('scripts.update_news_outcomes.get_price_on_date')
    def test_missing_price_at_news(self, mock_price_on):
        """Test article with no price data at publication time"""
        mock_price_on.return_value = None  # No price data
        
        article = {
            'id': 126,
            'ticker': 'NVDA',
            'published_at': '2025-09-15',
            'sentiment_label': 'positive'
        }
        
        outcome = evaluate_single_article(article)
        
        # Should return None (skip article)
        assert outcome is None
    
    @patch('scripts.update_news_outcomes.get_price_on_date')
    @patch('scripts.update_news_outcomes.get_price_after_days')
    def test_missing_7day_price(self, mock_price_after, mock_price_on):
        """Test article with no 7-day price data"""
        mock_price_on.return_value = 100.0
        mock_price_after.side_effect = [
            (101.0, '2025-09-16'),  # 1 day
            (103.0, '2025-09-18'),  # 3 days
            (None, None)             # 7 days - missing!
        ]
        
        article = {
            'id': 127,
            'ticker': 'NVDA',
            'published_at': '2025-09-15',
            'sentiment_label': 'positive'
        }
        
        outcome = evaluate_single_article(article)
        
        # Should return None (7-day price is required)
        assert outcome is None
    
    def test_invalid_date_format(self):
        """Test article with invalid date format"""
        article = {
            'id': 128,
            'ticker': 'NVDA',
            'published_at': 'invalid-date',
            'sentiment_label': 'positive'
        }
        
        outcome = evaluate_single_article(article)
        
        # Should return None (can't parse date)
        assert outcome is None


# ============================================================================
# TEST GROUP 3: BATCH EVALUATION
# ============================================================================

class TestBatchEvaluation:
    """Test run_evaluation function"""
    
    @patch('scripts.update_news_outcomes.get_news_outcomes_pending')
    def test_no_pending_articles(self, mock_pending):
        """Test when no articles need evaluation"""
        mock_pending.return_value = []
        
        results = run_evaluation(ticker='NVDA', limit=100, verbose=False)
        
        assert results['evaluated'] == 0
        assert results['skipped'] == 0
        assert results['total'] == 0
    
    @patch('scripts.update_news_outcomes.get_news_outcomes_pending')
    @patch('scripts.update_news_outcomes.get_price_on_date')
    @patch('scripts.update_news_outcomes.get_price_after_days')
    @patch('scripts.update_news_outcomes.save_news_outcome')
    def test_successful_batch_evaluation(self, mock_save, mock_after, mock_on, mock_pending):
        """Test successful evaluation of multiple articles"""
        # Mock 5 pending articles
        mock_pending.return_value = [
            {'id': i, 'ticker': 'NVDA', 'published_at': '2025-09-15', 'sentiment_label': 'positive'}
            for i in range(1, 6)
        ]
        
        # Mock price data (all succeed)
        mock_on.return_value = 100.0
        mock_after.side_effect = [
            (101.0, '2025-09-16'), (103.0, '2025-09-18'), (105.0, '2025-09-22'),  # Article 1
            (101.0, '2025-09-16'), (103.0, '2025-09-18'), (105.0, '2025-09-22'),  # Article 2
            (101.0, '2025-09-16'), (103.0, '2025-09-18'), (105.0, '2025-09-22'),  # Article 3
            (101.0, '2025-09-16'), (103.0, '2025-09-18'), (105.0, '2025-09-22'),  # Article 4
            (101.0, '2025-09-16'), (103.0, '2025-09-18'), (105.0, '2025-09-22'),  # Article 5
        ]
        
        results = run_evaluation(ticker='NVDA', limit=100, verbose=False)
        
        assert results['evaluated'] == 5
        assert results['skipped'] == 0
        assert results['accurate'] == 5  # All positive predictions, price went up
        assert results['accuracy_pct'] == 100.0
    
    @patch('scripts.update_news_outcomes.get_news_outcomes_pending')
    @patch('scripts.update_news_outcomes.get_price_on_date')
    @patch('scripts.update_news_outcomes.get_price_after_days')
    @patch('scripts.update_news_outcomes.save_news_outcome')
    def test_mixed_accuracy(self, mock_save, mock_after, mock_on, mock_pending):
        """Test evaluation with mixed correct/incorrect predictions"""
        mock_pending.return_value = [
            {'id': 1, 'ticker': 'NVDA', 'published_at': '2025-09-15', 'sentiment_label': 'positive'},
            {'id': 2, 'ticker': 'NVDA', 'published_at': '2025-09-15', 'sentiment_label': 'positive'},
        ]
        
        # Article 1: positive → price up (correct)
        # Article 2: positive → price down (incorrect)
        mock_on.return_value = 100.0
        mock_after.side_effect = [
            (101.0, '2025-09-16'), (103.0, '2025-09-18'), (105.0, '2025-09-22'),  # Article 1: +5%
            (99.0, '2025-09-16'), (97.0, '2025-09-18'), (95.0, '2025-09-22'),     # Article 2: -5%
        ]
        
        results = run_evaluation(ticker='NVDA', limit=100, verbose=False)
        
        assert results['evaluated'] == 2
        assert results['accurate'] == 1  # Only 1 correct
        assert results['accuracy_pct'] == 50.0


# ============================================================================
# TEST GROUP 4: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @patch('scripts.update_news_outcomes.get_price_on_date')
    @patch('scripts.update_news_outcomes.get_price_after_days')
    def test_weekend_publication(self, mock_after, mock_on):
        """Test article published on weekend (should find Friday's price)"""
        # Article published Saturday 2025-09-13
        mock_on.return_value = 100.0  # Should find Friday's price
        mock_after.side_effect = [
            (101.0, '2025-09-15'),  # Monday (1 day later)
            (103.0, '2025-09-17'),  # Wednesday (3 days later)
            (105.0, '2025-09-22')   # Monday (7 days later)
        ]
        
        article = {
            'id': 130,
            'ticker': 'NVDA',
            'published_at': '2025-09-13',  # Saturday
            'sentiment_label': 'positive'
        }
        
        outcome = evaluate_single_article(article)
        
        assert outcome is not None
        assert outcome['price_at_news'] == 100.0
    
    @patch('scripts.update_news_outcomes.get_price_on_date')
    @patch('scripts.update_news_outcomes.get_price_after_days')
    def test_minimal_price_change(self, mock_after, mock_on):
        """Test with minimal price change (within ±0.5% threshold)"""
        mock_on.return_value = 100.0
        mock_after.side_effect = [
            (100.2, '2025-09-16'),  # +0.2%
            (100.3, '2025-09-18'),  # +0.3%
            (100.4, '2025-09-22')   # +0.4% (< 0.5% threshold)
        ]
        
        article = {
            'id': 131,
            'ticker': 'NVDA',
            'published_at': '2025-09-15',
            'sentiment_label': 'neutral'
        }
        
        outcome = evaluate_single_article(article)
        
        assert outcome['actual_direction'] == 'FLAT'
        assert outcome['predicted_direction'] == 'FLAT'
        assert outcome['prediction_was_accurate_7day'] == True
    
    @patch('scripts.update_news_outcomes.get_price_on_date')
    @patch('scripts.update_news_outcomes.get_price_after_days')
    def test_missing_intermediate_prices(self, mock_after, mock_on):
        """Test when 1-day and 3-day prices are missing but 7-day exists"""
        mock_on.return_value = 100.0
        mock_after.side_effect = [
            (None, None),           # 1 day - missing
            (None, None),           # 3 days - missing
            (105.0, '2025-09-22')   # 7 days - available
        ]
        
        article = {
            'id': 132,
            'ticker': 'NVDA',
            'published_at': '2025-09-15',
            'sentiment_label': 'positive'
        }
        
        outcome = evaluate_single_article(article)
        
        # Should still work (7-day is required, 1/3 are optional)
        assert outcome is not None
        assert outcome['price_1day_later'] is None
        assert outcome['price_3day_later'] is None
        assert outcome['price_7day_later'] == 105.0
        assert outcome['prediction_was_accurate_7day'] == True
    
    @patch('scripts.update_news_outcomes.get_price_on_date')
    @patch('scripts.update_news_outcomes.get_price_after_days')
    def test_negative_sentiment_correct(self, mock_after, mock_on):
        """Test negative sentiment with price going down (correct)"""
        mock_on.return_value = 100.0
        mock_after.side_effect = [
            (98.0, '2025-09-16'),
            (96.0, '2025-09-18'),
            (94.0, '2025-09-22')  # -6%
        ]
        
        article = {
            'id': 133,
            'ticker': 'NVDA',
            'published_at': '2025-09-15',
            'sentiment_label': 'negative'
        }
        
        outcome = evaluate_single_article(article)
        
        assert outcome['predicted_direction'] == 'DOWN'
        assert outcome['actual_direction'] == 'DOWN'
        assert outcome['prediction_was_accurate_7day'] == True
    
    @patch('scripts.update_news_outcomes.get_price_on_date')
    @patch('scripts.update_news_outcomes.get_price_after_days')
    def test_divergence(self, mock_after, mock_on):
        """Test sentiment-price divergence (positive news, price drops)"""
        mock_on.return_value = 100.0
        mock_after.side_effect = [
            (98.0, '2025-09-16'),
            (96.0, '2025-09-18'),
            (94.0, '2025-09-22')  # -6%
        ]
        
        article = {
            'id': 134,
            'ticker': 'NVDA',
            'published_at': '2025-09-15',
            'sentiment_label': 'Bullish'  # Alpha Vantage format
        }
        
        outcome = evaluate_single_article(article)
        
        assert outcome['predicted_direction'] == 'UP'
        assert outcome['actual_direction'] == 'DOWN'
        assert outcome['prediction_was_accurate_7day'] == False


# ============================================================================
# TEST GROUP 5: DB_MANAGER FUNCTIONS
# ============================================================================

class TestDbManagerFunctions:
    """Test new db_manager functions (integration tests with mock DB)"""
    
    @patch('src.database.db_manager.get_connection')
    def test_get_news_outcomes_pending(self, mock_conn):
        """Test get_news_outcomes_pending function"""
        # Mock cursor result
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                'id': 1,
                'ticker': 'NVDA',
                'published_at': '2025-09-01',
                'sentiment_label': 'positive',
                'sentiment_score': 0.8
            }
        ]
        
        mock_connection = MagicMock()
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_connection
        
        results = get_news_outcomes_pending(ticker='NVDA', limit=100)
        
        # Should call the query and return results
        assert isinstance(results, list)
    
    @patch('src.database.db_manager.get_connection')
    def test_get_price_on_date(self, mock_conn):
        """Test get_price_on_date function"""
        # Mock cursor result
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {'close': 135.50}
        
        mock_connection = MagicMock()
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_connection
        
        price = get_price_on_date('NVDA', '2025-09-15')
        
        assert price == 135.50
    
    @patch('src.database.db_manager.get_connection')
    def test_get_price_after_days(self, mock_conn):
        """Test get_price_after_days function"""
        # Mock cursor result
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {'close': 139.50, 'date': '2025-09-22'}
        
        mock_connection = MagicMock()
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_connection
        
        price, date_str = get_price_after_days('NVDA', '2025-09-15', 7)
        
        assert price == 139.50
        assert date_str == '2025-09-22'
    
    @patch('src.database.db_manager.get_connection')
    def test_save_news_outcome(self, mock_conn):
        """Test save_news_outcome function"""
        mock_connection = MagicMock()
        mock_connection.__enter__.return_value = mock_connection
        mock_conn.return_value = mock_connection
        
        outcome = {
            'news_id': 123,
            'ticker': 'NVDA',
            'price_at_news': 100.0,
            'price_7day_later': 105.0,
            'price_change_7day': 5.0,
            'predicted_direction': 'UP',
            'actual_direction': 'UP',
            'prediction_was_accurate_7day': True
        }
        
        # Should not raise exception
        save_news_outcome(outcome)
        
        # Should have executed INSERT
        assert mock_connection.execute.called


# ============================================================================
# TEST GROUP 6: INTEGRATION SCENARIOS
# ============================================================================

class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @patch('scripts.update_news_outcomes.get_news_outcomes_pending')
    @patch('scripts.update_news_outcomes.get_price_on_date')
    @patch('scripts.update_news_outcomes.get_price_after_days')
    @patch('scripts.update_news_outcomes.save_news_outcome')
    def test_realistic_batch_with_skips(self, mock_save, mock_after, mock_on, mock_pending):
        """Test realistic batch with some articles skipped due to missing prices"""
        # Mock 10 pending articles
        mock_pending.return_value = [
            {'id': i, 'ticker': 'NVDA', 'published_at': '2025-09-15', 'sentiment_label': 'positive'}
            for i in range(1, 11)
        ]
        
        # Mock: First 7 articles have full data, last 3 missing 7-day price
        def get_price_side_effect(ticker, date):
            return 100.0  # All have initial price
        
        after_call_count = [0]
        def get_after_side_effect(ticker, date, days):
            after_call_count[0] += 1
            call_num = after_call_count[0]
            
            # First 7 articles: full data
            if call_num <= 21:  # 7 articles × 3 calls each
                if days == 1:
                    return (101.0, '2025-09-16')
                elif days == 3:
                    return (103.0, '2025-09-18')
                else:  # 7 days
                    return (105.0, '2025-09-22')
            # Last 3 articles: missing 7-day data
            else:
                if days == 7:
                    return (None, None)  # Missing!
                return (101.0, '2025-09-16')
        
        mock_on.side_effect = [100.0] * 10
        mock_after.side_effect = get_after_side_effect
        
        results = run_evaluation(ticker='NVDA', limit=100, verbose=False)
        
        # Should evaluate 7, skip 3
        assert results['evaluated'] == 7
        assert results['skipped'] == 3
        assert results['total'] == 10
