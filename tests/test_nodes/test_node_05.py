"""
Tests for Node 5: Sentiment Analysis

Tests cover the refactored node output philosophy:
- Alpha Vantage labels are preserved as-is (not recalculated)
- FinBERT only fills in labels for articles that have no AV sentiment
- No BUY/SELL/HOLD signal generation — that is Node 12's job
- sentiment_signal is now POSITIVE/NEGATIVE/NEUTRAL (direction only)
- sentiment_breakdown is the rich narrative dict for Node 13

Categories:
1. Alpha Vantage sentiment extraction (5 tests)
2. FinBERT model loading (3 tests)
3. FinBERT text analysis (5 tests)
4. Sentiment aggregation per stream (6 tests)
5. Combined sentiment calculation (4 tests)
6. Sentiment breakdown for Node 13 (4 tests)
7. Credibility weight calculation (4 tests)
8. Main node function (5 tests)
9. Integration (3 tests)

Total: 39 tests
"""

import sys
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.langgraph_nodes.node_05_sentiment_analysis import (
    extract_alpha_vantage_sentiment,
    load_finbert_model,
    analyze_text_with_finbert,
    analyze_articles_batch,
    aggregate_sentiment_by_type,
    calculate_combined_sentiment,
    build_sentiment_breakdown,
    calculate_credibility_weight,
    sentiment_analysis_node,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_av_articles():
    """Articles with Alpha Vantage sentiment scores and labels."""
    return [
        {
            'title': 'Company beats earnings expectations',
            'summary': 'Strong quarterly results',
            'overall_sentiment_score': 0.7,
            'overall_sentiment_label': 'Bullish',
            'ticker_sentiment_score': 0.8,
            'ticker_relevance_score': 0.9,
            'source': 'Bloomberg',
        },
        {
            'title': 'Market concerns over regulations',
            'summary': 'New regulations may impact growth',
            'overall_sentiment_score': -0.5,
            'overall_sentiment_label': 'Bearish',
            'ticker_sentiment_score': -0.6,
            'ticker_relevance_score': 0.7,
            'source': 'Reuters',
        },
        {
            'title': 'Company announces new product',
            'summary': 'Neutral market reception',
            'overall_sentiment_score': 0.1,
            'overall_sentiment_label': 'Neutral',
            'ticker_sentiment_score': 0.05,
            'ticker_relevance_score': 0.5,
            'source': 'CNBC',
        },
    ]


@pytest.fixture
def sample_mixed_articles():
    """Mix of Alpha Vantage and Finnhub articles (no AV sentiment)."""
    return [
        {
            'title': 'Good news',
            'overall_sentiment_label': 'Somewhat-Bullish',
            'ticker_sentiment_score': 0.7,
            'source': 'Alpha Vantage',
        },
        {
            'title': 'Neutral news from Finnhub',
            'source': 'Finnhub',
            # No sentiment fields — FinBERT fallback
        },
        {
            'title': 'Bad news',
            'overall_sentiment_label': 'Bearish',
            'ticker_sentiment_score': -0.6,
            'source': 'Alpha Vantage',
        },
    ]


@pytest.fixture
def base_state():
    """Minimal state for main node tests."""
    return {
        'ticker': 'AAPL',
        'cleaned_stock_news': [
            {
                'title': 'Positive stock news',
                'overall_sentiment_label': 'Bullish',
                'ticker_sentiment_score': 0.8,
                'source': 'Bloomberg',
                'source_credibility_score': 0.95,
                'composite_anomaly_score': 0.04,
                'relevance_score': 0.9,
            },
            {
                'title': 'More positive news',
                'overall_sentiment_label': 'Somewhat-Bullish',
                'ticker_sentiment_score': 0.7,
                'source': 'Reuters',
                'source_credibility_score': 0.85,
                'composite_anomaly_score': 0.10,
                'relevance_score': 0.8,
            },
        ],
        'cleaned_market_news': [
            {
                'title': 'Market bullish',
                'overall_sentiment_label': 'Somewhat-Bullish',
                'ticker_sentiment_score': 0.5,
                'source': 'CNBC',
            }
        ],
        'cleaned_related_company_news': [
            {
                'title': 'Competitor doing well',
                'overall_sentiment_label': 'Bullish',
                'ticker_sentiment_score': 0.6,
                'source': 'Bloomberg',
            }
        ],
        'errors': [],
        'node_execution_times': {},
    }


# ============================================================================
# CATEGORY 1: Alpha Vantage Sentiment Extraction
# ============================================================================

class TestExtractAVSentiment:

    def test_ticker_sentiment_preferred_over_overall(self, sample_av_articles):
        """When USE_TICKER_SENTIMENT=True, ticker_sentiment_score is used."""
        result = extract_alpha_vantage_sentiment(sample_av_articles)

        assert len(result) == 3
        assert result[0]['sentiment_score'] == 0.8   # ticker score
        assert result[0]['sentiment_source'] == 'alpha_vantage_ticker'
        assert result[0]['has_sentiment'] is True

    def test_av_label_preserved_not_recalculated(self, sample_av_articles):
        """The overall_sentiment_label from AV is kept as-is (e.g. 'Bullish')."""
        result = extract_alpha_vantage_sentiment(sample_av_articles)

        # Label should be the AV string, not recalculated 'positive'/'negative'
        assert result[0]['sentiment_label'] == 'Bullish'
        assert result[1]['sentiment_label'] == 'Bearish'
        assert result[2]['sentiment_label'] == 'Neutral'

    def test_overall_sentiment_fallback_when_no_ticker_score(self):
        """Articles without ticker_sentiment_score fall back to overall_sentiment_score."""
        articles = [
            {
                'title': 'Market news',
                'overall_sentiment_score': 0.4,
                'overall_sentiment_label': 'Somewhat-Bullish',
                'source': 'Reuters',
                # no ticker_sentiment_score
            }
        ]
        result = extract_alpha_vantage_sentiment(articles)

        assert result[0]['sentiment_score'] == pytest.approx(0.4, abs=1e-6)
        assert result[0]['sentiment_source'] == 'alpha_vantage_overall'
        assert result[0]['sentiment_label'] == 'Somewhat-Bullish'

    def test_no_av_sentiment_marks_has_sentiment_false(self):
        """Finnhub articles with no AV fields get has_sentiment=False."""
        articles = [{'title': 'Finnhub article', 'source': 'Finnhub'}]
        result = extract_alpha_vantage_sentiment(articles)

        assert result[0]['has_sentiment'] is False
        assert result[0]['sentiment_score'] == 0.0
        assert result[0]['sentiment_source'] == 'none'

    def test_malformed_or_empty_articles_do_not_crash(self):
        """Empty dicts and missing fields are handled gracefully."""
        articles = [{}, {'title': 'No scores'}]
        result = extract_alpha_vantage_sentiment(articles)
        assert len(result) == 2
        assert all('has_sentiment' in a for a in result)


# ============================================================================
# CATEGORY 2: FinBERT Model Loading
# ============================================================================

class TestFinBERTModelLoading:

    def test_cached_model_returned_without_reload(self):
        """Once loaded, the cached model is returned directly."""
        import src.langgraph_nodes.node_05_sentiment_analysis as node5
        mock_model = Mock()
        node5._FINBERT_MODEL = mock_model

        model = load_finbert_model()
        assert model is mock_model

    def test_loading_failure_returns_none(self):
        """If transformers raises, None is returned (no crash).

        Uses sys.modules patching instead of @patch('transformers.pipeline')
        because HuggingFace transformers uses lazy loading — the attribute patch
        is bypassed by the `from transformers import pipeline` import inside
        load_finbert_model(). Replacing the entire sys.modules entry intercepts
        the import correctly.
        """
        import src.langgraph_nodes.node_05_sentiment_analysis as node5
        node5._FINBERT_MODEL = None

        mock_transformers = MagicMock()
        mock_transformers.pipeline.side_effect = Exception("Download failed")

        with patch.dict(sys.modules, {'transformers': mock_transformers}):
            model = load_finbert_model()

        assert model is None
        # Reset so later tests can use real FinBERT if needed
        node5._FINBERT_MODEL = None

    def test_cached_model_is_callable(self):
        """If a model is cached it must be callable (pipeline contract)."""
        import src.langgraph_nodes.node_05_sentiment_analysis as node5
        mock_model = Mock()
        node5._FINBERT_MODEL = mock_model

        model = load_finbert_model()
        assert callable(model)


# ============================================================================
# CATEGORY 3: FinBERT Text Analysis
# ============================================================================

class TestFinBERTTextAnalysis:

    def _mock_positive(self):
        m = Mock()
        m.return_value = [[
            {'label': 'positive', 'score': 0.95},
            {'label': 'neutral',  'score': 0.03},
            {'label': 'negative', 'score': 0.02},
        ]]
        return m

    def _mock_negative(self):
        m = Mock()
        m.return_value = [[
            {'label': 'negative', 'score': 0.90},
            {'label': 'neutral',  'score': 0.07},
            {'label': 'positive', 'score': 0.03},
        ]]
        return m

    def _mock_neutral(self):
        m = Mock()
        m.return_value = [[
            {'label': 'neutral',  'score': 0.80},
            {'label': 'positive', 'score': 0.10},
            {'label': 'negative', 'score': 0.10},
        ]]
        return m

    def test_positive_text_returns_positive_label_and_score(self):
        result = analyze_text_with_finbert("Company beats earnings", self._mock_positive())
        assert result['label'] == 'positive'
        assert result['sentiment'] > 0.0
        assert 0.0 <= result['score'] <= 1.0

    def test_negative_text_returns_negative_score(self):
        result = analyze_text_with_finbert("Company misses earnings", self._mock_negative())
        assert result['label'] == 'negative'
        assert result['sentiment'] < 0.0

    def test_neutral_text_returns_zero_sentiment(self):
        result = analyze_text_with_finbert("Company releases a statement", self._mock_neutral())
        assert result['label'] == 'neutral'
        assert result['sentiment'] == 0.0

    def test_empty_text_returns_neutral_defaults(self):
        result = analyze_text_with_finbert("", Mock())
        assert result['label'] == 'neutral'
        assert result['sentiment'] == 0.0

    def test_model_none_returns_neutral_defaults(self):
        result = analyze_text_with_finbert("Some text", None)
        assert result['label'] == 'neutral'
        assert result['sentiment'] == 0.0


# ============================================================================
# CATEGORY 4: Sentiment Aggregation per Stream
# ============================================================================

class TestAggregateByType:

    def test_empty_articles_return_zero_defaults(self):
        result = aggregate_sentiment_by_type([], 'stock')
        assert result['article_count'] == 0
        assert result['weighted_sentiment'] == 0.0
        assert result['dominant_label'] == 'neutral'
        assert result['top_articles'] == []
        assert result['confidence'] == 0.0

    def test_no_buy_sell_hold_in_output(self):
        """aggregate_sentiment_by_type must NOT return a sentiment_signal."""
        articles = [
            {'sentiment_score': 0.8, 'sentiment_label': 'Bullish', 'title': 'T', 'source': 'X'},
        ]
        result = aggregate_sentiment_by_type(articles, 'stock')
        assert 'sentiment_signal' not in result

    def test_dominant_label_reflects_majority(self):
        """dominant_label should be the most frequent normalised label."""
        articles = [
            {'sentiment_score': 0.8, 'sentiment_label': 'Bullish',          'title': 'A', 'source': 'S'},
            {'sentiment_score': 0.7, 'sentiment_label': 'Somewhat-Bullish', 'title': 'B', 'source': 'S'},
            {'sentiment_score': -0.3,'sentiment_label': 'Bearish',          'title': 'C', 'source': 'S'},
        ]
        result = aggregate_sentiment_by_type(articles, 'stock')
        assert result['dominant_label'] == 'positive'

    def test_top_articles_are_top_3_by_credibility(self):
        """top_articles should be the 3 highest-credibility articles."""
        articles = [
            {'sentiment_score': 0.5, 'sentiment_label': 'Bullish', 'title': 'Low',
             'source': 'Blog', 'source_credibility_score': 0.3, 'composite_anomaly_score': 0.8, 'relevance_score': 0.3},
            {'sentiment_score': 0.7, 'sentiment_label': 'Bullish', 'title': 'High',
             'source': 'Bloomberg', 'source_credibility_score': 0.95, 'composite_anomaly_score': 0.04, 'relevance_score': 0.9},
            {'sentiment_score': 0.6, 'sentiment_label': 'Somewhat-Bullish', 'title': 'Medium',
             'source': 'Reuters', 'source_credibility_score': 0.75, 'composite_anomaly_score': 0.15, 'relevance_score': 0.7},
        ]
        result = aggregate_sentiment_by_type(articles, 'stock')
        assert len(result['top_articles']) <= 3
        # Bloomberg should be first (highest credibility)
        assert result['top_articles'][0]['source'] == 'Bloomberg'

    def test_time_weighted_sentiment_differs_from_simple_mean(self):
        """Recent article (index 0) has higher weight — weighted != simple mean."""
        articles = [
            {'sentiment_score': -0.8, 'sentiment_label': 'Bearish', 'title': 'T', 'source': 'S'},  # newest
            {'sentiment_score': 0.4,  'sentiment_label': 'Bullish', 'title': 'T', 'source': 'S'},
            {'sentiment_score': 0.4,  'sentiment_label': 'Bullish', 'title': 'T', 'source': 'S'},  # oldest
        ]
        result = aggregate_sentiment_by_type(articles, 'stock')
        assert result['weighted_sentiment'] < result['average_sentiment']

    def test_positive_and_negative_counts_are_correct(self):
        articles = [
            {'sentiment_score': 0.8, 'sentiment_label': 'Bullish',  'title': 'T', 'source': 'S'},
            {'sentiment_score': -0.5,'sentiment_label': 'Bearish',  'title': 'T', 'source': 'S'},
            {'sentiment_score': 0.1, 'sentiment_label': 'Neutral',  'title': 'T', 'source': 'S'},
        ]
        result = aggregate_sentiment_by_type(articles, 'stock')
        assert result['positive_count'] == 1
        assert result['negative_count'] == 1
        assert result['neutral_count'] == 1


# ============================================================================
# CATEGORY 5: Combined Sentiment Calculation
# ============================================================================

class TestCalculateCombinedSentiment:

    def _make_agg(self, ws: float, conf: float) -> dict:
        return {'weighted_sentiment': ws, 'confidence': conf,
                'article_count': 5, 'dominant_label': 'positive',
                'positive_count': 5, 'negative_count': 0, 'neutral_count': 0,
                'top_articles': []}

    def test_weighted_50_25_25_calculation(self):
        """combined = 0.8*0.5 + 0.4*0.25 + 0.6*0.25 = 0.65"""
        result = calculate_combined_sentiment(
            self._make_agg(0.8, 0.9),
            self._make_agg(0.4, 0.7),
            self._make_agg(0.6, 0.8),
        )
        assert abs(result['combined_sentiment'] - 0.65) < 0.01

    def test_no_signal_field_in_output(self):
        """calculate_combined_sentiment must NOT return sentiment_signal."""
        result = calculate_combined_sentiment(
            self._make_agg(0.5, 0.7),
            self._make_agg(0.3, 0.6),
            self._make_agg(0.4, 0.6),
        )
        assert 'sentiment_signal' not in result

    def test_negative_combined_sentiment(self):
        result = calculate_combined_sentiment(
            self._make_agg(-0.7, 0.8),
            self._make_agg(-0.5, 0.7),
            self._make_agg(-0.6, 0.7),
        )
        assert result['combined_sentiment'] < -0.5

    def test_confidence_is_weighted_average(self):
        """combined_confidence = 0.9*0.5 + 0.7*0.25 + 0.8*0.25 = 0.825"""
        result = calculate_combined_sentiment(
            self._make_agg(0.5, 0.9),
            self._make_agg(0.5, 0.7),
            self._make_agg(0.5, 0.8),
        )
        assert abs(result['confidence'] - 0.825) < 0.01


# ============================================================================
# CATEGORY 6: Build Sentiment Breakdown for Node 13
# ============================================================================

class TestBuildSentimentBreakdown:

    def _stream(self, ws: float) -> dict:
        return {
            'weighted_sentiment': ws,
            'article_count': 3,
            'positive_count': 2,
            'negative_count': 1,
            'neutral_count': 0,
            'dominant_label': 'positive',
            'top_articles': [
                {'title': 'T1', 'source': 'Bloomberg',
                 'sentiment_score': 0.7, 'sentiment_label': 'Bullish', 'credibility_weight': 0.92}
            ],
        }

    def test_breakdown_has_all_three_streams_and_overall(self):
        articles = [
            {'source_credibility_score': 0.9, 'composite_anomaly_score': 0.05,
             'sentiment_source': 'alpha_vantage_ticker'},
        ]
        bd = build_sentiment_breakdown(
            self._stream(0.7), self._stream(0.3), self._stream(0.5),
            0.55, 0.72, articles,
        )
        assert 'stock' in bd
        assert 'market' in bd
        assert 'related' in bd
        assert 'overall' in bd

    def test_overall_combined_sentiment_and_confidence_correct(self):
        bd = build_sentiment_breakdown(
            self._stream(0.7), self._stream(0.3), self._stream(0.5),
            0.55, 0.72, [],
        )
        assert bd['overall']['combined_sentiment'] == pytest.approx(0.55)
        assert bd['overall']['confidence'] == pytest.approx(0.72)

    def test_credibility_block_counts_tiers(self):
        articles = [
            {'source_credibility_score': 0.90, 'composite_anomaly_score': 0.05,
             'sentiment_source': 'alpha_vantage_ticker'},   # high
            {'source_credibility_score': 0.65, 'composite_anomaly_score': 0.20,
             'sentiment_source': 'alpha_vantage_overall'},  # medium
            {'source_credibility_score': 0.30, 'composite_anomaly_score': 0.70,
             'sentiment_source': 'finbert'},                # low
        ]
        bd = build_sentiment_breakdown(
            self._stream(0.5), self._stream(0.3), self._stream(0.4),
            0.43, 0.65, articles,
        )
        cred = bd['overall']['credibility']
        assert cred['high_credibility_articles'] == 1
        assert cred['medium_credibility_articles'] == 1
        assert cred['low_credibility_articles'] == 1

    def test_sentiment_source_mix_counts_correctly(self):
        articles = [
            {'sentiment_source': 'alpha_vantage_ticker',  'source_credibility_score': 0.9, 'composite_anomaly_score': 0.05},
            {'sentiment_source': 'alpha_vantage_overall', 'source_credibility_score': 0.7, 'composite_anomaly_score': 0.10},
            {'sentiment_source': 'finbert',               'source_credibility_score': 0.5, 'composite_anomaly_score': 0.30},
            {'sentiment_source': 'none',                  'source_credibility_score': 0.4, 'composite_anomaly_score': 0.50},
        ]
        bd = build_sentiment_breakdown(
            self._stream(0.5), self._stream(0.3), self._stream(0.4),
            0.43, 0.65, articles,
        )
        mix = bd['overall']['sentiment_source_mix']
        assert mix['alpha_vantage_ticker'] == 1
        assert mix['alpha_vantage_overall'] == 1
        assert mix['finbert'] == 1
        assert mix['none'] == 1


# ============================================================================
# CATEGORY 7: Credibility Weight Calculation
# ============================================================================

class TestCredibilityWeight:

    def test_high_quality_article_gets_high_weight(self):
        article = {
            'source_credibility_score': 0.95,
            'composite_anomaly_score': 0.04,
            'relevance_score': 0.85,
        }
        weight = calculate_credibility_weight(article)
        assert weight > 0.85
        assert weight <= 1.0

    def test_low_quality_article_gets_low_weight(self):
        article = {
            'source_credibility_score': 0.30,
            'composite_anomaly_score': 0.75,
            'relevance_score': 0.40,
        }
        weight = calculate_credibility_weight(article)
        assert weight < 0.40
        assert weight >= 0.1  # floor

    def test_missing_scores_give_moderate_default(self):
        weight = calculate_credibility_weight({})
        assert 0.4 < weight < 0.7

    def test_floor_is_respected(self):
        """Weight never drops below 0.1 even for worst-case article."""
        article = {
            'source_credibility_score': 0.0,
            'composite_anomaly_score': 1.0,
            'relevance_score': 0.0,
        }
        assert calculate_credibility_weight(article) >= 0.1


# ============================================================================
# CATEGORY 8: Main Node Function
# ============================================================================

class TestSentimentAnalysisNode:

    def test_output_fields_present(self, base_state):
        result = sentiment_analysis_node(base_state)

        assert 'raw_sentiment_scores' in result
        assert 'aggregated_sentiment' in result
        assert 'sentiment_signal' in result
        assert 'sentiment_confidence' in result
        assert 'sentiment_breakdown' in result
        assert 'node_execution_times' in result
        assert 'node_5' in result['node_execution_times']

    def test_sentiment_signal_is_directional_not_trading(self, base_state):
        """sentiment_signal must be POSITIVE/NEGATIVE/NEUTRAL, never BUY/SELL/HOLD."""
        result = sentiment_analysis_node(base_state)

        assert result['sentiment_signal'] in ('POSITIVE', 'NEGATIVE', 'NEUTRAL')
        assert result['sentiment_signal'] not in ('BUY', 'SELL', 'HOLD')

    def test_no_sentiment_credibility_summary_in_output(self, base_state):
        """sentiment_credibility_summary was removed — folded into sentiment_breakdown."""
        result = sentiment_analysis_node(base_state)
        assert 'sentiment_credibility_summary' not in result

    def test_sentiment_breakdown_structure_is_complete(self, base_state):
        result = sentiment_analysis_node(base_state)
        bd = result['sentiment_breakdown']

        assert 'stock' in bd
        assert 'market' in bd
        assert 'related' in bd
        assert 'overall' in bd
        assert 'combined_sentiment' in bd['overall']
        assert 'confidence' in bd['overall']
        assert 'credibility' in bd['overall']
        assert 'sentiment_source_mix' in bd['overall']

    def test_error_handling_returns_none_fields(self):
        """Missing cleaned_*_news keys should not crash — use defaults."""
        state = {
            'ticker': 'TEST',
            # no cleaned_*_news — will default to []
            'errors': [],
            'node_execution_times': {},
        }
        result = sentiment_analysis_node(state)
        # Should complete without exception and return all expected keys
        assert 'aggregated_sentiment' in result
        assert 'sentiment_breakdown' in result


# ============================================================================
# CATEGORY 9: Integration
# ============================================================================

class TestIntegration:

    def test_av_label_survives_full_pipeline(self):
        """AV label (e.g. 'Somewhat-Bullish') should appear in raw_sentiment_scores."""
        state = {
            'ticker': 'NVDA',
            'cleaned_stock_news': [
                {
                    'title': 'Positive',
                    'overall_sentiment_label': 'Somewhat-Bullish',
                    'ticker_sentiment_score': 0.6,
                    'source': 'Bloomberg',
                }
            ],
            'cleaned_market_news': [],
            'cleaned_related_company_news': [],
            'errors': [],
            'node_execution_times': {},
        }
        result = sentiment_analysis_node(state)
        assert len(result['raw_sentiment_scores']) == 1
        assert result['raw_sentiment_scores'][0]['sentiment_label'] == 'Somewhat-Bullish'

    def test_positive_news_gives_positive_aggregated_sentiment(self):
        """All positive AV articles → aggregated_sentiment > 0 → POSITIVE signal."""
        state = {
            'ticker': 'AAPL',
            'cleaned_stock_news': [
                {'title': 'Beat earnings', 'overall_sentiment_label': 'Bullish',
                 'ticker_sentiment_score': 0.8, 'source': 'Bloomberg'},
                {'title': 'iPhone sales surge', 'overall_sentiment_label': 'Bullish',
                 'ticker_sentiment_score': 0.7, 'source': 'Reuters'},
            ],
            'cleaned_market_news': [],
            'cleaned_related_company_news': [],
            'errors': [],
            'node_execution_times': {},
        }
        result = sentiment_analysis_node(state)

        assert result['aggregated_sentiment'] > 0.0
        assert result['sentiment_signal'] == 'POSITIVE'

    def test_parallel_execution_keys_only(self):
        """Node 5 must NOT return ticker, cleaned_*_news or other node fields."""
        state = {
            'ticker': 'AAPL',
            'cleaned_stock_news': [
                {'title': 'News', 'overall_sentiment_label': 'Bullish',
                 'ticker_sentiment_score': 0.5, 'source': 'X'},
            ],
            'cleaned_market_news': [],
            'cleaned_related_company_news': [],
            'errors': [],
            'node_execution_times': {},
        }
        result = sentiment_analysis_node(state)

        # Node 5 returns only its own fields (partial state for LangGraph merge)
        assert 'ticker' not in result
        assert 'cleaned_stock_news' not in result
        assert 'raw_price_data' not in result


# ============================================================================
# CREDIBILITY WEIGHTING — ADDITIONAL INTEGRATION
# ============================================================================

def test_high_credibility_source_outweighs_low_credibility():
    """Bloomberg (-0.3) should pull weighted sentiment below plain average."""
    articles = [
        {
            'sentiment_score': 0.9,
            'sentiment_label': 'Bullish',
            'title': 'Blog says buy',
            'source': 'Blog',
            'source_credibility_score': 0.30,
            'composite_anomaly_score': 0.75,
            'relevance_score': 0.40,
        },
        {
            'sentiment_score': -0.3,
            'sentiment_label': 'Bearish',
            'title': 'Bloomberg skeptical',
            'source': 'Bloomberg',
            'source_credibility_score': 0.95,
            'composite_anomaly_score': 0.04,
            'relevance_score': 0.85,
        },
    ]
    result = aggregate_sentiment_by_type(articles, 'stock')

    # Simple mean = (0.9 + -0.3) / 2 = 0.30
    # Credibility-weighted: Bloomberg should dominate → weighted < 0.15
    assert result['weighted_sentiment'] < 0.15


def test_raw_scores_include_credibility_fields():
    """raw_sentiment_scores must carry credibility_weight and related fields."""
    state = {
        'ticker': 'TEST',
        'cleaned_stock_news': [
            {
                'title': 'Test article',
                'source': 'Bloomberg',
                'overall_sentiment_label': 'Bullish',
                'ticker_sentiment_score': 0.7,
                'source_credibility_score': 0.95,
                'composite_anomaly_score': 0.04,
                'relevance_score': 0.85,
            }
        ],
        'cleaned_market_news': [],
        'cleaned_related_company_news': [],
        'errors': [],
        'node_execution_times': {},
    }
    result = sentiment_analysis_node(state)

    for score in result['raw_sentiment_scores']:
        assert 'credibility_weight' in score
        assert 'source_credibility_score' in score
        assert 'composite_anomaly_score' in score
        assert 'relevance_score' in score


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
