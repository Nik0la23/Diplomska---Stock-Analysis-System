"""
Tests for Node 9A: Content Analysis & Feature Extraction
Tests all scoring functions and main node function.
"""

import pytest
from src.langgraph_nodes.node_09a_content_analysis import (
    calculate_sensationalism_score,
    calculate_urgency_score,
    calculate_unverified_claims_score,
    assess_source_credibility,
    calculate_complexity_score,
    extract_content_tags,
    calculate_composite_anomaly_score,
    process_article,
    content_analysis_node
)


# ============================================================================
# Test 1: Sensationalism Scoring
# ============================================================================

def test_sensationalism_low():
    """Test low sensationalism for normal news"""
    text = "Company reports quarterly earnings results"
    score = calculate_sensationalism_score(text)
    assert score < 0.3, f"Expected low sensationalism, got {score}"


def test_sensationalism_high():
    """Test high sensationalism for clickbait"""
    text = "SHOCKING!!! REVOLUTIONARY BREAKTHROUGH EXPOSED!!!"
    score = calculate_sensationalism_score(text)
    assert score > 0.7, f"Expected high sensationalism, got {score}"


def test_sensationalism_caps():
    """Test detection of ALL CAPS words"""
    text = "URGENT BREAKING NEWS ALERT MAJOR DEVELOPMENT"
    score = calculate_sensationalism_score(text)
    assert score >= 0.3, f"Expected elevated sensationalism for caps, got {score}"


def test_sensationalism_empty():
    """Test handling of empty text"""
    score = calculate_sensationalism_score("")
    assert score == 0.0


# ============================================================================
# Test 2: Urgency Scoring
# ============================================================================

def test_urgency_none():
    """Test no urgency for normal news"""
    text = "Company announces new product line for next quarter"
    score = calculate_urgency_score(text)
    assert score == 0.0, f"Expected no urgency, got {score}"


def test_urgency_high():
    """Test high urgency for time-pressure language"""
    text = "BREAKING: URGENT - ACT NOW before it's too late!"
    score = calculate_urgency_score(text)
    assert score > 0.7, f"Expected high urgency, got {score}"


def test_urgency_medium():
    """Test medium urgency for single keyword"""
    text = "BREAKING: Company makes announcement"
    score = calculate_urgency_score(text)
    assert 0.2 < score < 0.5, f"Expected medium urgency, got {score}"


# ============================================================================
# Test 3: Unverified Claims Detection
# ============================================================================

def test_unverified_low():
    """Test low unverified score for factual news"""
    text = "Company officially announced earnings results in SEC filing"
    score = calculate_unverified_claims_score(text)
    assert score < 0.3, f"Expected low unverified score, got {score}"


def test_unverified_high():
    """Test high unverified score for speculative news"""
    text = "Allegedly, sources say the company might be facing rumors of potential issues"
    score = calculate_unverified_claims_score(text)
    assert score > 0.7, f"Expected high unverified score, got {score}"


def test_unverified_medium():
    """Test medium unverified score for partial hedging"""
    text = "The company reportedly plans to expand operations"
    score = calculate_unverified_claims_score(text)
    assert 0.3 < score < 0.6, f"Expected medium unverified score, got {score}"


# ============================================================================
# Test 4: Source Credibility Assessment
# ============================================================================

def test_credibility_high_bloomberg():
    """Test high credibility for Bloomberg"""
    score = assess_source_credibility("https://bloomberg.com/news/article", "Bloomberg")
    assert score >= 0.9, f"Expected high credibility for Bloomberg, got {score}"


def test_credibility_high_reuters():
    """Test high credibility for Reuters"""
    score = assess_source_credibility("https://reuters.com/article", "Reuters")
    assert score >= 0.9, f"Expected high credibility for Reuters, got {score}"


def test_credibility_medium_cnbc():
    """Test medium credibility for CNBC"""
    score = assess_source_credibility("https://cnbc.com/news", "CNBC")
    assert 0.7 <= score < 0.9, f"Expected medium credibility for CNBC, got {score}"


def test_credibility_low_unknown():
    """Test low credibility for unknown source"""
    score = assess_source_credibility("https://unknown-blog.com/post", "Unknown Blog")
    assert score < 0.5, f"Expected low credibility for unknown source, got {score}"


# ============================================================================
# Test 5: Complexity Analysis
# ============================================================================

def test_complexity_low():
    """Test low complexity for simple text"""
    text = "Stock price went up today"
    score = calculate_complexity_score(text)
    assert score < 0.3, f"Expected low complexity, got {score}"


def test_complexity_high():
    """Test high complexity for technical text"""
    text = "The EBITDA calculation incorporates derivatives, arbitrage opportunities, and hedging strategies"
    score = calculate_complexity_score(text)
    assert score > 0.5, f"Expected high complexity, got {score}"


def test_complexity_empty():
    """Test handling of empty text"""
    score = calculate_complexity_score("")
    assert score == 0.0


# ============================================================================
# Test 6: Content Tags Extraction
# ============================================================================

def test_tags_fraud():
    """Test extraction of fraud-related keywords"""
    text = "SEC launches fraud investigation into company practices"
    tags = extract_content_tags(text, "XYZ")
    assert 'fraud' in tags['keywords']
    assert 'SEC' in tags['keywords'] or 'investigation' in tags['keywords']
    assert tags['topic'] == 'regulatory_action'


def test_tags_earnings():
    """Test extraction of earnings-related keywords"""
    text = "Company reports strong earnings and revenue growth"
    tags = extract_content_tags(text, "ABC")
    assert 'earnings' in tags['keywords'] or 'revenue' in tags['keywords']
    assert tags['topic'] == 'earnings_report'


def test_tags_merger():
    """Test extraction of merger-related keywords"""
    text = "Company announces merger with competitor in major acquisition"
    tags = extract_content_tags(text, "DEF")
    assert 'merger' in tags['keywords'] or 'acquisition' in tags['keywords']
    assert tags['topic'] == 'merger_acquisition'


def test_tags_temporal_future():
    """Test temporal classification for future events"""
    text = "Company plans to expand operations next quarter"
    tags = extract_content_tags(text, "GHI")
    assert tags['temporal'] == 'future'


def test_tags_temporal_past():
    """Test temporal classification for past events"""
    text = "Company reported results yesterday"
    tags = extract_content_tags(text, "JKL")
    assert tags['temporal'] == 'past'


# ============================================================================
# Test 7: Composite Score Calculation
# ============================================================================

def test_composite_clean_article():
    """Test composite score for clean article (high credibility, low anomaly)"""
    scores = {
        'sensationalism': 0.1,
        'urgency': 0.0,
        'unverified': 0.1,
        'credibility': 0.95  # Bloomberg
    }
    composite = calculate_composite_anomaly_score(scores)
    assert composite < 0.3, f"Expected low composite for clean article, got {composite}"


def test_composite_suspicious_article():
    """Test composite score for suspicious article (low credibility, high anomaly)"""
    scores = {
        'sensationalism': 0.8,
        'urgency': 0.9,
        'unverified': 0.7,
        'credibility': 0.2  # Unknown source
    }
    composite = calculate_composite_anomaly_score(scores)
    assert composite > 0.7, f"Expected high composite for suspicious article, got {composite}"


def test_composite_mixed_article():
    """Test composite score for mixed signals"""
    scores = {
        'sensationalism': 0.5,
        'urgency': 0.3,
        'unverified': 0.4,
        'credibility': 0.6  # Medium source
    }
    composite = calculate_composite_anomaly_score(scores)
    assert 0.3 < composite < 0.7, f"Expected medium composite, got {composite}"


# ============================================================================
# Test 8: Process Single Article
# ============================================================================

def test_process_article_bloomberg():
    """Test processing Bloomberg article (should have low anomaly score)"""
    article = {
        'headline': 'Company reports strong quarterly earnings',
        'summary': 'The company announced revenue growth and positive outlook',
        'url': 'https://bloomberg.com/news/article123',
        'source': 'Bloomberg',
        'datetime': 1234567890
    }
    
    enriched = process_article(article, 'AAPL')
    
    # Check that all score fields are added
    assert 'sensationalism_score' in enriched
    assert 'urgency_score' in enriched
    assert 'unverified_claims_score' in enriched
    assert 'source_credibility_score' in enriched
    assert 'complexity_score' in enriched
    assert 'composite_anomaly_score' in enriched
    assert 'content_tags' in enriched
    
    # Check credibility is high
    assert enriched['source_credibility_score'] >= 0.9
    
    # Check composite is low (clean article)
    assert enriched['composite_anomaly_score'] < 0.5


def test_process_article_suspicious():
    """Test processing suspicious article (should have high anomaly score)"""
    article = {
        'headline': 'SHOCKING!!! Company EXPOSED - ACT NOW!!!',
        'summary': 'Allegedly, sources say the company might face issues. Rumored problems ahead.',
        'url': 'https://unknown-blog.com/post',
        'source': 'Unknown Blog',
        'datetime': 1234567890
    }
    
    enriched = process_article(article, 'XYZ')
    
    # Check credibility is low
    assert enriched['source_credibility_score'] < 0.5
    
    # Check sensationalism is high
    assert enriched['sensationalism_score'] > 0.5
    
    # Check composite is high (suspicious article)
    assert enriched['composite_anomaly_score'] > 0.5


# ============================================================================
# Test 9: Main Node Function - Normal Operation
# ============================================================================

def test_content_analysis_node_success():
    """Test main node function with valid input"""
    state = {
        'ticker': 'AAPL',
        'stock_news': [
            {
                'headline': 'Apple reports strong earnings',
                'summary': 'Revenue and profit exceed expectations',
                'url': 'https://bloomberg.com/article1',
                'source': 'Bloomberg',
                'datetime': 1234567890
            },
            {
                'headline': 'SHOCKING Apple news!!!',
                'summary': 'Allegedly major changes coming',
                'url': 'https://unknown.com/post',
                'source': 'Unknown',
                'datetime': 1234567891
            }
        ],
        'market_news': [
            {
                'headline': 'Market trends positive',
                'summary': 'Overall market showing strength',
                'url': 'https://reuters.com/market',
                'source': 'Reuters',
                'datetime': 1234567892
            }
        ],
        'related_company_news': [],
        'errors': [],
        'node_execution_times': {}
    }
    
    result = content_analysis_node(state)
    
    # Check that cleaned news lists are populated
    assert 'cleaned_stock_news' in result
    assert 'cleaned_market_news' in result
    assert 'cleaned_related_company_news' in result
    assert 'content_analysis_summary' in result
    
    # Check article counts
    assert len(result['cleaned_stock_news']) == 2
    assert len(result['cleaned_market_news']) == 1
    assert len(result['cleaned_related_company_news']) == 0
    
    # Check that scores are embedded in articles
    first_article = result['cleaned_stock_news'][0]
    assert 'composite_anomaly_score' in first_article
    assert 'content_tags' in first_article
    
    # Check summary
    summary = result['content_analysis_summary']
    assert summary['total_articles_processed'] == 3
    assert 'average_scores' in summary
    assert 'source_credibility_distribution' in summary
    
    # Check execution time tracked
    assert 'node_9a' in result['node_execution_times']
    assert result['node_execution_times']['node_9a'] > 0


# ============================================================================
# Test 10: Main Node Function - Empty News
# ============================================================================

def test_content_analysis_node_empty():
    """Test main node function with no news articles"""
    state = {
        'ticker': 'TEST',
        'stock_news': [],
        'market_news': [],
        'related_company_news': [],
        'errors': [],
        'node_execution_times': {}
    }
    
    result = content_analysis_node(state)
    
    # Check that it handles empty input gracefully
    assert len(result['cleaned_stock_news']) == 0
    assert len(result['cleaned_market_news']) == 0
    assert result['content_analysis_summary']['total_articles_processed'] == 0
    assert 'node_9a' in result['node_execution_times']


# ============================================================================
# Test 11: Main Node Function - Error Handling
# ============================================================================

def test_content_analysis_node_missing_fields():
    """Test main node function with missing state fields"""
    state = {
        'ticker': 'TEST',
        # Missing news fields
        'errors': [],
        'node_execution_times': {}
    }
    
    result = content_analysis_node(state)
    
    # Should handle missing fields gracefully
    assert 'cleaned_stock_news' in result
    assert 'cleaned_market_news' in result
    assert 'cleaned_related_company_news' in result
    assert 'node_9a' in result['node_execution_times']


# ============================================================================
# Test 12: Integration - Scores Embedded Correctly
# ============================================================================

def test_scores_embedded_in_all_articles():
    """Test that all articles get scores embedded"""
    state = {
        'ticker': 'NVDA',
        'stock_news': [
            {'headline': 'Test 1', 'summary': 'Summary 1', 'url': 'http://test1.com', 'source': 'Test'},
            {'headline': 'Test 2', 'summary': 'Summary 2', 'url': 'http://test2.com', 'source': 'Test'},
        ],
        'market_news': [
            {'headline': 'Test 3', 'summary': 'Summary 3', 'url': 'http://test3.com', 'source': 'Test'},
        ],
        'related_company_news': [
            {'headline': 'Test 4', 'summary': 'Summary 4', 'url': 'http://test4.com', 'source': 'Test'},
        ],
        'errors': [],
        'node_execution_times': {}
    }
    
    result = content_analysis_node(state)
    
    # Check all articles have scores
    all_articles = (
        result['cleaned_stock_news'] + 
        result['cleaned_market_news'] + 
        result['cleaned_related_company_news']
    )
    
    required_fields = [
        'sensationalism_score',
        'urgency_score',
        'unverified_claims_score',
        'source_credibility_score',
        'complexity_score',
        'composite_anomaly_score',
        'content_tags'
    ]
    
    for article in all_articles:
        for field in required_fields:
            assert field in article, f"Article missing {field}: {article.get('headline', 'Unknown')}"


# ============================================================================
# Test 13: Summary Statistics
# ============================================================================

def test_summary_statistics_accurate():
    """Test that summary statistics are calculated correctly"""
    state = {
        'ticker': 'TSLA',
        'stock_news': [
            {
                'headline': 'Normal article',
                'summary': 'Standard content',
                'url': 'https://reuters.com/article',
                'source': 'Reuters',
                'datetime': 123
            },
            {
                'headline': 'SHOCKING!!!',
                'summary': 'Allegedly something happened',
                'url': 'https://unknown.com/post',
                'source': 'Unknown',
                'datetime': 124
            }
        ],
        'market_news': [],
        'related_company_news': [],
        'errors': [],
        'node_execution_times': {}
    }
    
    result = content_analysis_node(state)
    summary = result['content_analysis_summary']
    
    # Check summary accuracy
    assert summary['total_articles_processed'] == 2
    assert summary['articles_by_type']['stock'] == 2
    assert summary['articles_by_type']['market'] == 0
    
    # Check distribution
    dist = summary['source_credibility_distribution']
    assert dist['high'] == 1  # Reuters
    assert dist['low'] >= 1  # Unknown


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
