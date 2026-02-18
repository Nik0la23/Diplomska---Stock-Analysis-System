-- ============================================================================
-- UPDATED DATABASE SCHEMA - Stock Analysis with News Outcome Tracking
-- ============================================================================
-- SQLite Database Schema
-- Last Updated: February 2026
-- Includes: News outcome tracking for learning system (Node 8)

-- ============================================================================
-- PRICE DATA
-- ============================================================================

CREATE TABLE IF NOT EXISTS price_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE INDEX idx_price_ticker_date ON price_data(ticker, date DESC);


-- ============================================================================
-- TECHNICAL INDICATORS
-- ============================================================================

CREATE TABLE IF NOT EXISTS technical_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    rsi REAL,
    macd_value REAL,
    macd_signal REAL,
    macd_histogram REAL,
    bb_upper REAL,
    bb_middle REAL,
    bb_lower REAL,
    sma_20 REAL,
    sma_50 REAL,
    ema_12 REAL,
    ema_26 REAL,
    volume_ratio REAL,
    technical_signal TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE INDEX idx_technical_ticker_date ON technical_indicators(ticker, date DESC);


-- ============================================================================
-- NEWS ARTICLES (Updated)
-- ============================================================================

CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    news_type TEXT NOT NULL,              -- 'stock', 'market', 'related'
    title TEXT NOT NULL,
    description TEXT,
    url TEXT,
    source TEXT,                          -- Source name (e.g., 'Bloomberg.com', 'Reuters.com')
    published_at TEXT,                    -- ISO format date
    sentiment_label TEXT,                 -- 'positive', 'negative', 'neutral'
    sentiment_score REAL,                 -- Confidence score from FinBERT
    is_filtered BOOLEAN DEFAULT 0,        -- True if filtered by Node 9A
    filter_reason TEXT,                   -- Why it was filtered
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(url)
);

CREATE INDEX idx_news_ticker ON news_articles(ticker, published_at DESC);
CREATE INDEX idx_news_source ON news_articles(source);
CREATE INDEX idx_news_sentiment ON news_articles(ticker, sentiment_label);


-- ============================================================================
-- 🆕 NEWS OUTCOMES (Core Learning Table)
-- ============================================================================
-- This table tracks what happened AFTER each news article was published
-- Enables the system to learn which sources and news types are reliable

CREATE TABLE IF NOT EXISTS news_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id INTEGER NOT NULL,             -- Foreign key to news_articles
    ticker TEXT NOT NULL,
    
    -- Price at time of news
    price_at_news REAL NOT NULL,
    
    -- Prices after news (1, 3, 7 days later)
    price_1day_later REAL,
    price_3day_later REAL,
    price_7day_later REAL,
    
    -- Price changes (percentage)
    price_change_1day REAL,
    price_change_3day REAL,
    price_change_7day REAL,
    
    -- Did the prediction come true?
    predicted_direction TEXT,             -- 'UP', 'DOWN', 'FLAT' (from sentiment)
    actual_direction TEXT,                -- 'UP', 'DOWN', 'FLAT' (actual price movement)
    prediction_was_accurate_1day BOOLEAN,
    prediction_was_accurate_3day BOOLEAN,
    prediction_was_accurate_7day BOOLEAN, -- Primary metric
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (news_id) REFERENCES news_articles(id),
    UNIQUE(news_id)
);

CREATE INDEX idx_outcomes_ticker ON news_outcomes(ticker);
CREATE INDEX idx_outcomes_accuracy ON news_outcomes(ticker, prediction_was_accurate_7day);
CREATE INDEX idx_outcomes_news_id ON news_outcomes(news_id);


-- ============================================================================
-- 🆕 SOURCE RELIABILITY (Aggregated Learning)
-- ============================================================================
-- This table stores the calculated reliability score for each news source
-- per stock (because Bloomberg might be good for NVDA but bad for penny stocks)

CREATE TABLE IF NOT EXISTS source_reliability (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    source_name TEXT NOT NULL,            -- e.g., 'Bloomberg.com', 'Reuters.com'
    analysis_date TEXT NOT NULL,          -- When this was calculated
    
    -- Accuracy metrics
    total_articles INTEGER NOT NULL,      -- How many articles from this source
    accurate_predictions INTEGER NOT NULL,-- How many were correct
    accuracy_rate REAL NOT NULL,          -- 0.0 to 1.0
    
    -- Impact metrics
    avg_price_impact REAL,                -- Average % price change after their news
    confidence_multiplier REAL,           -- How much to boost/reduce confidence (0.5 to 1.5)
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(ticker, source_name, analysis_date)
);

CREATE INDEX idx_reliability_ticker ON source_reliability(ticker, source_name);
CREATE INDEX idx_reliability_accuracy ON source_reliability(ticker, accuracy_rate DESC);


-- ============================================================================
-- 🆕 NEWS TYPE EFFECTIVENESS
-- ============================================================================
-- Tracks how effective each type of news is for predicting price movement
-- per stock (because 'market' news might be more predictive for ETFs)

CREATE TABLE IF NOT EXISTS news_type_effectiveness (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    news_type TEXT NOT NULL,              -- 'stock', 'market', 'related'
    analysis_date TEXT NOT NULL,
    
    -- Effectiveness metrics
    total_articles INTEGER NOT NULL,
    accurate_predictions INTEGER NOT NULL,
    accuracy_rate REAL NOT NULL,          -- 0.0 to 1.0
    avg_price_impact REAL,                -- Average % price change
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(ticker, news_type, analysis_date)
);

CREATE INDEX idx_type_effectiveness ON news_type_effectiveness(ticker, news_type);


-- ============================================================================
-- TICKER STATS (Velocity Baseline Cache)
-- ============================================================================
-- Stores the true per-calendar-day article average per ticker so the velocity
-- detector never has to do a full DB scan on repeat runs.
-- Written by Node 2 after any live API fetch.
-- Read by get_article_count_baseline() — cache is considered fresh for 1 hour.

CREATE TABLE IF NOT EXISTS ticker_stats (
    ticker               TEXT PRIMARY KEY,
    daily_article_avg    REAL    NOT NULL,   -- articles per calendar day (true mean)
    total_articles       INTEGER NOT NULL,   -- total articles counted in the window
    date_range_days      INTEGER NOT NULL,   -- calendar days from oldest article to today
    oldest_article_date  TEXT,               -- earliest published_at date in window
    computed_at          TIMESTAMP NOT NULL  -- when this row was last updated
);

CREATE INDEX IF NOT EXISTS idx_ticker_stats_computed ON ticker_stats(ticker, computed_at DESC);


-- ============================================================================
-- RELATED COMPANIES
-- ============================================================================

CREATE TABLE IF NOT EXISTS related_companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    related_ticker TEXT NOT NULL,
    relationship_type TEXT,               -- 'competitor', 'supplier', 'customer', 'sector'
    correlation_score REAL,               -- -1.0 to 1.0
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, related_ticker)
);

CREATE INDEX idx_related_ticker ON related_companies(ticker);


-- ============================================================================
-- EARLY ANOMALY DETECTION RESULTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS early_anomaly_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    analysis_date TEXT NOT NULL,
    
    -- Detection flags
    keyword_alerts TEXT,                  -- JSON array of dangerous keywords found
    news_surge_detected BOOLEAN,
    suspicious_sources TEXT,              -- JSON array of untrusted sources
    coordinated_posting BOOLEAN,
    
    -- Results
    filtered_news_count INTEGER,          -- How many articles were removed
    risk_level TEXT,                      -- 'LOW', 'MEDIUM', 'HIGH'
    warning_flags TEXT,                   -- JSON array of human-readable warnings
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_early_anomaly ON early_anomaly_results(ticker, analysis_date);


-- ============================================================================
-- BEHAVIORAL ANOMALY DETECTION RESULTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS behavioral_anomaly_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    analysis_date TEXT NOT NULL,
    
    -- Detection results
    pump_and_dump_score REAL,             -- 0-100
    is_pump_and_dump BOOLEAN,
    price_anomalies TEXT,                 -- JSON array: ['SPIKE', 'CRASH', 'GAP']
    volume_anomalies TEXT,                -- JSON array: ['UNUSUAL_HIGH', 'UNUSUAL_LOW']
    volatility_anomaly BOOLEAN,
    news_price_divergence BOOLEAN,        -- Positive news but price crashed
    
    -- Risk assessment
    manipulation_probability REAL,        -- 0.0 to 1.0
    risk_level TEXT,                      -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    warning_flags TEXT,                   -- JSON array
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_behavioral_anomaly ON behavioral_anomaly_results(ticker, analysis_date);
CREATE INDEX idx_pump_dump ON behavioral_anomaly_results(ticker, is_pump_and_dump);


-- ============================================================================
-- BACKTEST RESULTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    signal_type TEXT NOT NULL,            -- 'technical', 'stock_news', 'market_news', 'related'
    accuracy REAL,                        -- 0.0 to 1.0
    sample_size INTEGER,                  -- Number of predictions tested
    backtest_period_days INTEGER,         -- How many days analyzed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_backtest ON backtest_results(ticker, signal_type);


-- ============================================================================
-- ADAPTIVE WEIGHTS HISTORY
-- ============================================================================

CREATE TABLE IF NOT EXISTS weight_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    
    -- The 4 adaptive weights (sum to 1.0)
    technical_weight REAL,
    stock_news_weight REAL,
    market_news_weight REAL,
    related_companies_weight REAL,
    
    -- Metadata
    weight_explanation TEXT,              -- Why these weights?
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_weights ON weight_history(ticker, date DESC);


-- ============================================================================
-- FINAL SIGNALS
-- ============================================================================

CREATE TABLE IF NOT EXISTS final_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    
    -- Signal
    recommendation TEXT,                  -- 'BUY', 'SELL', 'HOLD'
    confidence REAL,                      -- 0-100
    strength TEXT,                        -- 'WEAK', 'MODERATE', 'STRONG'
    
    -- Targets
    target_price REAL,
    stop_loss REAL,
    
    -- Explanations
    contributing_factors TEXT,            -- JSON array
    risk_warnings TEXT,                   -- JSON array
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_signals ON final_signals(ticker, date DESC);


-- ============================================================================
-- NODE EXECUTION LOGS (Performance Monitoring)
-- ============================================================================

CREATE TABLE IF NOT EXISTS node_execution_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    node_name TEXT NOT NULL,              -- 'node_1', 'node_2', etc.
    execution_time_seconds REAL,
    success BOOLEAN,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_node_logs ON node_execution_logs(ticker, created_at DESC);
CREATE INDEX idx_node_performance ON node_execution_logs(node_name, execution_time_seconds);


-- ============================================================================
-- MONTE CARLO SIMULATION RESULTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS monte_carlo_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    analysis_date TEXT NOT NULL,
    
    -- Forecast parameters
    forecast_days INTEGER,                -- How many days forecasted
    num_simulations INTEGER,              -- How many paths (usually 1000)
    
    -- Results
    mean_forecast REAL,                   -- Expected price
    median_forecast REAL,
    confidence_68_lower REAL,             -- 1 standard deviation
    confidence_68_upper REAL,
    confidence_95_lower REAL,             -- 2 standard deviations
    confidence_95_upper REAL,
    probability_up REAL,                  -- P(price increases)
    probability_down REAL,                -- P(price decreases)
    expected_return REAL,                 -- Expected % return
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_monte_carlo ON monte_carlo_results(ticker, analysis_date);


-- ============================================================================
-- SENTIMENT ANALYSIS RESULTS (Daily Aggregates)
-- ============================================================================

CREATE TABLE IF NOT EXISTS sentiment_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    
    -- Sentiment by source type
    stock_news_sentiment REAL,            -- -1.0 to 1.0
    market_news_sentiment REAL,
    related_companies_sentiment REAL,
    
    -- Combined
    combined_sentiment REAL,
    sentiment_signal TEXT,                -- 'BUY', 'SELL', 'HOLD'
    confidence REAL,                      -- 0-100
    
    -- Counts
    total_articles INTEGER,
    positive_count INTEGER,
    negative_count INTEGER,
    neutral_count INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE INDEX idx_sentiment_daily ON sentiment_daily(ticker, date DESC);


-- ============================================================================
-- MARKET CONTEXT DATA
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    
    -- Market data
    sector_performance REAL,              -- % change
    market_trend TEXT,                    -- 'BULLISH', 'BEARISH', 'NEUTRAL'
    correlation_strength REAL,            -- 0.0 to 1.0
    
    -- Related companies signals
    related_companies_avg_signal REAL,    -- Average signal from competitors
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE INDEX idx_market_context ON market_context(ticker, date DESC);


-- ============================================================================
-- VIEWS FOR EASY QUERYING
-- ============================================================================

-- View: News with Outcomes (for learning queries)
CREATE VIEW IF NOT EXISTS news_with_outcomes AS
SELECT 
    n.id,
    n.ticker,
    n.news_type,
    n.title,
    n.source,
    n.published_at,
    n.sentiment_label,
    n.sentiment_score,
    o.price_at_news,
    o.price_7day_later,
    o.price_change_7day,
    o.predicted_direction,
    o.actual_direction,
    o.prediction_was_accurate_7day
FROM news_articles n
INNER JOIN news_outcomes o ON n.id = o.news_id
WHERE n.is_filtered = 0;

-- View: Source Performance Summary
CREATE VIEW IF NOT EXISTS source_performance_summary AS
SELECT 
    ticker,
    source,
    COUNT(*) as total_articles,
    SUM(CASE WHEN prediction_was_accurate_7day = 1 THEN 1 ELSE 0 END) as accurate_count,
    ROUND(AVG(CASE WHEN prediction_was_accurate_7day = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as accuracy_pct,
    ROUND(AVG(ABS(price_change_7day)), 2) as avg_impact
FROM news_with_outcomes
GROUP BY ticker, source
HAVING total_articles >= 5
ORDER BY ticker, accuracy_pct DESC;

-- View: Daily Behavioural Aggregates (price-based metrics per ticker/day)
-- Used by Node 9B Historical Pattern Matcher.
-- NOTE: The Python get_historical_daily_aggregates() computes this in-process
-- so Node 9B never queries this view directly, but it is available for ad-hoc
-- SQL analysis and dashboards.
CREATE VIEW IF NOT EXISTS daily_behavioral_aggregates AS
SELECT
    p.ticker,
    p.date,
    p.volume,
    CASE
        WHEN (SELECT AVG(v.volume) FROM price_data v
              WHERE v.ticker = p.ticker
                AND v.date >  date(p.date, '-31 days')
                AND v.date <= p.date) > 0
        THEN CAST(p.volume AS REAL) /
             (SELECT AVG(v.volume) FROM price_data v
              WHERE v.ticker = p.ticker
                AND v.date >  date(p.date, '-31 days')
                AND v.date <= p.date)
        ELSE 1.0
    END AS volume_ratio,
    CASE
        WHEN (SELECT close FROM price_data pr
              WHERE pr.ticker = p.ticker AND pr.date < p.date
              ORDER BY pr.date DESC LIMIT 1) > 0
        THEN (p.close -
              (SELECT close FROM price_data pr
               WHERE pr.ticker = p.ticker AND pr.date < p.date
               ORDER BY pr.date DESC LIMIT 1))
             / (SELECT close FROM price_data pr
                WHERE pr.ticker = p.ticker AND pr.date < p.date
                ORDER BY pr.date DESC LIMIT 1) * 100
        ELSE 0.0
    END AS price_change_1d,
    CASE
        WHEN (SELECT close FROM price_data pf
              WHERE pf.ticker = p.ticker
                AND pf.date >  p.date
                AND pf.date <= date(p.date, '+12 days')
              ORDER BY pf.date ASC LIMIT 1) IS NOT NULL
        THEN ((SELECT close FROM price_data pf
               WHERE pf.ticker = p.ticker
                 AND pf.date >  p.date
                 AND pf.date <= date(p.date, '+12 days')
               ORDER BY pf.date ASC LIMIT 1) - p.close)
             / p.close * 100
        ELSE NULL
    END AS price_change_7d
FROM price_data p;

-- View: Daily News Aggregates (article-level metrics per ticker/day)
CREATE VIEW IF NOT EXISTS daily_news_aggregates AS
SELECT
    n.ticker,
    DATE(n.published_at)                                                  AS date,
    COUNT(*)                                                              AS article_count,
    AVG(CASE WHEN n.is_filtered = 1 THEN 0.7
             ELSE ABS(COALESCE(n.sentiment_score, 0)) * 0.3 END)         AS avg_composite_anomaly,
    COALESCE(
        (SELECT AVG(sr.accuracy_rate)
         FROM source_reliability sr
         WHERE sr.ticker = n.ticker),
        0.5)                                                              AS avg_source_credibility,
    AVG(CASE WHEN n.is_filtered = 0
             THEN COALESCE(n.sentiment_score, 0) ELSE 0 END)             AS sentiment_avg,
    CASE
        WHEN AVG(CASE WHEN n.is_filtered = 0
                      THEN COALESCE(n.sentiment_score, 0) ELSE 0 END) >  0.05 THEN 'positive'
        WHEN AVG(CASE WHEN n.is_filtered = 0
                      THEN COALESCE(n.sentiment_score, 0) ELSE 0 END) < -0.05 THEN 'negative'
        ELSE 'neutral'
    END                                                                   AS sentiment_label
FROM news_articles n
WHERE n.published_at IS NOT NULL
GROUP BY n.ticker, DATE(n.published_at);

-- View: Latest Signals with Confidence
CREATE VIEW IF NOT EXISTS latest_signals AS
SELECT 
    ticker,
    date,
    recommendation,
    confidence,
    strength,
    target_price,
    created_at
FROM final_signals
WHERE date >= date('now', '-7 days')
ORDER BY ticker, date DESC;


-- ============================================================================
-- SAMPLE QUERIES (Documentation)
-- ============================================================================

/*
-- Query 1: Get source reliability for a specific stock
SELECT * FROM source_reliability 
WHERE ticker = 'NVDA' 
ORDER BY accuracy_rate DESC;

-- Query 2: Find most accurate news sources across all stocks
SELECT source_name, AVG(accuracy_rate) as avg_accuracy, SUM(total_articles) as total
FROM source_reliability
GROUP BY source_name
HAVING total >= 20
ORDER BY avg_accuracy DESC;

-- Query 3: Get recent news with their outcomes
SELECT * FROM news_with_outcomes
WHERE ticker = 'NVDA'
AND published_at >= date('now', '-30 days')
ORDER BY published_at DESC;

-- Query 4: Find news that caused biggest price movements
SELECT ticker, title, source, price_change_7day
FROM news_with_outcomes
WHERE ABS(price_change_7day) > 5.0
ORDER BY ABS(price_change_7day) DESC
LIMIT 20;

-- Query 5: Compare news type effectiveness for a stock
SELECT news_type, accuracy_rate, avg_price_impact
FROM news_type_effectiveness
WHERE ticker = 'AAPL'
ORDER BY accuracy_rate DESC;

-- Query 6: Get pump-and-dump detections
SELECT ticker, analysis_date, pump_and_dump_score, warning_flags
FROM behavioral_anomaly_results
WHERE is_pump_and_dump = 1
ORDER BY analysis_date DESC;
*/
