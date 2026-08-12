# Quantitative Finance Portfolio

A suite of self-directed Python projects designed to demonstrate applied data, finance, and NLP skills relevant to markets-facing graduate roles.

---

## Projects

### 1. [Automated Equity Analyser](./equity-analyser/README.md)

A data pipeline that pulls live market data for a configurable set of equities via the Yahoo Finance API, runs an automated cleaning and validation pipeline, computes a full risk and return analytics suite, and exports a formatted multi-sheet Excel report.

**Key outputs:**
- Annualised return, volatility, Sharpe ratio, beta, and information ratio benchmarked against the S&P 500
- Rolling 6-month Sharpe ratio and drawdown series
- Momentum signals based on 3- and 6-month moving averages
- Return correlation matrix across all tickers
- Normalised price trend and drawdown charts

**Stack:** Python, pandas, numpy, yfinance, openpyxl, xlsxwriter

---

### 2. [Financial News Sentiment Analyser](./sentiment-analyser/README_sentiment.md)

An NLP pipeline that pulls financial news headlines for the same equity basket, scores each headline using FinBERT (a BERT model fine-tuned on financial text), aggregates sentiment monthly, and tests correlation against contemporaneous and forward returns.

**Key outputs:**
- Per-headline sentiment scores (positive, negative, neutral) and composite score [-1, +1]
- Monthly aggregated sentiment per ticker with % positive/negative breakdown
- Contemporaneous and forward return correlations — tests whether news sentiment predicts next-period returns
- Accumulating headline log — designed to be run daily to build a time series

**Stack:** Python, FinBERT (HuggingFace/transformers), pandas, yfinance, openpyxl, xlsxwriter

---

## Design rationale

Both projects use the same ticker set, allowing the outputs to be read together. The equity analyser provides quantitative price and return data; the sentiment analyser provides a qualitative news signal. The central research question connecting them:

> *Does the sentiment of financial news headlines predict forward equity returns, or are publicly available signals already priced in?*

Weak forward correlation between sentiment and next-period returns is consistent with the semi-strong form of the Efficient Market Hypothesis. A meaningful forward correlation would suggest either a market inefficiency or a sentiment lag in a specific asset — both of which are worth investigating further.

---

## Repository structure

```
.
├── equity-analyser/
│   ├── analyser.py                       # Main script
│   ├── Financial_Data_Report.xlsx        # Sample output
│   ├── Financial_Data_Report_charts.xlsx # Sample charts
│   └── README.md
│
├── sentiment-analyser/
│   ├── sentiment_analyser.py             # Main script
│   ├── headlines_log.csv                 # Accumulated headline log
│   ├── Sentiment_Report.xlsx             # Sample output
│   ├── Sentiment_Report_charts.xlsx      # Sample charts
│   └── README_sentiment.md
│
└── README.md                             # This file
```

---

## How to run

Each project is self-contained. See the individual READMEs for installation and setup instructions. Both require Python 3.10+ and are tested on Anaconda 3.12.

**Recommended order:**
1. Run the equity analyser to generate price and return data
2. Run the sentiment analyser daily to accumulate headlines
3. After 3+ months, the sentiment correlation table populates automatically

---

## Skills demonstrated

| Area | Detail |
|---|---|
| Data engineering | API ingestion, automated cleaning, deduplication, outlier detection, missing value imputation |
| Financial analysis | Sharpe ratio, beta, information ratio, drawdown, momentum signals, return correlation |
| NLP | FinBERT sentiment scoring, batch inference, signal aggregation |
| Software practice | Modular pipeline design, logging, config-driven parameters, CSV accumulation with deduplication |
| Communication | Formatted Excel output designed to be readable by a non-technical audience |
