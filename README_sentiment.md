# Financial News Sentiment Analyser

A self-directed Python project that pulls financial news headlines for a configurable set of equity tickers, scores each headline using FinBERT (a BERT model fine-tuned on financial text), aggregates sentiment monthly, and tests whether sentiment predicts forward returns.

Built alongside an automated equity analyser.

---

## What it does

1. Pulls the latest financial news headlines for each ticker via the `yfinance` API
2. Scores each headline using **FinBERT** (`ProsusAI/finbert`) — returning positive, negative, and neutral probabilities plus a composite sentiment score in the range [-1, +1]
3. Appends scored headlines to a running CSV log, deduplicating on URL so re-runs never double-count
4. Aggregates sentiment monthly per ticker — mean score, % positive/negative/neutral, headline count
5. Joins monthly sentiment to return data and tests correlation against both contemporaneous and forward (T+1) returns
6. Exports a formatted Excel workbook and a companion charts file

Run it regularly — the pipeline is designed to accumulate data over time. Correlations populate automatically once 3+ months of data exist per ticker.

---

## Output files

| File | Contents |
|---|---|
| `Sentiment_Report.xlsx` | Four-sheet workbook — raw headlines, monthly sentiment, correlation table, log summary |
| `Sentiment_Report_charts.xlsx` | Line charts — monthly sentiment trend and % positive headlines per ticker |
| `headlines_log.csv` | Accumulated headline log — do not delete between runs |

---

## How to run

### 1. Install dependencies

```bash
conda install -c huggingface transformers
conda install pytorch -c pytorch
conda install xlsxwriter
pip install yfinance openpyxl
```

Or using pip directly into Anaconda:

```bash
/opt/anaconda3/bin/pip install transformers torch yfinance openpyxl xlsxwriter
```

### 2. Configure

Edit the config block at the top of `sentiment_analyser.py`:

```python
TICKERS       = ["^GSPC", "AAPL", "MSFT", "NVDA", "AMD", "^VIX"]
OUTPUT_FILE   = "Sentiment_Report.xlsx"
HEADLINES_LOG = "/path/to/your/Desktop/headlines_log.csv"  # use absolute path
FINBERT_MODEL = "ProsusAI/finbert"
```

Index tickers (`^GSPC`, `^VIX`) are automatically skipped for news ingestion — yfinance does not carry news for indices. They can remain in the list for reference.

> **Important:** set `HEADLINES_LOG` to an absolute path so the log file is always found regardless of how the script is launched.

### 3. Run

```bash
/opt/anaconda3/bin/python3 sentiment_analyser.py
```

Or from IDLE launched via:

```bash
/opt/anaconda3/bin/python3 -m idlelib
```

> **First run:** FinBERT downloads ~400MB on first use. This takes 1–2 minutes depending on connection speed. The model is cached locally after that — subsequent runs load in under 30 seconds.

### 4. Run regularly

Run every 2–3 days. Each run pulls the latest headlines, scores them, and appends to the log. Correlations populate automatically once 3+ months of data exist per ticker.

---

## Workbook sheets

| Sheet | Tab colour | Contents |
|---|---|---|
| **Raw Headlines** | Grey | Every headline in the log with FinBERT scores — label and sentiment score colour-coded green (positive) / red (negative) |
| **Monthly Sentiment** | Blue | Mean sentiment score, % positive/negative/neutral, and headline count per ticker per month |
| **Correlations** | Purple | Contemporaneous and forward return correlations per ticker, plus pooled figures. Shows a placeholder message until 3+ months of data exist |
| **Log Summary** | Green | Running totals — total headlines, date range, breakdown by ticker and month |

---

## Sentiment scoring

Each headline is passed through FinBERT, which returns three probabilities summing to 1.0:

| Output | Description |
|---|---|
| `positive` | Probability the headline expresses positive financial sentiment |
| `negative` | Probability the headline expresses negative financial sentiment |
| `neutral` | Probability the headline is neutral |
| `label` | Winning class — whichever probability is highest |
| `sentiment_score` | Composite score: `positive − negative`, range [-1, +1] |

The composite score is preferred over the label for analysis because it captures both direction and confidence. A score of +0.03 (barely positive) and +0.91 (strongly positive) both label as "positive" but carry very different signal.

**Why FinBERT over generic sentiment models:** Standard NLP models (VADER, TextBlob) are trained on general text and misread financial language. Phrases like "profit warning" or "beats estimates" have specific meanings in finance that FinBERT is trained to handle correctly.

---

## Correlation analysis

Monthly sentiment scores are joined to monthly return data fetched from yfinance. Two correlations are computed per ticker:

| Metric | Description |
|---|---|
| **Contemporaneous correlation (T)** | Sentiment in month T vs return in month T. Typically moderate and positive — news reflects events that already moved prices |
| **Forward correlation (T+1)** | Sentiment in month T vs return in month T+1. Tests whether sentiment predicts future returns |

**Expected finding:** contemporaneous correlation is usually stronger than forward correlation. Weak forward correlation is consistent with the semi-strong form of the Efficient Market Hypothesis — publicly available news is already priced in. A meaningful forward correlation would suggest either a market inefficiency or a sentiment lag in a specific asset.

Both pooled (across all tickers) and per-ticker correlations are reported.

---

## Data quality notes

**Headline relevance:** yfinance aggregates broadly related financial news rather than strictly company-specific stories. Some headlines tagged to a ticker may be only tangentially related, which dilutes sentiment signal. This is noted as a limitation in any write-up.

**Headline volume:** yfinance returns approximately 10 headlines per ticker per pull. With daily runs this produces 70–100 headlines per month per ticker — sufficient for indicative analysis but not a large-sample study.

**Deduplication:** headlines are deduplicated on URL and on ticker + date + title combination. Running the script multiple times per day is safe.

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Yahoo Finance news and market data |
| `transformers` | HuggingFace library — loads and runs FinBERT |
| `torch` | PyTorch backend required by transformers |
| `pandas` | Data manipulation and aggregation |
| `numpy` | Numerical operations |
| `openpyxl` | Excel workbook creation and formatting |
| `xlsxwriter` | Chart generation — produces clean Excel-compatible chart XML |

---
---

## Relationship to the Equity Analyser

This project is designed as a companion to the [Automated Equity Analyser](../analyser/README.md). Both use the same ticker set. The equity analyser provides the price and return data; the sentiment analyser provides the news signal. Together they form a two-tool suite covering quantitative data pipeline and NLP-based signal extraction — two distinct skill sets built on the same underlying assets.
