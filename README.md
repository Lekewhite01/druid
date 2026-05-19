# DRUID

**AI-powered data science assistant — from raw data to trained models.**

DRUID is a Python library that combines automated data analysis with large language models to guide you through the entire machine learning workflow. Load your data, and DRUID inspects it, surfaces quality issues, recommends preprocessing steps, generates visualisations, runs model experiments, and explains the results — all with a few lines of code.

## Why DRUID?

Data scientists spend 60–80% of their time on data understanding and preparation. DRUID automates the mechanical parts while using AI to surface the insights that matter.

- **Load anything** — CSV, Parquet, Excel, JSON, BigQuery, PostgreSQL. Auto-detected.
- **AI schema inspection** — Spots data quality issues, type mismatches, and potential label leakage before you start.
- **Guided EDA** — Generates the right visualisations for your data and explains what they mean.
- **One-line preprocessing** — Handles missing values, outliers, encoding, and feature engineering with sensible defaults.
- **Model experimentation** — Trains and compares multiple algorithms in one call. Shows a ranked leaderboard.
- **Multi-provider AI** — Works with OpenAI, Anthropic (Claude), or Google (Gemini). Bring your own API key.
- **Fully reproducible** — Every step is logged. Export the session as a standalone Python script anytime.

## Installation

```bash
pip install druid-ai
```

With AI providers:

```bash
pip install druid-ai[openai]      # OpenAI (GPT-4o, etc.)
pip install druid-ai[anthropic]   # Anthropic (Claude)
pip install druid-ai[google]      # Google (Gemini)
pip install druid-ai[all]         # Everything
```

## Quick Start

```python
import druid

# Configure your AI provider
druid.configure(provider="anthropic")  # or "openai", "google"

# Load data (auto-detects format)
ds = druid.load("transactions.csv", target="is_fraud")

# AI inspects the schema
druid.inspect(ds)

# Exploratory analysis with AI commentary
druid.explore(ds)

# Preprocess in one line
ds = druid.prepare(ds)

# Train and compare models
results = druid.experiment(ds)

# AI recommends next steps
druid.recommend(ds, results)
```

That's the full workflow. Five function calls from raw data to model comparison.

## Supported Data Sources

| Source | Example |
|---|---|
| CSV / TSV | `druid.load("data.csv")` |
| Parquet | `druid.load("data.parquet")` |
| Excel | `druid.load("data.xlsx")` |
| JSON / JSONL | `druid.load("data.json")` |
| BigQuery | `druid.load("project.dataset.table")` |
| SQL query | `druid.load("SELECT * FROM users", loaders=[SQLLoader(conn_str)])` |

## AI Providers

DRUID works with any of the three major LLM providers. Set your API key as an environment variable or pass it directly:

```python
# Environment variable (recommended)
# export OPENAI_API_KEY=sk-...
# export ANTHROPIC_API_KEY=sk-ant-...
# export GOOGLE_API_KEY=...

druid.configure(provider="openai")

# Or pass directly
druid.configure(provider="anthropic", api_key="sk-ant-...")
```

DRUID works without an API key too — you get automated profiling, visualisations, preprocessing, and model experiments. The AI layer adds schema analysis, natural-language insights, and model recommendations on top.

## Detailed Usage

### Loading Data

```python
ds = druid.load("data.csv", target="label", name="my_dataset")
```

The `DruidDataset` wraps your DataFrame with metadata, profiling, and session tracking. Access the raw data anytime with `ds.df`.

### Schema Inspection

```python
result = druid.inspect(ds)
```

This computes a statistical profile and, if AI is configured, sends it to the LLM for analysis. The AI flags issues like mislabelled columns, suspicious distributions, and potential data leakage.

### Exploratory Analysis

```python
eda = druid.explore(ds, question="What separates fraud from non-fraud?")
```

Generates missing-value plots, distribution histograms, correlation heatmaps, and target analysis. The optional `question` parameter focuses the AI's commentary.

### Preprocessing

```python
# Automatic — sensible defaults
ds = druid.prepare(ds)

# Manual — full control
from druid.preprocessing import (
    auto_clean, featurize_datetime, treat_outliers,
    encode_categoricals, bin_numeric,
)

ds = auto_clean(ds)
ds = featurize_datetime(ds, columns=["created_at"])
ds = treat_outliers(ds, method="iqr", fold=1.5)
ds = encode_categoricals(ds, high_card_threshold=15)
```

### Model Experiments

```python
results = druid.experiment(ds)
```

Trains Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, Decision Tree, KNN, and SGD out of the box. If `xgboost` or `lightgbm` are installed, those are included automatically.

The leaderboard shows accuracy, F1, precision, recall, ROC AUC, and training time for each model.

### AI Recommendations

```python
druid.recommend(ds, results)
```

Passes the experiment results to the AI, which recommends which model to invest in, specific hyperparameters to tune, and strategies for handling class imbalance or other issues.

### Reproducibility

```python
# Export session as a Python script
from druid.utils import export_script
export_script(ds, "pipeline.py")

# Save and reload datasets
from druid.utils import save_dataset, load_dataset
save_dataset(ds, "checkpoint.pkl")
ds = load_dataset("checkpoint.pkl")
```

## Configuration

```python
from druid import DruidConfig, AIConfig

config = DruidConfig(
    ai=AIConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        temperature=0.3,
    ),
)

# Or load from YAML
config = DruidConfig.from_yaml("druid_config.yaml")
```

## Project Structure

```
druid/
├── core/          # DruidDataset, config, session tracking
├── loaders/       # File, database, and auto-detect loaders
├── ai/            # LLM providers, prompts, schema inspector
├── eda/           # Profiling, visualisation, AI insights
├── preprocessing/ # Cleaning, transforms, encoding, pipelines
├── modelling/     # Experiments, evaluation, AI recommendations
└── utils/         # Display, sampling, export tools
```

## Requirements

- Python ≥ 3.9
- pandas, numpy, scikit-learn, matplotlib, seaborn
- feature-engine, missingno, rich, httpx, pyyaml

AI providers and database connectors are optional extras.

## Development

```bash
git clone https://github.com/Lekewhite01/druid.git
cd druid
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE) for details.
