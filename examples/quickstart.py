"""
DRUID Quick Start
~~~~~~~~~~~~~~~~~

This example demonstrates the core DRUID workflow:
load → inspect → explore → prepare → experiment → recommend.
"""

import druid

# 1. Configure your AI provider (pick one)
druid.configure(provider="openai")          # Uses OPENAI_API_KEY env var
# druid.configure(provider="anthropic")     # Uses ANTHROPIC_API_KEY env var
# druid.configure(provider="google")        # Uses GOOGLE_API_KEY env var

# 2. Load your data (auto-detects format)
ds = druid.load("your_data.csv", target="label_column")

# 3. AI inspects the schema — flags data quality issues, type mismatches
druid.inspect(ds)

# 4. Guided EDA — generates plots, AI explains the findings
druid.explore(ds, save_dir="./eda_output")

# 5. One-line preprocessing — clean, encode, impute, handle outliers
ds = druid.prepare(ds)

# 6. Train and compare models automatically
results = druid.experiment(ds)

# 7. AI recommends next steps based on the results
druid.recommend(ds, results)

# 8. Export the full session as a reproducible script
from druid.utils import export_script
export_script(ds, "reproducible_pipeline.py")
