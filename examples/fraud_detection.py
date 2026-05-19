"""
DRUID Example: Fraud Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

End-to-end fraud detection workflow using DRUID,
End-to-end fraud detection workflow.
"""

import druid
from druid.preprocessing import bin_numeric, split_data
from druid.modelling import classification_summary, feature_importance

# Configure AI (Claude in this case)
druid.configure(provider="anthropic")

# Load from BigQuery
ds = druid.load(
    "pastel-data-science-general.training_data.central_fraud_training_preprocessed",
    target="fraud_label",
    name="fraud_training",
)

# AI schema inspection
inspection = druid.inspect(ds)

# Explore the data
eda = druid.explore(ds, question="What patterns distinguish fraudulent transactions?")

# Custom preprocessing with more control
from druid.preprocessing import (
    auto_clean,
    featurize_datetime,
    treat_outliers,
    encode_categoricals,
)

ds = auto_clean(ds)
ds = featurize_datetime(ds, columns=["txn_date", "date_joined"])
ds = bin_numeric(
    ds,
    column="age",
    bins=[0, 2, 12, 25, 45, 65, 120],
    labels=["baby", "child", "young_adult", "mid_age", "senior", "elderly"],
)
ds = treat_outliers(ds)
ds = encode_categoricals(ds)

# Run the experiment
results = druid.experiment(ds, task_type="classification")

# Detailed evaluation of the best model
X_train, X_test, y_train, y_test = split_data(ds)
best = results["best_model"]
summary = classification_summary(best, X_test.values, y_test.values)
print(summary["report"])

# Feature importance
fi_fig = feature_importance(best, list(X_train.columns))

# AI recommendations
druid.recommend(ds, results)
