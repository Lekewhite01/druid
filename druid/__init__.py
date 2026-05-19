"""
DRUID — AI-powered data science assistant.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From raw data to trained models, guided by AI.

Quick start::

    import druid

    # Load any data source
    ds = druid.load("data.csv", target="fraud_label")

    # AI inspects the schema and flags issues
    druid.inspect(ds)

    # Guided EDA with AI commentary
    druid.explore(ds)

    # One-line preprocessing
    ds = druid.prepare(ds)

    # Train and compare models
    results = druid.experiment(ds)

    # AI recommends next steps
    druid.recommend(ds, results)
"""

__version__ = "0.1.1"

from druid.core.config import AIConfig, DruidConfig
from druid.core.dataset import DruidDataset
from druid.loaders.registry import auto_load

# Module-level default config (users can override)
_config = DruidConfig()


# ------------------------------------------------------------------
# Top-level convenience API
# ------------------------------------------------------------------

def configure(
    provider: str = "openai",
    api_key: str | None = None,
    model: str | None = None,
    **kwargs,
) -> DruidConfig:
    """
    Configure DRUID's AI provider and settings.

    Parameters
    ----------
    provider : str
        ``"openai"``, ``"anthropic"``, or ``"google"``.
    api_key : str, optional
        API key.  Falls back to environment variables if omitted.
    model : str, optional
        Model name override.
    **kwargs
        Additional config options.

    Returns
    -------
    DruidConfig
        The updated global config.

    Examples
    --------
    >>> druid.configure(provider="anthropic", api_key="sk-...")
    >>> druid.configure(provider="openai")  # uses OPENAI_API_KEY env var
    """
    global _config
    _config.ai = AIConfig(provider=provider, api_key=api_key, model=model)
    return _config


def load(
    source: str,
    target: str | None = None,
    name: str | None = None,
    **kwargs,
) -> DruidDataset:
    """
    Load data from any source into a DruidDataset.

    Automatically detects the format: CSV, Parquet, Excel, JSON,
    BigQuery tables, and SQL queries.

    Parameters
    ----------
    source : str
        File path, table reference, or SQL query.
    target : str, optional
        Name of the target / label column.
    name : str, optional
        Human-friendly dataset name.
    **kwargs
        Passed to the underlying loader.

    Returns
    -------
    DruidDataset

    Examples
    --------
    >>> ds = druid.load("transactions.csv", target="is_fraud")
    >>> ds = druid.load("project.dataset.table", target="label")
    >>> ds = druid.load("data.parquet")
    """
    from pathlib import Path

    df = auto_load(source, **kwargs)
    dataset_name = name or Path(source).stem if "." in source else source
    ds = DruidDataset(df=df, name=dataset_name, target=target, config=_config)
    return ds


def inspect(ds: DruidDataset) -> dict:
    """
    AI-powered schema inspection.

    Analyses the dataset profile and returns observations about
    data quality, column types, and recommended actions.

    Parameters
    ----------
    ds : DruidDataset

    Returns
    -------
    dict
        Inspection results including AI commentary.
    """
    from druid.utils.display import print_header, print_profile_summary

    print_header("Schema Inspection")
    profile = ds.profile(force=True)
    print_profile_summary(profile)

    # If AI is configured, get AI analysis
    if ds.config.ai.is_configured:
        from druid.ai.schema_inspector import inspect_schema
        result = inspect_schema(ds)
        print(f"\n{result['raw_response']}")
        return result
    else:
        print("\nTip: Configure an AI provider for deeper analysis:")
        print('  druid.configure(provider="openai")  # or "anthropic", "google"')
        return {"profile": profile, "raw_response": None}


def explore(
    ds: DruidDataset,
    question: str | None = None,
    save_dir: str | None = None,
) -> dict:
    """
    Exploratory data analysis with optional AI guidance.

    Generates visualisations and, if AI is configured, provides
    AI commentary on findings.

    Parameters
    ----------
    ds : DruidDataset
    question : str, optional
        Specific question to guide the analysis.
    save_dir : str, optional
        Directory to save plot images.

    Returns
    -------
    dict
        Plots and AI insights.
    """
    from druid.eda.profiler import full_profile
    from druid.eda.visualizer import auto_eda_plots
    from druid.utils.display import print_header

    print_header("Exploratory Data Analysis")
    profile = full_profile(ds)
    plots = auto_eda_plots(ds, save_dir=save_dir)

    result = {"profile": profile, "plots": plots}

    if ds.config.ai.is_configured:
        from druid.eda.insights import generate_eda_guidance
        guidance = generate_eda_guidance(ds, question=question)
        print(f"\n{guidance}")
        result["ai_guidance"] = guidance

    return result


def prepare(ds: DruidDataset, auto: bool = True) -> DruidDataset:
    """
    Preprocess the dataset for modelling.

    Runs cleaning, feature engineering, encoding, and dtype
    standardisation.

    Parameters
    ----------
    ds : DruidDataset
    auto : bool
        If True, uses sensible defaults for the full pipeline.

    Returns
    -------
    DruidDataset
        Preprocessed dataset.
    """
    from druid.preprocessing.pipeline import prepare as _prepare
    from druid.utils.display import print_header

    print_header("Preprocessing")
    ds = _prepare(ds, auto=auto)
    print(f"Preprocessed: {ds.shape[0]:,} rows × {ds.shape[1]} columns")
    return ds


def experiment(
    ds: DruidDataset,
    task_type: str | None = None,
) -> dict:
    """
    Train multiple models and compare performance.

    Parameters
    ----------
    ds : DruidDataset
        Must have ``target`` set.
    task_type : str, optional
        ``"classification"`` or ``"regression"``.  Auto-detected
        from the target if omitted.

    Returns
    -------
    dict
        Experiment results with leaderboard.
    """
    from druid.modelling.experiment import run_experiment
    from druid.preprocessing.pipeline import split_data
    from druid.utils.display import print_header, print_leaderboard

    if ds.target is None:
        raise ValueError("Set ds.target before running experiments")

    # Auto-detect task type
    if task_type is None:
        nunique = ds.df[ds.target].nunique()
        task_type = "classification" if nunique <= 20 else "regression"

    print_header(f"Model Experiment ({task_type})")

    X_train, X_test, y_train, y_test = split_data(ds)
    results = run_experiment(
        X_train, X_test, y_train, y_test,
        task_type=task_type,
        dataset=ds,
    )

    print_leaderboard(results["leaderboard"])
    return results


def recommend(ds: DruidDataset, experiment_results: dict | None = None) -> str:
    """
    Get AI-powered recommendations for next steps.

    Parameters
    ----------
    ds : DruidDataset
    experiment_results : dict, optional
        Results from ``druid.experiment()``.

    Returns
    -------
    str
        AI recommendations.
    """
    from druid.modelling.recommender import recommend_models
    from druid.utils.display import print_header

    if not ds.config.ai.is_configured:
        return "Configure an AI provider first: druid.configure(provider='openai')"

    task_type = experiment_results.get("task_type", "classification") if experiment_results else "classification"

    print_header("AI Recommendations")
    response = recommend_models(ds, task_type=task_type, experiment_results=experiment_results)
    print(response)
    return response
