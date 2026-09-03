# =============================================================================
# 11_final_model_specification_table.py
#
# FINAL MODEL-SPECIFICATION ARCHITECTURE
#
# Dissertation:
# Do Social Media Sentiment Signals Improve the Prediction of
# Cryptocurrency Returns Beyond Traditional Market Indicators?
# Evidence from Bitcoin and Ethereum
#
# PURPOSE
# -------
# This script documents the FINAL empirical model architecture.
#
# It DOES NOT estimate regressions.
# It DOES NOT require Reddit data to be available yet.
#
# Primary explanatory models:
#
# M1 = Baseline controls only
# M2 = Controls + Reddit activity
# M3 = Controls + Reddit sentiment
# M4 = Controls + Reddit activity + sentiment
#
# Robustness:
#
# R1 = Baseline controls + cross-crypto lagged return
# R2 = Full Reddit model + cross-crypto lagged return
#
# Alternative lag robustness:
#
# t-1 = primary
# t-2, t-3, t-7 = robustness
#
# Forecast architecture:
#
# F1 = Controls only
# F2 = Controls + Reddit activity
# F3 = Controls + Reddit sentiment
# F4 = Controls + Reddit activity + sentiment
#
# IMPORTANT:
# Reddit activity and Reddit sentiment are intentionally separate.
# =============================================================================


from pathlib import Path
import pandas as pd


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation"
)

DATA_FILE = (
    PROJECT_ROOT
    / "data_processed"
    / "final_forecast_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "final_model_specification"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. STUDY SETTINGS
# =============================================================================

STUDY_START = "2021-01-01"
STUDY_END = "2025-12-31"

FREQUENCY = "Daily"

PRIMARY_LAG = 1

ALTERNATIVE_LAGS = [2, 3, 7]

HAC_MAXLAGS = 7

FORECAST_HORIZON = "One day ahead"

FORECAST_WINDOW = "Expanding window"

OOS_START = "2024-01-02"
OOS_END = "2025-12-31"


# =============================================================================
# 3. CORRECT INFORMATION-ALIGNED TRADITIONAL-MARKET CONTROLS
# =============================================================================

TRADITIONAL_CONTROLS = [
    "Lagged_SP500_Return_Aligned",
    "Lagged_VIX_Change_Aligned",
    "Lagged_Gold_Return_Aligned",
    "Lagged_DXY_Return_Aligned",
    "Lagged_US10Y_Change_Aligned",
]


# =============================================================================
# 4. OLD NON-ALIGNED VARIABLES THAT MUST NOT BE USED
# =============================================================================

OLD_NONALIGNED_CONTROLS = [
    "Lagged_SP500_Return",
    "Lagged_VIX_Change",
    "Lagged_Gold_Return",
    "Lagged_DXY_Return",
    "Lagged_US10Y_Change",
]


# =============================================================================
# 5. ASSET-SPECIFIC VARIABLES
# =============================================================================

ASSETS = {
    "BTC": {
        "name": "Bitcoin",
        "dependent": "BTC_Return",
        "own_return": "BTC_Lagged_Return",
        "volume": "Lagged_Log_BTC_Volume",
        "activity": "Lagged_Log_BTC_Reddit_Post_Count",
        "sentiment": "Lagged_BTC_Reddit_Sentiment",
        "cross_crypto": "ETH_Lagged_Return",
    },

    "ETH": {
        "name": "Ethereum",
        "dependent": "ETH_Return",
        "own_return": "ETH_Lagged_Return",
        "volume": "Lagged_Log_ETH_Volume",
        "activity": "Lagged_Log_ETH_Reddit_Post_Count",
        "sentiment": "Lagged_ETH_Reddit_Sentiment",
        "cross_crypto": "BTC_Lagged_Return",
    },
}


# =============================================================================
# 6. MODEL DEFINITIONS
# =============================================================================

MODELS = {
    "M1": {
        "name": "Baseline controls only",
        "activity": False,
        "sentiment": False,
        "cross_crypto": False,
        "role": "Primary",
        "purpose": (
            "Benchmark explanatory model containing no Reddit variables."
        ),
    },

    "M2": {
        "name": "Controls + Reddit activity",
        "activity": True,
        "sentiment": False,
        "cross_crypto": False,
        "role": "Primary",
        "purpose": (
            "Tests whether Reddit activity or attention adds information "
            "beyond the baseline market controls."
        ),
    },

    "M3": {
        "name": "Controls + Reddit sentiment",
        "activity": False,
        "sentiment": True,
        "cross_crypto": False,
        "role": "Primary",
        "purpose": (
            "Tests whether Reddit sentiment adds information beyond "
            "the baseline market controls."
        ),
    },

    "M4": {
        "name": "Controls + Reddit activity + sentiment",
        "activity": True,
        "sentiment": True,
        "cross_crypto": False,
        "role": "Primary",
        "purpose": (
            "Tests whether Reddit sentiment remains informative after "
            "separately controlling for Reddit activity."
        ),
    },

    "R1": {
        "name": "Controls + cross-crypto return",
        "activity": False,
        "sentiment": False,
        "cross_crypto": True,
        "role": "Robustness",
        "purpose": (
            "Tests baseline robustness after controlling for the other "
            "cryptocurrency's lagged return."
        ),
    },

    "R2": {
        "name": (
            "Controls + Reddit activity + sentiment "
            "+ cross-crypto return"
        ),
        "activity": True,
        "sentiment": True,
        "cross_crypto": True,
        "role": "Robustness",
        "purpose": (
            "Tests whether the full Reddit result survives control for "
            "cross-cryptocurrency return dynamics."
        ),
    },
}


# =============================================================================
# 7. FORECAST MODEL DEFINITIONS
# =============================================================================

FORECAST_MODELS = {
    "F1": {
        "name": "Controls only",
        "activity": False,
        "sentiment": False,
    },

    "F2": {
        "name": "Controls + Reddit activity",
        "activity": True,
        "sentiment": False,
    },

    "F3": {
        "name": "Controls + Reddit sentiment",
        "activity": False,
        "sentiment": True,
    },

    "F4": {
        "name": "Controls + Reddit activity + sentiment",
        "activity": True,
        "sentiment": True,
    },
}


# =============================================================================
# 8. HELPER FUNCTIONS
# =============================================================================

def print_section(title):
    """Print a formatted console section heading."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def yes_no(value):
    """Convert Boolean to dissertation-friendly Yes/No."""

    if value:
        return "Yes"

    return "No"


def get_baseline_predictors(asset):
    """
    Return the exact baseline predictor set for BTC or ETH.
    """

    settings = ASSETS[asset]

    predictors = [
        settings["own_return"],
        settings["volume"],
    ]

    predictors.extend(TRADITIONAL_CONTROLS)

    return predictors


def get_model_predictors(asset, model_code):
    """
    Return the exact predictor set for an explanatory model.
    """

    settings = ASSETS[asset]
    model = MODELS[model_code]

    predictors = get_baseline_predictors(asset).copy()

    if model["activity"]:
        predictors.append(settings["activity"])

    if model["sentiment"]:
        predictors.append(settings["sentiment"])

    if model["cross_crypto"]:
        predictors.append(settings["cross_crypto"])

    return predictors


def get_forecast_predictors(asset, model_code):
    """
    Return the exact predictor set for a forecast model.
    """

    settings = ASSETS[asset]
    model = FORECAST_MODELS[model_code]

    predictors = get_baseline_predictors(asset).copy()

    if model["activity"]:
        predictors.append(settings["activity"])

    if model["sentiment"]:
        predictors.append(settings["sentiment"])

    return predictors


# =============================================================================
# 9. START
# =============================================================================

print_section(
    "FINAL MODEL-SPECIFICATION ARCHITECTURE"
)

print("\nStudy period:")
print(f"{STUDY_START} to {STUDY_END}")

print("\nFrequency:")
print(FREQUENCY)

print("\nPrimary lag:")
print("t-1")

print("\nAlternative robustness lags:")
print("t-2, t-3, t-7")

print("\nHAC/Newey-West maximum lag:")
print(HAC_MAXLAGS)

print("\nForecast horizon:")
print(FORECAST_HORIZON)

print("\nForecast estimation window:")
print(FORECAST_WINDOW)

print("\nPlanned OOS period:")
print(f"{OOS_START} to {OOS_END}")

print(
    "\nThis script defines the final model architecture only. "
    "It does not estimate regressions or forecasts."
)


# =============================================================================
# 10. CURRENT DATASET CHECK
# =============================================================================

print_section(
    "CURRENT DATASET CHECK"
)

if DATA_FILE.exists():

    dataset_header = pd.read_csv(
        DATA_FILE,
        nrows=1,
    )

    current_columns = set(
        dataset_header.columns
    )

    print("\nDataset found:")
    print(DATA_FILE)

    print("\nNumber of columns:")
    print(len(current_columns))

else:

    current_columns = set()

    print("\nWARNING:")
    print("final_forecast_dataset.csv was not found.")

    print(
        "\nThis does NOT prevent the model-specification "
        "tables from being created."
    )


# =============================================================================
# 11. CHECK BASELINE VARIABLES
# =============================================================================

print_section(
    "BASELINE VARIABLE CHECK"
)

baseline_required = set()

for asset in ["BTC", "ETH"]:

    settings = ASSETS[asset]

    baseline_required.add(
        settings["dependent"]
    )

    baseline_required.update(
        get_baseline_predictors(asset)
    )


if DATA_FILE.exists():

    missing_baseline = sorted(
        baseline_required
        - current_columns
    )

    if len(missing_baseline) == 0:

        print(
            "\nPASS: All required baseline variables "
            "exist in the current dataset."
        )

    else:

        print(
            "\nWARNING: Some expected baseline variables "
            "are missing:"
        )

        for variable in missing_baseline:
            print(f"  - {variable}")

else:

    missing_baseline = []


# =============================================================================
# 12. PRINT EXACT BASELINE SPECIFICATIONS
# =============================================================================

print_section(
    "EXACT BASELINE SPECIFICATIONS"
)

for asset in ["BTC", "ETH"]:

    print(f"\n{asset} dependent variable:")
    print(ASSETS[asset]["dependent"])

    print(f"\n{asset} baseline predictors:")

    for variable in get_baseline_predictors(asset):
        print(f"  - {variable}")


# =============================================================================
# 13. DETAILED EXPLANATORY MODEL TABLE
# =============================================================================

print_section(
    "CREATING DETAILED EXPLANATORY MODEL TABLE"
)

detailed_rows = []

for asset in ["BTC", "ETH"]:

    settings = ASSETS[asset]

    for model_code in MODELS:

        model = MODELS[model_code]

        predictors = get_model_predictors(
            asset,
            model_code,
        )

        for predictor_number, predictor in enumerate(
            predictors,
            start=1,
        ):

            if predictor == settings["own_return"]:
                variable_group = "Own lagged return"

            elif predictor == settings["volume"]:
                variable_group = "Crypto trading volume"

            elif predictor in TRADITIONAL_CONTROLS:
                variable_group = "Traditional market control"

            elif predictor == settings["activity"]:
                variable_group = "Reddit activity / attention"

            elif predictor == settings["sentiment"]:
                variable_group = "Reddit sentiment"

            elif predictor == settings["cross_crypto"]:
                variable_group = "Cross-crypto robustness"

            else:
                variable_group = "Other"

            row = {
                "Asset": asset,
                "Asset_Name": settings["name"],
                "Model": model_code,
                "Model_Name": model["name"],
                "Role": model["role"],
                "Dependent_Variable": settings["dependent"],
                "Predictor_Number": predictor_number,
                "Predictor": predictor,
                "Variable_Group": variable_group,
                "Primary_Lag": "t-1",
                "Purpose": model["purpose"],
            }

            detailed_rows.append(row)


detailed_df = pd.DataFrame(
    detailed_rows
)

print("\nNumber of detailed rows:")
print(len(detailed_df))


# =============================================================================
# 14. COMPACT EXPLANATORY MODEL TABLE
# =============================================================================

print_section(
    "CREATING COMPACT EXPLANATORY MODEL TABLE"
)

compact_rows = []

for asset in ["BTC", "ETH"]:

    settings = ASSETS[asset]

    for model_code in MODELS:

        model = MODELS[model_code]

        predictors = get_model_predictors(
            asset,
            model_code,
        )

        row = {
            "Asset": settings["name"],
            "Model": model_code,
            "Specification": model["name"],
            "Dependent_Variable": settings["dependent"],
            "Predictors": "; ".join(predictors),
            "Reddit_Activity": yes_no(
                model["activity"]
            ),
            "Reddit_Sentiment": yes_no(
                model["sentiment"]
            ),
            "Cross_Crypto_Return": yes_no(
                model["cross_crypto"]
            ),
            "Lag": "t-1",
            "Role": model["role"],
            "Purpose": model["purpose"],
        }

        compact_rows.append(row)


compact_df = pd.DataFrame(
    compact_rows
)

display_columns = [
    "Asset",
    "Model",
    "Specification",
    "Reddit_Activity",
    "Reddit_Sentiment",
    "Cross_Crypto_Return",
    "Role",
]

print()
print(
    compact_df[
        display_columns
    ].to_string(
        index=False
    )
)


# =============================================================================
# 15. FINAL MODEL-INCLUSION MATRIX
# =============================================================================

print_section(
    "FINAL MODEL-INCLUSION MATRIX"
)

inclusion_rows = []

for model_code in MODELS:

    model = MODELS[model_code]

    row = {
        "Model": model_code,
        "Specification": model["name"],
        "Own_Lagged_Return": "Yes",
        "Lagged_Log_Volume": "Yes",
        "Aligned_Traditional_Controls": "Yes",
        "Reddit_Activity": yes_no(
            model["activity"]
        ),
        "Reddit_Sentiment": yes_no(
            model["sentiment"]
        ),
        "Cross_Crypto_Return": yes_no(
            model["cross_crypto"]
        ),
        "Role": model["role"],
    }

    inclusion_rows.append(row)


inclusion_df = pd.DataFrame(
    inclusion_rows
)

print()
print(
    inclusion_df.to_string(
        index=False
    )
)


# =============================================================================
# 16. REDDIT ACTIVITY VS SENTIMENT DEFINITIONS
# =============================================================================

print_section(
    "REDDIT ACTIVITY VS SENTIMENT"
)

reddit_concept_rows = [
    {
        "Concept": "Reddit activity",
        "Economic_Concept": "Attention / discussion intensity",
        "BTC_Variable": ASSETS["BTC"]["activity"],
        "ETH_Variable": ASSETS["ETH"]["activity"],
        "Construction": (
            "log(1 + daily retained Reddit post count)"
        ),
        "Primary_Timing": "t-1",
        "Interpretation": (
            "Measures how much retained Reddit discussion "
            "occurred, not whether it was positive or negative."
        ),
    },

    {
        "Concept": "Reddit sentiment",
        "Economic_Concept": "Tone / polarity of discussion",
        "BTC_Variable": ASSETS["BTC"]["sentiment"],
        "ETH_Variable": ASSETS["ETH"]["sentiment"],
        "Construction": (
            "Daily aggregate sentiment score from the "
            "retained asset-specific Reddit sample"
        ),
        "Primary_Timing": "t-1",
        "Interpretation": (
            "Measures the tone of retained Reddit discussion "
            "separately from the amount of discussion."
        ),
    },
]

reddit_concepts_df = pd.DataFrame(
    reddit_concept_rows
)

print()
print(
    reddit_concepts_df.to_string(
        index=False
    )
)


# =============================================================================
# 17. REDDIT VARIABLE AVAILABILITY
# =============================================================================

print_section(
    "REDDIT VARIABLE AVAILABILITY"
)

reddit_status_rows = []

for asset in ["BTC", "ETH"]:

    settings = ASSETS[asset]

    concepts = [
        ("Activity", settings["activity"]),
        ("Sentiment", settings["sentiment"]),
    ]

    for concept_name, variable in concepts:

        available = (
            variable in current_columns
        )

        if available:
            status = "Available"
        else:
            status = "Pending Reddit data"

        row = {
            "Asset": asset,
            "Concept": concept_name,
            "Variable": variable,
            "Available_In_Current_Dataset": available,
            "Status": status,
        }

        reddit_status_rows.append(row)


reddit_status_df = pd.DataFrame(
    reddit_status_rows
)

print()
print(
    reddit_status_df.to_string(
        index=False
    )
)


# =============================================================================
# 18. MODEL COMPARISON LOGIC
# =============================================================================

print_section(
    "MODEL COMPARISON LOGIC"
)

comparison_rows = [
    {
        "Comparison": "M1 vs M2",
        "Added_Information": "Reddit activity",
        "Question": (
            "Does Reddit activity add explanatory information "
            "beyond the baseline market controls?"
        ),
        "Interpretation": "Activity / attention comparison",
    },

    {
        "Comparison": "M1 vs M3",
        "Added_Information": "Reddit sentiment",
        "Question": (
            "Does Reddit sentiment add explanatory information "
            "beyond the baseline market controls?"
        ),
        "Interpretation": "Primary sentiment comparison",
    },

    {
        "Comparison": "M2 vs M4",
        "Added_Information": "Reddit sentiment",
        "Question": (
            "Does Reddit sentiment add information once Reddit "
            "activity is already controlled for?"
        ),
        "Interpretation": (
            "Activity-adjusted sentiment comparison"
        ),
    },

    {
        "Comparison": "M3 vs M4",
        "Added_Information": "Reddit activity",
        "Question": (
            "Does Reddit activity add information once Reddit "
            "sentiment is already controlled for?"
        ),
        "Interpretation": (
            "Sentiment-adjusted activity comparison"
        ),
    },

    {
        "Comparison": "M1 vs R1",
        "Added_Information": "Cross-crypto lagged return",
        "Question": (
            "Does adding the other cryptocurrency's lagged "
            "return alter the baseline conclusions?"
        ),
        "Interpretation": "Baseline cross-crypto robustness",
    },

    {
        "Comparison": "M4 vs R2",
        "Added_Information": "Cross-crypto lagged return",
        "Question": (
            "Does the full Reddit result survive control for "
            "the other cryptocurrency's lagged return?"
        ),
        "Interpretation": "Full cross-crypto robustness",
    },

    {
        "Comparison": "t-1 vs t-2/t-3/t-7",
        "Added_Information": "Alternative lag horizons",
        "Question": (
            "Are the conclusions sensitive to the primary "
            "one-day lag assumption?"
        ),
        "Interpretation": "Alternative-lag robustness",
    },
]

comparison_df = pd.DataFrame(
    comparison_rows
)

print()
print(
    comparison_df.to_string(
        index=False
    )
)


# =============================================================================
# 19. HYPOTHESIS-TO-MODEL MAPPING
# =============================================================================

print_section(
    "HYPOTHESIS-TO-MODEL MAPPING"
)

hypothesis_rows = [
    {
        "Hypothesis": "H1",
        "Asset": "Bitcoin",
        "Research_Focus": (
            "Lagged Bitcoin Reddit sentiment and "
            "subsequent Bitcoin returns"
        ),
        "Primary_Model": "M3",
        "Conditional_Model": "M4",
        "Primary_Comparison": "M1 vs M3",
        "Activity_Adjusted_Comparison": "M2 vs M4",
    },

    {
        "Hypothesis": "H2",
        "Asset": "Ethereum",
        "Research_Focus": (
            "Lagged Ethereum Reddit sentiment and "
            "subsequent Ethereum returns"
        ),
        "Primary_Model": "M3",
        "Conditional_Model": "M4",
        "Primary_Comparison": "M1 vs M3",
        "Activity_Adjusted_Comparison": "M2 vs M4",
    },

    {
        "Hypothesis": "H3",
        "Asset": "Bitcoin",
        "Research_Focus": (
            "Whether lagged Bitcoin Reddit sentiment improves "
            "one-day-ahead OOS Bitcoin return forecasts"
        ),
        "Primary_Model": "F3",
        "Conditional_Model": "F4",
        "Primary_Comparison": "F1 vs F3",
        "Activity_Adjusted_Comparison": "F2 vs F4",
    },

    {
        "Hypothesis": "H4",
        "Asset": "Ethereum",
        "Research_Focus": (
            "Whether lagged Ethereum Reddit sentiment improves "
            "one-day-ahead OOS Ethereum return forecasts"
        ),
        "Primary_Model": "F3",
        "Conditional_Model": "F4",
        "Primary_Comparison": "F1 vs F3",
        "Activity_Adjusted_Comparison": "F2 vs F4",
    },

    {
        "Hypothesis": "H5",
        "Asset": "BTC versus ETH",
        "Research_Focus": (
            "Whether the sentiment-return relationship "
            "differs between Bitcoin and Ethereum"
        ),
        "Primary_Model": "M3",
        "Conditional_Model": "M4",
        "Primary_Comparison": (
            "Formal BTC-versus-ETH sentiment coefficient test"
        ),
        "Activity_Adjusted_Comparison": (
            "Formal BTC-versus-ETH comparison using M4"
        ),
    },
]

hypothesis_df = pd.DataFrame(
    hypothesis_rows
)

print()
print(
    hypothesis_df.to_string(
        index=False
    )
)


# =============================================================================
# 20. CROSS-CRYPTO ROBUSTNESS ARCHITECTURE
# =============================================================================

print_section(
    "CROSS-CRYPTO ROBUSTNESS ARCHITECTURE"
)

cross_crypto_rows = []

for asset in ["BTC", "ETH"]:

    settings = ASSETS[asset]

    row = {
        "Asset": settings["name"],
        "Dependent_Variable": settings["dependent"],
        "Cross_Crypto_Control": settings["cross_crypto"],
        "Baseline_Robustness_Model": "R1",
        "Full_Reddit_Robustness_Model": "R2",
        "Purpose": (
            "Controls for broader cryptocurrency-market "
            "return dynamics and omitted cross-market effects."
        ),
    }

    cross_crypto_rows.append(row)


cross_crypto_df = pd.DataFrame(
    cross_crypto_rows
)

print()
print(
    cross_crypto_df.to_string(
        index=False
    )
)


# =============================================================================
# 21. ALTERNATIVE-LAG ROBUSTNESS ARCHITECTURE
# =============================================================================

print_section(
    "ALTERNATIVE-LAG ROBUSTNESS ARCHITECTURE"
)

lag_rows = []

ALL_LAGS = [
    PRIMARY_LAG,
    *ALTERNATIVE_LAGS,
]

for asset in ["BTC", "ETH"]:

    settings = ASSETS[asset]

    for lag in ALL_LAGS:

        if lag == PRIMARY_LAG:
            role = "Primary"
        else:
            role = "Robustness"

        row = {
            "Asset": settings["name"],
            "Dependent_Variable": settings["dependent"],
            "Lag_Days": lag,
            "Lag_Label": f"t-{lag}",
            "Role": role,
            "Own_Return_Horizon": f"t-{lag}",
            "Volume_Horizon": f"t-{lag}",
            "Traditional_Control_Horizon": (
                f"Information-aligned calendar-day horizon t-{lag}"
            ),
            "Reddit_Activity_Horizon": f"t-{lag}",
            "Reddit_Sentiment_Horizon": f"t-{lag}",
            "Cross_Crypto_Horizon": f"t-{lag}",
        }

        lag_rows.append(row)


alternative_lag_df = pd.DataFrame(
    lag_rows
)

print()
print(
    alternative_lag_df.to_string(
        index=False
    )
)


# =============================================================================
# 22. FORECAST MODEL ARCHITECTURE
# =============================================================================

print_section(
    "FORECAST MODEL ARCHITECTURE"
)

forecast_rows = []

for asset in ["BTC", "ETH"]:

    settings = ASSETS[asset]

    for model_code in FORECAST_MODELS:

        model = FORECAST_MODELS[model_code]

        predictors = get_forecast_predictors(
            asset,
            model_code,
        )

        row = {
            "Asset": settings["name"],
            "Forecast_Model": model_code,
            "Specification": model["name"],
            "Dependent_Variable": settings["dependent"],
            "Predictors": "; ".join(predictors),
            "Reddit_Activity": yes_no(
                model["activity"]
            ),
            "Reddit_Sentiment": yes_no(
                model["sentiment"]
            ),
            "Forecast_Horizon": FORECAST_HORIZON,
            "Estimation_Window": FORECAST_WINDOW,
            "OOS_Start": OOS_START,
            "OOS_End": OOS_END,
            "Evaluation": (
                "RMSE; MAE; OOS R2; directional accuracy; "
                "forecast-loss differences; DM-type test"
            ),
        }

        forecast_rows.append(row)


forecast_df = pd.DataFrame(
    forecast_rows
)

forecast_display_columns = [
    "Asset",
    "Forecast_Model",
    "Specification",
    "Reddit_Activity",
    "Reddit_Sentiment",
    "Forecast_Horizon",
]

print()
print(
    forecast_df[
        forecast_display_columns
    ].to_string(
        index=False
    )
)


# =============================================================================
# 23. FORECAST COMPARISON LOGIC
# =============================================================================

print_section(
    "FORECAST COMPARISON LOGIC"
)

forecast_comparison_rows = [
    {
        "Comparison": "F1 vs F2",
        "Added_Information": "Reddit activity",
        "Purpose": (
            "Tests whether Reddit activity improves OOS forecast "
            "accuracy beyond the baseline benchmark."
        ),
    },

    {
        "Comparison": "F1 vs F3",
        "Added_Information": "Reddit sentiment",
        "Purpose": (
            "Primary H3/H4 comparison: tests whether sentiment "
            "improves OOS forecasting beyond the benchmark."
        ),
    },

    {
        "Comparison": "F2 vs F4",
        "Added_Information": "Reddit sentiment",
        "Purpose": (
            "Tests whether sentiment improves OOS forecasts after "
            "Reddit activity is already controlled for."
        ),
    },

    {
        "Comparison": "F3 vs F4",
        "Added_Information": "Reddit activity",
        "Purpose": (
            "Tests whether activity improves OOS forecasts after "
            "sentiment is already included."
        ),
    },
]

forecast_comparison_df = pd.DataFrame(
    forecast_comparison_rows
)

print()
print(
    forecast_comparison_df.to_string(
        index=False
    )
)


# =============================================================================
# 24. STRUCTURAL-STABILITY ARCHITECTURE
# =============================================================================

print_section(
    "STRUCTURAL-STABILITY ARCHITECTURE"
)

stability_rows = [
    {
        "Analysis": "Baseline coefficient stability",
        "Model": "M1",
        "Timing": "Before Reddit arrives",
        "Purpose": (
            "Examine whether baseline return relationships differ "
            "across predetermined cryptocurrency-market regimes."
        ),
    },

    {
        "Analysis": "Sentiment coefficient stability",
        "Model": "M3",
        "Timing": "After Reddit arrives",
        "Purpose": (
            "Examine whether the sentiment-return relationship "
            "differs across predetermined market regimes."
        ),
    },

    {
        "Analysis": "Activity-adjusted sentiment stability",
        "Model": "M4",
        "Timing": "After Reddit arrives",
        "Purpose": (
            "Examine whether sentiment remains stable across regimes "
            "after controlling for Reddit activity."
        ),
    },

    {
        "Analysis": "Full robustness coefficient stability",
        "Model": "R2",
        "Timing": "After Reddit arrives",
        "Purpose": (
            "Examine sentiment stability conditional on Reddit "
            "activity and cross-crypto return dynamics."
        ),
    },
]

stability_df = pd.DataFrame(
    stability_rows
)

print()
print(
    stability_df.to_string(
        index=False
    )
)


# =============================================================================
# 25. ECONOMIC-SIGNIFICANCE ARCHITECTURE
# =============================================================================

print_section(
    "ECONOMIC-SIGNIFICANCE ARCHITECTURE"
)

economic_rows = []

for asset in ["BTC", "ETH"]:

    settings = ASSETS[asset]

    economic_rows.append(
        {
            "Asset": settings["name"],
            "Model": "M2",
            "Variable": settings["activity"],
            "Concept": "Reddit activity",
            "Economic_Effect": (
                "Coefficient x sample standard deviation "
                "of lagged Reddit activity"
            ),
        }
    )

    economic_rows.append(
        {
            "Asset": settings["name"],
            "Model": "M3",
            "Variable": settings["sentiment"],
            "Concept": "Reddit sentiment",
            "Economic_Effect": (
                "Coefficient x sample standard deviation "
                "of lagged Reddit sentiment"
            ),
        }
    )

    economic_rows.append(
        {
            "Asset": settings["name"],
            "Model": "M4",
            "Variable": settings["sentiment"],
            "Concept": (
                "Reddit sentiment conditional on activity"
            ),
            "Economic_Effect": (
                "Coefficient x sample standard deviation "
                "of lagged Reddit sentiment"
            ),
        }
    )


economic_df = pd.DataFrame(
    economic_rows
)

print()
print(
    economic_df.to_string(
        index=False
    )
)


# =============================================================================
# 26. VARIABLE DICTIONARY
# =============================================================================

print_section(
    "FINAL VARIABLE DICTIONARY"
)

variable_rows = [
    {
        "Variable": "BTC_Return",
        "Description": "Bitcoin daily log return",
        "Category": "Dependent variable",
        "Transformation": "Daily log return",
        "Timing": "t",
    },

    {
        "Variable": "ETH_Return",
        "Description": "Ethereum daily log return",
        "Category": "Dependent variable",
        "Transformation": "Daily log return",
        "Timing": "t",
    },

    {
        "Variable": "BTC_Lagged_Return",
        "Description": "Lagged Bitcoin daily log return",
        "Category": "Own-return control",
        "Transformation": "Daily log return",
        "Timing": "t-1",
    },

    {
        "Variable": "ETH_Lagged_Return",
        "Description": "Lagged Ethereum daily log return",
        "Category": "Own-return control",
        "Transformation": "Daily log return",
        "Timing": "t-1",
    },

    {
        "Variable": "Lagged_Log_BTC_Volume",
        "Description": "Lagged Bitcoin log trading volume",
        "Category": "Crypto-market control",
        "Transformation": "log(1 + Volume)",
        "Timing": "t-1 calendar day",
    },

    {
        "Variable": "Lagged_Log_ETH_Volume",
        "Description": "Lagged Ethereum log trading volume",
        "Category": "Crypto-market control",
        "Transformation": "log(1 + Volume)",
        "Timing": "t-1 calendar day",
    },

    {
        "Variable": "Lagged_SP500_Return_Aligned",
        "Description": "Information-aligned S&P 500 return",
        "Category": "Traditional-market control",
        "Transformation": "Daily return",
        "Timing": "Most recent information strictly before t",
    },

    {
        "Variable": "Lagged_VIX_Change_Aligned",
        "Description": "Information-aligned VIX change",
        "Category": "Traditional-market control",
        "Transformation": "Daily change",
        "Timing": "Most recent information strictly before t",
    },

    {
        "Variable": "Lagged_Gold_Return_Aligned",
        "Description": "Information-aligned gold futures return",
        "Category": "Traditional-market control",
        "Transformation": "Daily return",
        "Timing": "Most recent information strictly before t",
    },

    {
        "Variable": "Lagged_DXY_Return_Aligned",
        "Description": "Information-aligned DXY return",
        "Category": "Traditional-market control",
        "Transformation": "Daily return",
        "Timing": "Most recent information strictly before t",
    },

    {
        "Variable": "Lagged_US10Y_Change_Aligned",
        "Description": (
            "Information-aligned US 10-year Treasury yield change"
        ),
        "Category": "Traditional-market control",
        "Transformation": "Daily change",
        "Timing": "Most recent information strictly before t",
    },

    {
        "Variable": "Lagged_Log_BTC_Reddit_Post_Count",
        "Description": "Lagged Bitcoin Reddit activity",
        "Category": "Reddit activity / attention",
        "Transformation": (
            "log(1 + daily retained Reddit post count)"
        ),
        "Timing": "t-1",
    },

    {
        "Variable": "Lagged_Log_ETH_Reddit_Post_Count",
        "Description": "Lagged Ethereum Reddit activity",
        "Category": "Reddit activity / attention",
        "Transformation": (
            "log(1 + daily retained Reddit post count)"
        ),
        "Timing": "t-1",
    },

    {
        "Variable": "Lagged_BTC_Reddit_Sentiment",
        "Description": "Lagged Bitcoin Reddit sentiment",
        "Category": "Reddit sentiment",
        "Transformation": (
            "Daily aggregate sentiment score"
        ),
        "Timing": "t-1",
    },

    {
        "Variable": "Lagged_ETH_Reddit_Sentiment",
        "Description": "Lagged Ethereum Reddit sentiment",
        "Category": "Reddit sentiment",
        "Transformation": (
            "Daily aggregate sentiment score"
        ),
        "Timing": "t-1",
    },
]

variable_dictionary_df = pd.DataFrame(
    variable_rows
)

print()
print(
    variable_dictionary_df.to_string(
        index=False
    )
)


# =============================================================================
# 27. FORMAL VALIDATION
# =============================================================================

print_section(
    "FORMAL VALIDATION"
)

validation = {}


# M1
validation[
    "M1 is controls only"
] = (
    MODELS["M1"]["activity"] is False
    and MODELS["M1"]["sentiment"] is False
    and MODELS["M1"]["cross_crypto"] is False
)


# M2
validation[
    "M2 is controls + activity"
] = (
    MODELS["M2"]["activity"] is True
    and MODELS["M2"]["sentiment"] is False
    and MODELS["M2"]["cross_crypto"] is False
)


# M3
validation[
    "M3 is controls + sentiment"
] = (
    MODELS["M3"]["activity"] is False
    and MODELS["M3"]["sentiment"] is True
    and MODELS["M3"]["cross_crypto"] is False
)


# M4
validation[
    "M4 is controls + activity + sentiment"
] = (
    MODELS["M4"]["activity"] is True
    and MODELS["M4"]["sentiment"] is True
    and MODELS["M4"]["cross_crypto"] is False
)


# R1
validation[
    "R1 includes cross-crypto return"
] = (
    MODELS["R1"]["cross_crypto"] is True
)


# R2
validation[
    "R2 includes activity, sentiment and cross-crypto return"
] = (
    MODELS["R2"]["activity"] is True
    and MODELS["R2"]["sentiment"] is True
    and MODELS["R2"]["cross_crypto"] is True
)


# Alternative lags
validation[
    "Alternative lags are exactly 2, 3 and 7"
] = (
    ALTERNATIVE_LAGS == [2, 3, 7]
)


# Correct aligned controls
all_predictors = set(
    detailed_df["Predictor"].tolist()
)

old_controls_used = all_predictors.intersection(
    set(OLD_NONALIGNED_CONTROLS)
)

validation[
    "Old non-aligned controls are excluded"
] = (
    len(old_controls_used) == 0
)


# Verify every model contains aligned controls
for asset in ["BTC", "ETH"]:

    for model_code in MODELS:

        predictor_set = set(
            get_model_predictors(
                asset,
                model_code,
            )
        )

        aligned_set = set(
            TRADITIONAL_CONTROLS
        )

        check_name = (
            f"{asset} {model_code} contains all aligned controls"
        )

        validation[
            check_name
        ] = aligned_set.issubset(
            predictor_set
        )


# Activity and sentiment must remain different variables
validation[
    "BTC activity and sentiment are separate variables"
] = (
    ASSETS["BTC"]["activity"]
    != ASSETS["BTC"]["sentiment"]
)

validation[
    "ETH activity and sentiment are separate variables"
] = (
    ASSETS["ETH"]["activity"]
    != ASSETS["ETH"]["sentiment"]
)


# Forecast architecture
validation[
    "F1 is controls only"
] = (
    FORECAST_MODELS["F1"]["activity"] is False
    and FORECAST_MODELS["F1"]["sentiment"] is False
)

validation[
    "F2 adds activity only"
] = (
    FORECAST_MODELS["F2"]["activity"] is True
    and FORECAST_MODELS["F2"]["sentiment"] is False
)

validation[
    "F3 adds sentiment only"
] = (
    FORECAST_MODELS["F3"]["activity"] is False
    and FORECAST_MODELS["F3"]["sentiment"] is True
)

validation[
    "F4 contains activity and sentiment"
] = (
    FORECAST_MODELS["F4"]["activity"] is True
    and FORECAST_MODELS["F4"]["sentiment"] is True
)


# Print validation
for check_name, result in validation.items():

    if result:
        status = "PASS"
    else:
        status = "FAIL"

    print(
        f"\n{check_name}: {status}"
    )


overall_validation = all(
    validation.values()
)

print("\n" + "-" * 80)

print("\nOVERALL VALIDATION:")

if overall_validation:
    print("PASS")
else:
    print("FAIL")


if not overall_validation:

    failed_checks = []

    for check_name, result in validation.items():

        if not result:
            failed_checks.append(
                check_name
            )

    print("\nFAILED CHECKS:")

    for failed_check in failed_checks:
        print(
            f"  - {failed_check}"
        )

    raise ValueError(
        "Final model-specification validation failed."
    )


# =============================================================================
# 28. SAVE CSV OUTPUTS
# =============================================================================

print_section(
    "SAVING CSV OUTPUTS"
)

csv_outputs = {
    "01_detailed_model_specification.csv":
        detailed_df,

    "02_compact_final_model_specification.csv":
        compact_df,

    "03_model_inclusion_matrix.csv":
        inclusion_df,

    "04_reddit_activity_vs_sentiment.csv":
        reddit_concepts_df,

    "05_reddit_variable_availability.csv":
        reddit_status_df,

    "06_model_comparison_logic.csv":
        comparison_df,

    "07_hypothesis_model_mapping.csv":
        hypothesis_df,

    "08_cross_crypto_robustness_architecture.csv":
        cross_crypto_df,

    "09_alternative_lag_robustness_architecture.csv":
        alternative_lag_df,

    "10_forecast_model_architecture.csv":
        forecast_df,

    "11_forecast_comparison_logic.csv":
        forecast_comparison_df,

    "12_structural_stability_architecture.csv":
        stability_df,

    "13_economic_significance_architecture.csv":
        economic_df,

    "14_variable_dictionary.csv":
        variable_dictionary_df,
}


for filename, dataframe in csv_outputs.items():

    output_path = (
        OUTPUT_DIR
        / filename
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print("\nSaved:")
    print(output_path)


# =============================================================================
# 29. SAVE DISSERTATION-READY MODEL TABLE
# =============================================================================

print_section(
    "SAVING DISSERTATION-READY MODEL TABLE"
)

dissertation_table_path = (
    OUTPUT_DIR
    / "dissertation_ready_model_specification.txt"
)

dissertation_table = """
FINAL EMPIRICAL MODEL ARCHITECTURE

Model | Controls | Reddit Activity | Reddit Sentiment | Cross-Crypto | Role
-----------------------------------------------------------------------------
M1    | Yes      | No              | No               | No           | Primary
M2    | Yes      | Yes             | No               | No           | Primary
M3    | Yes      | No              | Yes              | No           | Primary
M4    | Yes      | Yes             | Yes              | No           | Primary
R1    | Yes      | No              | No               | Yes          | Robustness
R2    | Yes      | Yes             | Yes              | Yes          | Robustness


M1: BASELINE CONTROLS ONLY

Bitcoin:
- BTC_Lagged_Return
- Lagged_Log_BTC_Volume
- Lagged_SP500_Return_Aligned
- Lagged_VIX_Change_Aligned
- Lagged_Gold_Return_Aligned
- Lagged_DXY_Return_Aligned
- Lagged_US10Y_Change_Aligned

Ethereum:
- ETH_Lagged_Return
- Lagged_Log_ETH_Volume
- Lagged_SP500_Return_Aligned
- Lagged_VIX_Change_Aligned
- Lagged_Gold_Return_Aligned
- Lagged_DXY_Return_Aligned
- Lagged_US10Y_Change_Aligned


M2: CONTROLS + REDDIT ACTIVITY

Bitcoin:
Add Lagged_Log_BTC_Reddit_Post_Count

Ethereum:
Add Lagged_Log_ETH_Reddit_Post_Count

Activity is defined as:

log(1 + daily retained Reddit post count)

This represents attention or discussion intensity.


M3: CONTROLS + REDDIT SENTIMENT

Bitcoin:
Add Lagged_BTC_Reddit_Sentiment

Ethereum:
Add Lagged_ETH_Reddit_Sentiment

This represents the tone or polarity of the retained Reddit discussion.


M4: CONTROLS + REDDIT ACTIVITY + SENTIMENT

M4 includes both activity and sentiment.

This is important because Reddit activity and Reddit sentiment are
conceptually distinct.

Activity measures how much discussion occurred.

Sentiment measures the tone of that discussion.

The M4 sentiment coefficient therefore evaluates sentiment while
controlling separately for Reddit attention.


R1: CROSS-CRYPTO BASELINE ROBUSTNESS

Bitcoin:
M1 + ETH_Lagged_Return

Ethereum:
M1 + BTC_Lagged_Return


R2: FULL CROSS-CRYPTO ROBUSTNESS

Bitcoin:
M4 + ETH_Lagged_Return

Ethereum:
M4 + BTC_Lagged_Return


ALTERNATIVE-LAG ROBUSTNESS

Primary specification:
t-1

Robustness specifications:
t-2
t-3
t-7


FORECAST ARCHITECTURE

F1:
Controls only

F2:
Controls + Reddit activity

F3:
Controls + Reddit sentiment

F4:
Controls + Reddit activity + sentiment


PRIMARY H3/H4 COMPARISON

F1 versus F3

This asks whether sentiment improves one-day-ahead OOS return
forecasting relative to an otherwise identical benchmark without
sentiment.


ACTIVITY-ADJUSTED SENTIMENT COMPARISON

F2 versus F4

This asks whether sentiment adds forecast value after Reddit activity
has already been included.


FORECAST EVALUATION

All competing forecasts must be evaluated on exactly the same OOS
dates.

Evaluation metrics:

- RMSE
- MAE
- OOS R-squared
- Directional accuracy
- Forecast-error/loss differences
- DM-type formal forecast-accuracy comparison


IMPORTANT INTERPRETATION

Reddit activity is not Reddit sentiment.

Activity:
quantity/intensity of retained Reddit discussion.

Sentiment:
tone/polarity of retained Reddit discussion.

The dissertation therefore estimates the two concepts separately and
jointly.
""".strip()


with open(
    dissertation_table_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        dissertation_table
    )


print("\nSaved:")
print(dissertation_table_path)


# =============================================================================
# 30. SAVE METHODOLOGY NOTE
# =============================================================================

print_section(
    "SAVING METHODOLOGY NOTE"
)

methodology_path = (
    OUTPUT_DIR
    / "final_model_specification_methodology_note.txt"
)

methodology_note = f"""
FINAL MODEL-SPECIFICATION METHODOLOGY

STUDY PERIOD
------------

{STUDY_START} to {STUDY_END}

Frequency:
Daily

Assets:
Bitcoin and Ethereum

Primary lag:
t-1

Alternative robustness lags:
t-2, t-3 and t-7


1. BASELINE MODEL
-----------------

M1 contains:

- own lagged cryptocurrency return;
- lagged log cryptocurrency trading volume;
- information-aligned S&P 500 return;
- information-aligned VIX change;
- information-aligned gold futures return;
- information-aligned DXY return; and
- information-aligned US 10-year Treasury yield change.

No Reddit information is included in M1.


2. REDDIT ACTIVITY MODEL
------------------------

M2 adds lagged Reddit activity.

Reddit activity is defined as:

log(1 + daily retained Reddit post count)

The activity variable represents discussion intensity or attention.

It does not represent whether discussion is positive or negative.


3. REDDIT SENTIMENT MODEL
-------------------------

M3 adds lagged Reddit sentiment to the baseline specification.

The purpose is to test whether the tone of retained Reddit discussion
adds information beyond traditional and cryptocurrency-market
controls.


4. FULL REDDIT MODEL
--------------------

M4 contains both Reddit activity and Reddit sentiment.

This specification explicitly distinguishes attention from sentiment.

The sentiment coefficient in M4 therefore represents the estimated
sentiment-return relationship conditional on the quantity of Reddit
discussion.


5. CROSS-CRYPTO ROBUSTNESS
--------------------------

R1 adds the other cryptocurrency's lagged return to M1.

Bitcoin:
ETH_Lagged_Return

Ethereum:
BTC_Lagged_Return

R2 adds the corresponding cross-crypto lagged return to M4.

This addresses possible omitted cross-cryptocurrency return dynamics.


6. ALTERNATIVE LAGS
-------------------

The primary specification uses t-1 information.

Alternative robustness horizons are:

t-2
t-3
t-7

These alternative horizons assess whether conclusions are dependent
on the one-day lag assumption.

After Reddit variables become available, the same conceptual lag
architecture should be applied to Reddit activity and sentiment.


7. INFORMATION ALIGNMENT
------------------------

The final traditional-market controls are:

Lagged_SP500_Return_Aligned
Lagged_VIX_Change_Aligned
Lagged_Gold_Return_Aligned
Lagged_DXY_Return_Aligned
Lagged_US10Y_Change_Aligned

The old non-aligned traditional-market variables are excluded.

This is necessary because cryptocurrency trades seven days per week
whereas the traditional financial markets used as controls do not.

The aligned variables ensure that only information available before
the cryptocurrency return date enters the model.


8. EXPLANATORY INFERENCE
------------------------

The explanatory models distinguish coefficient estimation from
forecasting performance.

OLS coefficients are accompanied by HAC/Newey-West inference.

Maximum HAC lag:
{HAC_MAXLAGS}


9. ECONOMIC SIGNIFICANCE
------------------------

Statistical significance is not treated as equivalent to economic
importance.

For continuous predictors, the economic effect of a one-standard-
deviation increase is calculated as:

estimated coefficient x predictor standard deviation

The same framework will be applied to Reddit sentiment and Reddit
activity after the Reddit data become available.


10. FORECASTING
---------------

Forecasting uses a genuine chronological out-of-sample design.

Forecast horizon:
{FORECAST_HORIZON}

Estimation window:
{FORECAST_WINDOW}

Planned OOS period:
{OOS_START} to {OOS_END}

Forecast models are:

F1 = controls only

F2 = controls + Reddit activity

F3 = controls + Reddit sentiment

F4 = controls + Reddit activity + sentiment


11. H3 AND H4
-------------

The primary sentiment forecast comparison is:

F1 versus F3

This directly tests whether sentiment improves forecast accuracy
relative to an otherwise identical benchmark without sentiment.

An additional comparison is:

F2 versus F4

This determines whether sentiment adds predictive information beyond
Reddit activity itself.

Competing forecasts must be evaluated on exactly the same OOS dates.


12. FORECAST EVALUATION
-----------------------

Forecast evaluation includes:

- RMSE;
- MAE;
- OOS R-squared;
- directional accuracy;
- forecast-error/loss differences; and
- a DM-type formal forecast-accuracy comparison.

H3 and H4 should therefore not be accepted or rejected solely because
one model has a marginally smaller RMSE.


13. STRUCTURAL STABILITY
------------------------

The same model architecture should be retained across predetermined
market regimes.

Before Reddit arrives, M1 can be used to assess baseline coefficient
stability.

After Reddit arrives, M3 and M4 can be estimated across the same
regimes to determine whether the sentiment-return relationship is
stable across different cryptocurrency-market conditions.


14. REDDIT SAMPLE INTERPRETATION
--------------------------------

The eventual Reddit variables represent activity and sentiment in the
retained asset-specific Reddit research sample.

They should not automatically be described as measures of general
investor sentiment.


15. FINAL ARCHITECTURE
----------------------

M1:
Controls only

M2:
Controls + Reddit activity

M3:
Controls + Reddit sentiment

M4:
Controls + Reddit activity + sentiment

R1:
Controls + cross-crypto return

R2:
Controls + Reddit activity + Reddit sentiment + cross-crypto return

Alternative lags:
t-2, t-3, t-7

Forecast models:
F1, F2, F3, F4

This architecture explicitly distinguishes Reddit attention/activity
from Reddit sentiment.
""".strip()


with open(
    methodology_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        methodology_note
    )


print("\nSaved:")
print(methodology_path)


# =============================================================================
# 31. SAVE REDDIT-INTEGRATION INSTRUCTIONS
# =============================================================================

print_section(
    "SAVING REDDIT-INTEGRATION INSTRUCTIONS"
)

reddit_instructions_path = (
    OUTPUT_DIR
    / "when_reddit_arrives.txt"
)

reddit_instructions = """
WHEN REDDIT DATA ARRIVES

Do not redesign the model architecture in response to the results.

Construct and merge the following primary variables:

BITCOIN

Lagged_Log_BTC_Reddit_Post_Count
Lagged_BTC_Reddit_Sentiment


ETHEREUM

Lagged_Log_ETH_Reddit_Post_Count
Lagged_ETH_Reddit_Sentiment


STEP 1

Estimate M1:

Controls only.


STEP 2

Estimate M2:

Controls + Reddit activity.


STEP 3

Estimate M3:

Controls + Reddit sentiment.


STEP 4

Estimate M4:

Controls + Reddit activity + Reddit sentiment.


STEP 5

Estimate cross-crypto robustness.

Bitcoin:
Add ETH_Lagged_Return.

Ethereum:
Add BTC_Lagged_Return.


STEP 6

Estimate alternative-lag robustness:

t-2
t-3
t-7


STEP 7

Calculate economic significance.

For sentiment:

sentiment coefficient
x
standard deviation of lagged sentiment


STEP 8

Repeat the predetermined market-regime / structural-stability
analysis for M3 and M4.


STEP 9

Produce genuine OOS forecasts.

F1:
Controls only.

F2:
Controls + activity.

F3:
Controls + sentiment.

F4:
Controls + activity + sentiment.


STEP 10

For H3/H4, compare F1 versus F3 on exactly the same OOS dates.

Also compare F2 versus F4 to determine whether sentiment adds
predictive information beyond Reddit activity.


REPORT

RMSE
MAE
OOS R-squared
Directional accuracy
Forecast-error/loss differences
DM-type formal forecast comparison


IMPORTANT

Reddit activity measures attention / discussion intensity.

Reddit sentiment measures tone / polarity.

The two variables must remain conceptually and empirically separate.
""".strip()


with open(
    reddit_instructions_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        reddit_instructions
    )


print("\nSaved:")
print(reddit_instructions_path)


# =============================================================================
# 32. FINAL SUMMARY
# =============================================================================

print_section(
    "FINAL MODEL ARCHITECTURE SUMMARY"
)

print(
    """
PRIMARY EXPLANATORY MODELS

M1 = Controls only

M2 = Controls + Reddit activity

M3 = Controls + Reddit sentiment

M4 = Controls + Reddit activity + sentiment


ROBUSTNESS

R1 = Controls + cross-crypto lagged return

R2 = Controls + Reddit activity + sentiment
     + cross-crypto lagged return


ALTERNATIVE LAGS

Primary:
t-1

Robustness:
t-2
t-3
t-7


FORECAST MODELS

F1 = Controls only

F2 = Controls + Reddit activity

F3 = Controls + Reddit sentiment

F4 = Controls + Reddit activity + sentiment
"""
)

print("\nPrimary H1/H2 sentiment comparison:")
print("M1 vs M3")

print("\nActivity-adjusted sentiment comparison:")
print("M2 vs M4")

print("\nPrimary H3/H4 forecast comparison:")
print("F1 vs F3")

print("\nActivity-adjusted forecast comparison:")
print("F2 vs F4")

print("\nReddit data required for this script:")
print("NO")

print("\nReddit variables currently missing:")
print("EXPECTED until Reddit data arrive")

print("\nOld non-aligned traditional controls used:")
print("NO")

print("\nCorrect information-aligned controls used:")
print("YES")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nOverall validation:")
print("PASS")


print_section(
    "FINAL MODEL-SPECIFICATION TABLE COMPLETE"
)