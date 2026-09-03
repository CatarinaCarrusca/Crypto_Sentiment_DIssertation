# =============================================================================
# MSc Financial Technology Dissertation
# Stage 04 — Post-Level Reddit Sentiment Analysis
#
# Research Question:
# "Does Social Media Sentiment Improve the Prediction of Cryptocurrency
# Returns Beyond Traditional Market Indicators?
# Evidence from Bitcoin and Ethereum."
#
# PRIMARY NLP MODEL
# -----------------
# Hugging Face model:
# cardiffnlp/twitter-roberta-base-sentiment-latest
#
# Unit of analysis:
# Individual retained Reddit post
#
# Input:
# analysis_text
#
# Primary continuous sentiment measure:
#
# Sentiment_i = P(Positive)_i - P(Negative)_i
#
# Range:
# approximately -1 to +1
#
# Negative values = relatively negative sentiment
# Positive values = relatively positive sentiment
# Values around zero = neutral / mixed balance
#
# IMPORTANT STAGE BOUNDARIES
# --------------------------
# This script:
#   - DOES perform post-level sentiment classification.
#   - DOES preserve Negative / Neutral / Positive probabilities.
#   - DOES construct the continuous sentiment score.
#   - DOES perform extensive QC.
#   - DOES audit model truncation.
#   - DOES create a blinded manual validation sample.
#
# This script:
#   - DOES NOT aggregate to daily sentiment.
#   - DOES NOT create lagged variables.
#   - DOES NOT merge market variables.
#   - DOES NOT run return regressions.
#   - DOES NOT modify Stage 02 or Stage 03 outputs.
#
# Stage 05 will perform daily aggregation.
# Stage 06 will construct forecasting-safe lags.
# Stage 07 will merge Reddit variables with traditional market variables.
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

from pathlib import Path
from datetime import datetime, timezone
import json
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import transformers

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
)


# =============================================================================
# 2. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)


# -----------------------------------------------------------------------------
# Frozen Stage 02 primary sentiment sample
# -----------------------------------------------------------------------------

INPUT_FILE = (
    PROJECT_ROOT
    / "data_clean"
    / "reddit"
    / "reddit_posts_primary_sentiment_sample.csv"
)


# -----------------------------------------------------------------------------
# Stage 04 receives its own isolated output directory
# -----------------------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data_clean"
    / "reddit"
    / "stage04_sentiment"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 3. MODEL CONFIGURATION
# =============================================================================

MODEL_NAME = (
    "cardiffnlp/"
    "twitter-roberta-base-sentiment-latest"
)

MAX_LENGTH = 512


# =============================================================================
# 4. EXPECTED FROZEN SAMPLE PROPERTIES
# =============================================================================

EXPECTED_ROWS = 136_019

EXPECTED_ASSETS = {
    "BTC",
    "ETH",
}

EXPECTED_SUBREDDITS = {
    "Bitcoin",
    "BitcoinMarkets",
    "ethereum",
}

EXPECTED_START_YEAR = 2021
EXPECTED_END_YEAR = 2025


# =============================================================================
# 5. REPRODUCIBILITY
# =============================================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =============================================================================
# 6. COMPUTATION SETTINGS
# =============================================================================

# Conservative default for a Mac.
#
# If you receive an out-of-memory error, reduce this to 8.
# If your machine handles it easily, it could later be increased.
#
# Changing batch size does NOT change the methodology or predictions.

BATCH_SIZE = 16


# Save progress periodically so that a long scoring run can be resumed.

CHECKPOINT_EVERY = 1000


# Token-length audit is run in larger batches because it does not run
# the neural network itself.

TOKEN_AUDIT_BATCH_SIZE = 250


# =============================================================================
# 7. MANUAL VALIDATION CONFIGURATION
# =============================================================================

# 30 posts per Asset × Year stratum
#
# 2 assets × 5 years × 30 = 300 manually reviewed posts.

MANUAL_SAMPLE_PER_ASSET_YEAR = 30

VALID_SENTIMENT_LABELS = {
    "negative",
    "neutral",
    "positive",
}


# =============================================================================
# 8. OUTPUT FILE PATHS
# =============================================================================

FINAL_SENTIMENT_FILE = (
    OUTPUT_DIR
    / "reddit_post_level_sentiment.csv"
)


CHECKPOINT_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_checkpoint.csv"
)


OVERALL_SUMMARY_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_overall_summary.csv"
)


LABEL_DISTRIBUTION_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_label_distribution.csv"
)


BY_ASSET_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_by_asset.csv"
)


BY_YEAR_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_by_year.csv"
)


BY_YEAR_ASSET_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_by_year_asset.csv"
)


TRUNCATION_QC_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_truncation_qc.csv"
)


LOW_CONFIDENCE_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_low_confidence_examples.csv"
)


MODEL_METADATA_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_model_metadata.json"
)


FINAL_QC_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_final_qc.csv"
)


MANUAL_TEMPLATE_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_manual_validation_blinded.csv"
)


MANUAL_PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_manual_validation_predictions.csv"
)


MANUAL_RESULTS_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_manual_validation_results.csv"
)


MANUAL_CLASS_METRICS_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_manual_validation_class_metrics.csv"
)


MANUAL_CONFUSION_FILE = (
    OUTPUT_DIR
    / "reddit_sentiment_manual_validation_confusion_matrix.csv"
)


# =============================================================================
# 9. HELPER FUNCTIONS
# =============================================================================

def print_section(title):
    """Print clearly separated console sections."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def detect_device():
    """
    Select the best available computing device.

    Priority:
        1. NVIDIA CUDA
        2. Apple Silicon MPS
        3. CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def normalise_model_label(label):
    """
    Standardise model class labels.

    Expected final labels:
        negative
        neutral
        positive
    """

    text = str(label).strip().lower()

    if "negative" in text:
        return "negative"

    if "neutral" in text:
        return "neutral"

    if "positive" in text:
        return "positive"

    return text


def model_preprocess(text):
    """
    Prepare text for CardiffNLP social-media inference.

    IMPORTANT:
    The stored Stage 02 analysis_text is NEVER altered.

    This transformation exists only temporarily when text is passed
    to the model.

    User mentions are mapped to @user.
    URLs are mapped to http if any survived Stage 02.

    Capitalisation, punctuation, negation and emojis are preserved.
    """

    text = str(text)

    words = text.split()

    processed_words = []

    for word in words:

        if (
            word.startswith("@")
            and len(word) > 1
        ):
            processed_words.append("@user")

        elif word.lower().startswith("http"):
            processed_words.append("http")

        else:
            processed_words.append(word)

    return " ".join(processed_words)


def validate_numeric_finite(series, column_name):
    """
    Convert a series to numeric and verify all values are finite.
    """

    numeric = pd.to_numeric(
        series,
        errors="raise",
    )

    if not np.isfinite(numeric).all():

        raise AssertionError(
            f"{column_name} contains NaN or infinite values."
        )

    return numeric


def save_and_reload_csv(df_to_save, path):
    """
    Save CSV and immediately reload it as an integrity check.
    """

    df_to_save.to_csv(
        path,
        index=False,
    )

    reloaded = pd.read_csv(path)

    if len(reloaded) != len(df_to_save):

        raise AssertionError(
            f"Reload row-count mismatch for {path.name}"
        )

    return reloaded


# =============================================================================
# 10. START STAGE 04
# =============================================================================

print_section(
    "STAGE 04 — POST-LEVEL REDDIT SENTIMENT ANALYSIS"
)

print(f"Input:\n{INPUT_FILE}")

print(f"\nOutput directory:\n{OUTPUT_DIR}")

print(f"\nPrimary model:\n{MODEL_NAME}")


# =============================================================================
# 11. VERIFY INPUT FILE EXISTS
# =============================================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        "\nFrozen Stage 02 primary sentiment sample "
        "was not found:\n"
        f"{INPUT_FILE}"
    )


# =============================================================================
# 12. LOAD FROZEN STAGE 02 PRIMARY SAMPLE
# =============================================================================

print_section(
    "LOADING FROZEN STAGE 02 PRIMARY SAMPLE"
)

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")

print(f"Columns loaded: {len(df.columns):,}")


# =============================================================================
# 13. VALIDATE REQUIRED INPUT COLUMNS
# =============================================================================

required_columns = {
    "post_id",
    "post_date",
    "asset",
    "subreddit",
    "analysis_text",
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:

    raise AssertionError(
        "Missing required Stage 04 columns: "
        f"{sorted(missing_columns)}"
    )


# =============================================================================
# 14. VALIDATE EXACT FROZEN ROW COUNT
# =============================================================================

if len(df) != EXPECTED_ROWS:

    raise AssertionError(
        "\nStage 02 primary sample row-count mismatch."
        f"\nExpected: {EXPECTED_ROWS:,}"
        f"\nFound:    {len(df):,}"
    )


# =============================================================================
# 15. VALIDATE POST IDs
# =============================================================================

if df["post_id"].isna().any():

    raise AssertionError(
        "Missing post_id values detected."
    )


df["post_id"] = (
    df["post_id"]
    .astype(str)
    .str.strip()
)


if df["post_id"].eq("").any():

    raise AssertionError(
        "Blank post_id values detected."
    )


duplicate_ids = int(
    df["post_id"]
    .duplicated()
    .sum()
)


if duplicate_ids != 0:

    raise AssertionError(
        f"{duplicate_ids:,} duplicate post IDs detected."
    )


# =============================================================================
# 16. VALIDATE DATES
# =============================================================================

df["post_date"] = pd.to_datetime(
    df["post_date"],
    errors="raise",
)


df["year"] = (
    df["post_date"]
    .dt.year
)


minimum_year = int(
    df["year"].min()
)

maximum_year = int(
    df["year"].max()
)


if minimum_year != EXPECTED_START_YEAR:

    raise AssertionError(
        f"Unexpected minimum year: {minimum_year}"
    )


if maximum_year != EXPECTED_END_YEAR:

    raise AssertionError(
        f"Unexpected maximum year: {maximum_year}"
    )


# =============================================================================
# 17. VALIDATE ASSETS
# =============================================================================

actual_assets = set(
    df["asset"]
    .dropna()
    .astype(str)
    .unique()
)


if actual_assets != EXPECTED_ASSETS:

    raise AssertionError(
        "Unexpected asset set."
        f"\nExpected: {EXPECTED_ASSETS}"
        f"\nFound:    {actual_assets}"
    )


# =============================================================================
# 18. VALIDATE SUBREDDITS
# =============================================================================

actual_subreddits = set(
    df["subreddit"]
    .dropna()
    .astype(str)
    .unique()
)


if actual_subreddits != EXPECTED_SUBREDDITS:

    raise AssertionError(
        "Unexpected subreddit set."
        f"\nExpected: {EXPECTED_SUBREDDITS}"
        f"\nFound:    {actual_subreddits}"
    )


# =============================================================================
# 19. VALIDATE ANALYSIS TEXT
# =============================================================================

if df["analysis_text"].isna().any():

    raise AssertionError(
        "Missing analysis_text values detected."
    )


blank_analysis_text = (
    df["analysis_text"]
    .astype(str)
    .str.strip()
    .eq("")
)


if blank_analysis_text.any():

    raise AssertionError(
        f"{int(blank_analysis_text.sum()):,} "
        "blank analysis_text observations detected."
    )


# =============================================================================
# 20. PRESERVE EXACT SOURCE ORDER
# =============================================================================

df["_source_order"] = np.arange(
    len(df)
)


print("\nStage 02 input validation: PASS")


print("\nAsset counts:")

print(
    df["asset"]
    .value_counts()
    .sort_index()
)


print("\nYear × asset counts:")

print(
    df
    .groupby(
        ["year", "asset"]
    )
    .size()
)


# =============================================================================
# 21. DETECT COMPUTING DEVICE
# =============================================================================

print_section(
    "COMPUTING DEVICE"
)


device = detect_device()

print(f"Selected device: {device}")


if str(device) == "mps":

    print(
        "\nApple Metal Performance Shaders (MPS) detected."
    )

elif str(device) == "cpu":

    print(
        "\nNo GPU/MPS device detected."
        "\nInference will run on CPU and may take considerably longer."
    )


# =============================================================================
# 22. LOAD CARDIFFNLP TOKENIZER AND MODEL
# =============================================================================

print_section(
    "LOADING CARDIFFNLP TWITTER-ROBERTA MODEL"
)


print(
    "Downloading/loading tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


print(
    "Downloading/loading model..."
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)


model.to(device)

model.eval()


print("\nModel loaded successfully.")


# =============================================================================
# 23. IDENTIFY MODEL LABEL MAPPING
# =============================================================================

print_section(
    "VALIDATING MODEL LABEL MAPPING"
)


raw_id2label = model.config.id2label

print(
    "Raw model id2label:"
)

print(raw_id2label)


id2label = {}

for raw_id, raw_label in raw_id2label.items():

    numeric_id = int(raw_id)

    label = normalise_model_label(
        raw_label
    )

    id2label[numeric_id] = label


# -----------------------------------------------------------------------------
# Cardiff model should expose semantic labels.
#
# However, if the Transformers configuration returns generic LABEL_0 etc.,
# use the documented CardiffNLP three-class order.
# -----------------------------------------------------------------------------

if set(id2label.values()) != VALID_SENTIMENT_LABELS:

    print(
        "\nSemantic class names were not fully exposed "
        "by the loaded configuration."
    )

    print(
        "Applying CardiffNLP documented class mapping:"
    )

    id2label = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }


if set(id2label.values()) != VALID_SENTIMENT_LABELS:

    raise AssertionError(
        "Unable to establish valid Negative / Neutral / Positive mapping."
    )


label2id = {
    label: numeric_id
    for numeric_id, label in id2label.items()
}


NEGATIVE_ID = label2id[
    "negative"
]

NEUTRAL_ID = label2id[
    "neutral"
]

POSITIVE_ID = label2id[
    "positive"
]


print("\nValidated model mapping:")

print(id2label)


# =============================================================================
# 24. TOKEN-LENGTH / TRUNCATION AUDIT
# =============================================================================

print_section(
    "TOKEN-LENGTH AND TRUNCATION AUDIT"
)


print(
    "Calculating token lengths before truncation..."
)


texts = (
    df["analysis_text"]
    .astype(str)
    .tolist()
)


token_lengths = np.empty(
    len(df),
    dtype=np.int32,
)


for start in range(
    0,
    len(df),
    TOKEN_AUDIT_BATCH_SIZE,
):

    end = min(
        start + TOKEN_AUDIT_BATCH_SIZE,
        len(df),
    )


    batch_texts = [
        model_preprocess(text)
        for text in texts[start:end]
    ]


    tokenized = tokenizer(
        batch_texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )


    batch_lengths = [
        len(token_ids)
        for token_ids
        in tokenized["input_ids"]
    ]


    token_lengths[
        start:end
    ] = batch_lengths


    if (
        end % 10_000 == 0
        or end == len(df)
    ):

        print(
            f"Token audit progress: "
            f"{end:,}/{len(df):,}"
        )


df["token_length"] = (
    token_lengths
)


df["was_truncated"] = (
    df["token_length"]
    > MAX_LENGTH
)


number_truncated = int(
    df["was_truncated"]
    .sum()
)


percentage_truncated = (
    number_truncated
    / len(df)
    * 100
)


print(
    f"\nPosts longer than {MAX_LENGTH} tokens: "
    f"{number_truncated:,}"
)


print(
    "Percentage requiring model truncation: "
    f"{percentage_truncated:.4f}%"
)


print(
    "\nMaximum observed token length: "
    f"{int(df['token_length'].max()):,}"
)


# =============================================================================
# 25. LOAD EXISTING CHECKPOINT IF PRESENT
# =============================================================================

print_section(
    "CHECKPOINT / RESUME STATUS"
)


prediction_columns = [
    "post_id",
    "prob_negative",
    "prob_neutral",
    "prob_positive",
    "sentiment_score",
    "sentiment_label",
    "max_probability",
]


if CHECKPOINT_FILE.exists():

    checkpoint = pd.read_csv(
        CHECKPOINT_FILE
    )


    checkpoint["post_id"] = (
        checkpoint["post_id"]
        .astype(str)
        .str.strip()
    )


    print(
        "Existing checkpoint found."
    )

    print(
        f"Checkpoint predictions: "
        f"{len(checkpoint):,}"
    )


    missing_checkpoint_columns = (
        set(prediction_columns)
        - set(checkpoint.columns)
    )


    if missing_checkpoint_columns:

        raise AssertionError(
            "Checkpoint is missing columns: "
            f"{sorted(missing_checkpoint_columns)}"
        )


    if checkpoint[
        "post_id"
    ].duplicated().any():

        raise AssertionError(
            "Checkpoint contains duplicate post IDs."
        )


    invalid_checkpoint_ids = (
        set(checkpoint["post_id"])
        - set(df["post_id"])
    )


    if invalid_checkpoint_ids:

        raise AssertionError(
            "Checkpoint contains post IDs that are not "
            "present in the frozen Stage 02 sample."
        )


else:

    checkpoint = pd.DataFrame(
        columns=prediction_columns
    )

    print(
        "No existing checkpoint found."
    )


completed_ids = set(
    checkpoint[
        "post_id"
    ].astype(str)
)


remaining = df[
    ~df["post_id"].isin(
        completed_ids
    )
].copy()


print(
    f"\nAlready scored: "
    f"{len(completed_ids):,}"
)


print(
    f"Remaining to score: "
    f"{len(remaining):,}"
)


# =============================================================================
# 26. RUN POST-LEVEL SENTIMENT INFERENCE
# =============================================================================

print_section(
    "POST-LEVEL SENTIMENT INFERENCE"
)


new_predictions = []

processed_since_save = 0

run_start_time = time.time()


with torch.inference_mode():

    for batch_start in range(
        0,
        len(remaining),
        BATCH_SIZE,
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            len(remaining),
        )


        batch_df = remaining.iloc[
            batch_start:batch_end
        ]


        batch_texts = [
            model_preprocess(text)
            for text
            in batch_df[
                "analysis_text"
            ].astype(str)
        ]


        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )


        encoded = {
            key: value.to(device)
            for key, value
            in encoded.items()
        }


        outputs = model(
            **encoded
        )


        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )


        probabilities = (
            probabilities
            .detach()
            .cpu()
            .numpy()
        )


        # ---------------------------------------------------------------------
        # Convert each row of probabilities to dissertation variables
        # ---------------------------------------------------------------------

        for local_index, (_, row) in enumerate(
            batch_df.iterrows()
        ):

            probs = probabilities[
                local_index
            ]


            prob_negative = float(
                probs[
                    NEGATIVE_ID
                ]
            )


            prob_neutral = float(
                probs[
                    NEUTRAL_ID
                ]
            )


            prob_positive = float(
                probs[
                    POSITIVE_ID
                ]
            )


            # -----------------------------------------------------------------
            # PRIMARY CONTINUOUS SENTIMENT MEASURE
            #
            # Sentiment_i =
            # P(Positive)_i - P(Negative)_i
            # -----------------------------------------------------------------

            sentiment_score = (
                prob_positive
                - prob_negative
            )


            predicted_class_id = int(
                np.argmax(
                    probs
                )
            )


            sentiment_label = (
                id2label[
                    predicted_class_id
                ]
            )


            max_probability = float(
                np.max(
                    probs
                )
            )


            new_predictions.append(
                {
                    "post_id": str(
                        row["post_id"]
                    ),
                    "prob_negative": (
                        prob_negative
                    ),
                    "prob_neutral": (
                        prob_neutral
                    ),
                    "prob_positive": (
                        prob_positive
                    ),
                    "sentiment_score": (
                        sentiment_score
                    ),
                    "sentiment_label": (
                        sentiment_label
                    ),
                    "max_probability": (
                        max_probability
                    ),
                }
            )


        processed_since_save += len(
            batch_df
        )


        # ---------------------------------------------------------------------
        # Periodically save checkpoint
        # ---------------------------------------------------------------------

        if (
            processed_since_save
            >= CHECKPOINT_EVERY
        ):

            latest_predictions = (
                pd.DataFrame(
                    new_predictions
                )
            )


            checkpoint = pd.concat(
                [
                    checkpoint,
                    latest_predictions,
                ],
                ignore_index=True,
            )


            checkpoint = (
                checkpoint
                .drop_duplicates(
                    subset="post_id",
                    keep="last",
                )
            )


            checkpoint.to_csv(
                CHECKPOINT_FILE,
                index=False,
            )


            new_predictions = []

            processed_since_save = 0


            elapsed_minutes = (
                time.time()
                - run_start_time
            ) / 60


            total_complete = len(
                checkpoint
            )


            print(
                f"Scored: "
                f"{total_complete:,}/"
                f"{EXPECTED_ROWS:,}"
                f" | elapsed this run: "
                f"{elapsed_minutes:.1f} minutes"
            )


# =============================================================================
# 27. SAVE ANY REMAINING UNSAVED PREDICTIONS
# =============================================================================

if new_predictions:

    latest_predictions = pd.DataFrame(
        new_predictions
    )


    checkpoint = pd.concat(
        [
            checkpoint,
            latest_predictions,
        ],
        ignore_index=True,
    )


    checkpoint = (
        checkpoint
        .drop_duplicates(
            subset="post_id",
            keep="last",
        )
    )


    checkpoint.to_csv(
        CHECKPOINT_FILE,
        index=False,
    )


print(
    f"\nTotal checkpoint predictions: "
    f"{len(checkpoint):,}"
)


# =============================================================================
# 28. VALIDATE COMPLETE MODEL PREDICTIONS
# =============================================================================

print_section(
    "VALIDATING COMPLETE SENTIMENT OUTPUT"
)


if len(checkpoint) != EXPECTED_ROWS:

    raise AssertionError(
        "\nSentiment prediction count mismatch."
        f"\nExpected: {EXPECTED_ROWS:,}"
        f"\nFound:    {len(checkpoint):,}"
    )


if checkpoint[
    "post_id"
].duplicated().any():

    raise AssertionError(
        "Duplicate prediction post IDs detected."
    )


source_ids = set(
    df["post_id"]
)


prediction_ids = set(
    checkpoint[
        "post_id"
    ].astype(str)
)


if source_ids != prediction_ids:

    missing_predictions = (
        source_ids
        - prediction_ids
    )

    extra_predictions = (
        prediction_ids
        - source_ids
    )

    raise AssertionError(
        "Prediction ID membership does not exactly "
        "match the Stage 02 primary sample."
        f"\nMissing predictions: {len(missing_predictions):,}"
        f"\nUnexpected predictions: {len(extra_predictions):,}"
    )


# =============================================================================
# 29. VALIDATE NUMERIC OUTPUTS
# =============================================================================

numeric_prediction_columns = [
    "prob_negative",
    "prob_neutral",
    "prob_positive",
    "sentiment_score",
    "max_probability",
]


for column in numeric_prediction_columns:

    checkpoint[column] = (
        validate_numeric_finite(
            checkpoint[column],
            column,
        )
    )


# =============================================================================
# 30. VALIDATE PROBABILITY RANGES
# =============================================================================

probability_columns = [
    "prob_negative",
    "prob_neutral",
    "prob_positive",
    "max_probability",
]


for column in probability_columns:

    valid_range = (
        checkpoint[column]
        .between(
            0,
            1,
            inclusive="both",
        )
    )


    if not valid_range.all():

        raise AssertionError(
            f"{column} contains values outside [0, 1]."
        )


# =============================================================================
# 31. VALIDATE PROBABILITIES SUM TO ONE
# =============================================================================

probability_sums = (
    checkpoint[
        [
            "prob_negative",
            "prob_neutral",
            "prob_positive",
        ]
    ]
    .sum(
        axis=1
    )
)


if not np.allclose(
    probability_sums,
    1.0,
    atol=1e-5,
):

    maximum_deviation = float(
        np.max(
            np.abs(
                probability_sums
                - 1
            )
        )
    )

    raise AssertionError(
        "Class probabilities do not sum to 1."
        f"\nMaximum deviation: {maximum_deviation}"
    )


# =============================================================================
# 32. VALIDATE CONTINUOUS SENTIMENT RANGE
# =============================================================================

if not checkpoint[
    "sentiment_score"
].between(
    -1,
    1,
    inclusive="both",
).all():

    raise AssertionError(
        "sentiment_score contains values outside [-1, 1]."
    )


# =============================================================================
# 33. VALIDATE SENTIMENT LABELS
# =============================================================================

checkpoint[
    "sentiment_label"
] = (
    checkpoint[
        "sentiment_label"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)


actual_labels = set(
    checkpoint[
        "sentiment_label"
    ].unique()
)


if actual_labels != VALID_SENTIMENT_LABELS:

    raise AssertionError(
        "Unexpected sentiment label set."
        f"\nExpected: {VALID_SENTIMENT_LABELS}"
        f"\nFound:    {actual_labels}"
    )


print(
    "Complete model prediction validation: PASS"
)


# =============================================================================
# 34. MERGE PREDICTIONS BACK TO ORIGINAL STAGE 02 ORDER
# =============================================================================

final = df.merge(
    checkpoint[
        prediction_columns
    ],
    on="post_id",
    how="left",
    validate="one_to_one",
)


final = (
    final
    .sort_values(
        "_source_order"
    )
    .reset_index(
        drop=True
    )
)


if len(final) != EXPECTED_ROWS:

    raise AssertionError(
        "Row count changed after merging sentiment predictions."
    )


if final[
    [
        "prob_negative",
        "prob_neutral",
        "prob_positive",
        "sentiment_score",
        "sentiment_label",
    ]
].isna().any().any():

    raise AssertionError(
        "Missing predictions detected after final merge."
    )


# =============================================================================
# 35. SAVE PRIMARY POST-LEVEL SENTIMENT DATASET
# =============================================================================

print_section(
    "SAVING PRIMARY POST-LEVEL SENTIMENT DATASET"
)


final_output_columns = [
    "post_id",
    "post_date",
    "asset",
    "subreddit",
    "analysis_text",
    "prob_negative",
    "prob_neutral",
    "prob_positive",
    "sentiment_score",
    "sentiment_label",
    "max_probability",
    "token_length",
    "was_truncated",
]


final_output = final[
    final_output_columns
].copy()


final_output[
    "post_date"
] = (
    final_output[
        "post_date"
    ]
    .dt.strftime(
        "%Y-%m-%d"
    )
)


save_and_reload_csv(
    final_output,
    FINAL_SENTIMENT_FILE,
)


print(
    "\nSaved:"
)

print(
    FINAL_SENTIMENT_FILE
)


# =============================================================================
# 36. OVERALL SENTIMENT SUMMARY
# =============================================================================

print_section(
    "CREATING SENTIMENT DIAGNOSTICS"
)


overall_summary = pd.DataFrame(
    {
        "metric": [
            "n_posts",
            "mean_sentiment_score",
            "median_sentiment_score",
            "std_sentiment_score",
            "min_sentiment_score",
            "max_sentiment_score",
            "mean_prob_negative",
            "mean_prob_neutral",
            "mean_prob_positive",
            "mean_max_probability",
            "n_truncated",
            "pct_truncated",
        ],

        "value": [
            len(final),

            final[
                "sentiment_score"
            ].mean(),

            final[
                "sentiment_score"
            ].median(),

            final[
                "sentiment_score"
            ].std(),

            final[
                "sentiment_score"
            ].min(),

            final[
                "sentiment_score"
            ].max(),

            final[
                "prob_negative"
            ].mean(),

            final[
                "prob_neutral"
            ].mean(),

            final[
                "prob_positive"
            ].mean(),

            final[
                "max_probability"
            ].mean(),

            int(
                final[
                    "was_truncated"
                ].sum()
            ),

            (
                final[
                    "was_truncated"
                ].mean()
                * 100
            ),
        ],
    }
)


save_and_reload_csv(
    overall_summary,
    OVERALL_SUMMARY_FILE,
)


# =============================================================================
# 37. SENTIMENT LABEL DISTRIBUTION
# =============================================================================

label_distribution = (
    final
    .groupby(
        [
            "asset",
            "sentiment_label",
        ],
        observed=True,
    )
    .size()
    .reset_index(
        name="n_posts"
    )
)


label_distribution[
    "pct_within_asset"
] = (
    label_distribution[
        "n_posts"
    ]
    /
    label_distribution
    .groupby(
        "asset"
    )[
        "n_posts"
    ]
    .transform(
        "sum"
    )
    * 100
)


save_and_reload_csv(
    label_distribution,
    LABEL_DISTRIBUTION_FILE,
)


# =============================================================================
# 38. GROUPED SUMMARY FUNCTION
# =============================================================================

def grouped_sentiment_summary(
    data,
    group_columns,
):

    summary = (
        data
        .groupby(
            group_columns,
            observed=True,
        )
        .agg(
            n_posts=(
                "post_id",
                "size",
            ),

            mean_sentiment_score=(
                "sentiment_score",
                "mean",
            ),

            median_sentiment_score=(
                "sentiment_score",
                "median",
            ),

            std_sentiment_score=(
                "sentiment_score",
                "std",
            ),

            min_sentiment_score=(
                "sentiment_score",
                "min",
            ),

            max_sentiment_score=(
                "sentiment_score",
                "max",
            ),

            mean_prob_negative=(
                "prob_negative",
                "mean",
            ),

            mean_prob_neutral=(
                "prob_neutral",
                "mean",
            ),

            mean_prob_positive=(
                "prob_positive",
                "mean",
            ),

            mean_max_probability=(
                "max_probability",
                "mean",
            ),

            n_truncated=(
                "was_truncated",
                "sum",
            ),
        )
        .reset_index()
    )


    summary[
        "pct_truncated"
    ] = (
        summary[
            "n_truncated"
        ]
        /
        summary[
            "n_posts"
        ]
        * 100
    )


    return summary


# =============================================================================
# 39. SENTIMENT BY ASSET
# =============================================================================

by_asset = (
    grouped_sentiment_summary(
        final,
        ["asset"],
    )
)


save_and_reload_csv(
    by_asset,
    BY_ASSET_FILE,
)


# =============================================================================
# 40. SENTIMENT BY YEAR
# =============================================================================

by_year = (
    grouped_sentiment_summary(
        final,
        ["year"],
    )
)


save_and_reload_csv(
    by_year,
    BY_YEAR_FILE,
)


# =============================================================================
# 41. SENTIMENT BY YEAR × ASSET
# =============================================================================

by_year_asset = (
    grouped_sentiment_summary(
        final,
        [
            "year",
            "asset",
        ],
    )
)


save_and_reload_csv(
    by_year_asset,
    BY_YEAR_ASSET_FILE,
)


# =============================================================================
# 42. DETAILED TRUNCATION QC
# =============================================================================

truncation_qc = (
    final
    .groupby(
        [
            "asset",
            "year",
        ],
        observed=True,
    )
    .agg(
        n_posts=(
            "post_id",
            "size",
        ),

        n_truncated=(
            "was_truncated",
            "sum",
        ),

        mean_token_length=(
            "token_length",
            "mean",
        ),

        median_token_length=(
            "token_length",
            "median",
        ),

        max_token_length=(
            "token_length",
            "max",
        ),
    )
    .reset_index()
)


truncation_qc[
    "pct_truncated"
] = (
    truncation_qc[
        "n_truncated"
    ]
    /
    truncation_qc[
        "n_posts"
    ]
    * 100
)


save_and_reload_csv(
    truncation_qc,
    TRUNCATION_QC_FILE,
)


# =============================================================================
# 43. LOW-CONFIDENCE DIAGNOSTIC SAMPLE
# =============================================================================

# No confidence threshold is used to exclude observations.
#
# This is purely a qualitative QC output.
#
# The 100 observations for which the model has the lowest maximum
# probability are retained for inspection.

low_confidence = (
    final
    .sort_values(
        "max_probability",
        ascending=True,
    )
    .head(
        100
    )
    [
        [
            "post_id",
            "post_date",
            "asset",
            "subreddit",
            "analysis_text",
            "sentiment_label",
            "sentiment_score",
            "prob_negative",
            "prob_neutral",
            "prob_positive",
            "max_probability",
        ]
    ]
    .copy()
)


low_confidence[
    "post_date"
] = (
    low_confidence[
        "post_date"
    ]
    .dt.strftime(
        "%Y-%m-%d"
    )
)


save_and_reload_csv(
    low_confidence,
    LOW_CONFIDENCE_FILE,
)


# =============================================================================
# 44. CREATE BLINDED MANUAL VALIDATION SAMPLE
# =============================================================================

print_section(
    "MANUAL SENTIMENT VALIDATION"
)


# -----------------------------------------------------------------------------
# IMPORTANT:
#
# Never overwrite an existing manual template.
#
# Once you start manually entering labels, those labels must be protected.
# -----------------------------------------------------------------------------

if not MANUAL_TEMPLATE_FILE.exists():

    manual_sample_parts = []


    for (
        asset,
        year
    ), group in final.groupby(
        [
            "asset",
            "year",
        ],
        observed=True,
    ):


        if len(group) < MANUAL_SAMPLE_PER_ASSET_YEAR:

            raise AssertionError(
                f"Insufficient observations for "
                f"{asset}, {year} manual validation."
            )


        # Asset/year-specific reproducible seed

        stratum_seed = (
            RANDOM_SEED
            + int(year)
            + (
                0
                if asset == "BTC"
                else 10_000
            )
        )


        sampled = group.sample(
            n=MANUAL_SAMPLE_PER_ASSET_YEAR,
            random_state=stratum_seed,
        )


        manual_sample_parts.append(
            sampled
        )


    manual_sample = pd.concat(
        manual_sample_parts,
        ignore_index=True,
    )


    expected_manual_sample_size = (
        MANUAL_SAMPLE_PER_ASSET_YEAR
        * len(EXPECTED_ASSETS)
        * (
            EXPECTED_END_YEAR
            - EXPECTED_START_YEAR
            + 1
        )
    )


    if (
        len(manual_sample)
        != expected_manual_sample_size
    ):

        raise AssertionError(
            "Unexpected manual validation sample size."
        )


    if manual_sample[
        "post_id"
    ].duplicated().any():

        raise AssertionError(
            "Duplicate IDs detected in manual validation sample."
        )


    # -------------------------------------------------------------------------
    # Randomise display order so observations are not presented in year blocks.
    # -------------------------------------------------------------------------

    manual_sample = (
        manual_sample
        .sample(
            frac=1,
            random_state=RANDOM_SEED,
        )
        .reset_index(
            drop=True
        )
    )


    # -------------------------------------------------------------------------
    # BLINDED HUMAN-CODING FILE
    #
    # Model predictions intentionally excluded.
    # -------------------------------------------------------------------------

    manual_blinded = manual_sample[
        [
            "post_id",
            "post_date",
            "asset",
            "subreddit",
            "analysis_text",
        ]
    ].copy()


    manual_blinded[
        "human_label"
    ] = ""


    manual_blinded[
        "post_date"
    ] = (
        manual_blinded[
            "post_date"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )


    manual_blinded.to_csv(
        MANUAL_TEMPLATE_FILE,
        index=False,
    )


    # -------------------------------------------------------------------------
    # Separately preserve the model predictions for exactly those IDs.
    #
    # Do NOT look at this file while manually coding.
    # -------------------------------------------------------------------------

    manual_predictions = manual_sample[
        [
            "post_id",
            "sentiment_label",
            "sentiment_score",
            "prob_negative",
            "prob_neutral",
            "prob_positive",
            "max_probability",
        ]
    ].copy()


    manual_predictions.to_csv(
        MANUAL_PREDICTIONS_FILE,
        index=False,
    )


    print(
        "\nCreated blinded manual-validation sample."
    )


    print(
        f"\nNumber of manual observations: "
        f"{len(manual_blinded):,}"
    )


    print(
        "\nHuman coding file:"
    )

    print(
        MANUAL_TEMPLATE_FILE
    )


    print(
        "\nEnter ONLY one of the following in human_label:"
        "\nnegative"
        "\nneutral"
        "\npositive"
    )


    print(
        "\nDo NOT view the separate prediction file "
        "until human coding is complete."
    )


else:

    print(
        "Existing manual-validation file detected."
    )

    print(
        "The file will NOT be overwritten."
    )


# =============================================================================
# 45. CHECK MANUAL HUMAN LABELS
# =============================================================================

manual_validation_status = (
    "PENDING HUMAN LABELS"
)


manual = pd.read_csv(
    MANUAL_TEMPLATE_FILE
)


if "human_label" not in manual.columns:

    raise AssertionError(
        "Manual validation file does not contain human_label."
    )


human_labels = (
    manual[
        "human_label"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)


completed_manual_mask = (
    human_labels
    .isin(
        VALID_SENTIMENT_LABELS
    )
)


invalid_nonblank_mask = (
    human_labels.ne("")
    &
    ~completed_manual_mask
)


if invalid_nonblank_mask.any():

    invalid_labels = sorted(
        set(
            human_labels[
                invalid_nonblank_mask
            ]
        )
    )

    raise AssertionError(
        "Invalid human sentiment labels detected."
        f"\nInvalid values: {invalid_labels}"
        "\nAllowed values are:"
        "\nnegative"
        "\nneutral"
        "\npositive"
    )


number_manually_completed = int(
    completed_manual_mask.sum()
)


print(
    f"\nManual labels completed: "
    f"{number_manually_completed:,}/"
    f"{len(manual):,}"
)


# =============================================================================
# 46. COMPUTE HUMAN-vs-MODEL VALIDATION IF ALL LABELS COMPLETE
# =============================================================================

if completed_manual_mask.all():

    print(
        "\nAll human labels are complete."
    )

    print(
        "Calculating manual validation statistics..."
    )


    manual_predictions = pd.read_csv(
        MANUAL_PREDICTIONS_FILE
    )


    manual_predictions[
        "post_id"
    ] = (
        manual_predictions[
            "post_id"
        ]
        .astype(str)
    )


    manual[
        "post_id"
    ] = (
        manual[
            "post_id"
        ]
        .astype(str)
    )


    validation = manual.merge(
        manual_predictions[
            [
                "post_id",
                "sentiment_label",
            ]
        ],
        on="post_id",
        how="left",
        validate="one_to_one",
    )


    if validation[
        "sentiment_label"
    ].isna().any():

        raise AssertionError(
            "Missing model predictions in manual validation merge."
        )


    validation[
        "human_label"
    ] = (
        validation[
            "human_label"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    validation[
        "sentiment_label"
    ] = (
        validation[
            "sentiment_label"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    y_true = validation[
        "human_label"
    ]


    y_pred = validation[
        "sentiment_label"
    ]


    # -------------------------------------------------------------------------
    # Accuracy
    # -------------------------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )


    # -------------------------------------------------------------------------
    # Per-class precision / recall / F1
    # -------------------------------------------------------------------------

    ordered_labels = [
        "negative",
        "neutral",
        "positive",
    ]


    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=ordered_labels,
        zero_division=0,
    )


    macro_precision = float(
        np.mean(
            precision
        )
    )


    macro_recall = float(
        np.mean(
            recall
        )
    )


    macro_f1 = float(
        np.mean(
            f1
        )
    )


    # -------------------------------------------------------------------------
    # Cohen's kappa
    # -------------------------------------------------------------------------

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
    )


    manual_results = pd.DataFrame(
        {
            "metric": [
                "n_manual_posts",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "cohen_kappa",
            ],

            "value": [
                len(validation),
                accuracy,
                macro_precision,
                macro_recall,
                macro_f1,
                kappa,
            ],
        }
    )


    save_and_reload_csv(
        manual_results,
        MANUAL_RESULTS_FILE,
    )


    # -------------------------------------------------------------------------
    # Per-class results
    # -------------------------------------------------------------------------

    class_metrics = pd.DataFrame(
        {
            "class": ordered_labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


    save_and_reload_csv(
        class_metrics,
        MANUAL_CLASS_METRICS_FILE,
    )


    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=ordered_labels,
    )


    confusion_df = pd.DataFrame(
        matrix,

        index=[
            "actual_negative",
            "actual_neutral",
            "actual_positive",
        ],

        columns=[
            "pred_negative",
            "pred_neutral",
            "pred_positive",
        ],
    )


    confusion_df.to_csv(
        MANUAL_CONFUSION_FILE,
        index=True,
    )


    manual_validation_status = (
        "COMPLETE"
    )


    print(
        "\nManual sentiment validation: COMPLETE"
    )


    print(
        f"Accuracy:    {accuracy:.4f}"
    )


    print(
        f"Macro-F1:    {macro_f1:.4f}"
    )


    print(
        f"Cohen kappa: {kappa:.4f}"
    )


else:

    print(
        "\nManual sentiment validation remains PENDING."
    )

    print(
        "This is expected on the first Stage 04 run."
    )


# =============================================================================
# 47. SAVE MODEL AND METHODOLOGY METADATA
# =============================================================================

print_section(
    "SAVING MODEL / METHODOLOGY METADATA"
)


model_commit = getattr(
    model.config,
    "_commit_hash",
    None,
)


metadata = {

    "dissertation_stage": (
        "04_post_level_sentiment_analysis"
    ),

    "created_utc": (
        datetime
        .now(
            timezone.utc
        )
        .isoformat()
    ),

    "input_file": str(
        INPUT_FILE
    ),

    "input_rows": int(
        len(df)
    ),

    "model_name": MODEL_NAME,

    "model_commit_hash_if_available": (
        model_commit
    ),

    "model_type": (
        model.config.model_type
    ),

    "model_id2label": {
        str(key): value
        for key, value
        in id2label.items()
    },

    "unit_of_analysis": (
        "individual retained Reddit post"
    ),

    "text_variable": (
        "analysis_text"
    ),

    "sentiment_score_formula": (
        "P(positive) - P(negative)"
    ),

    "categorical_label_rule": (
        "class with maximum predicted probability"
    ),

    "probability_variables": [
        "prob_negative",
        "prob_neutral",
        "prob_positive",
    ],

    "max_token_length": (
        MAX_LENGTH
    ),

    "batch_size": (
        BATCH_SIZE
    ),

    "checkpoint_every_posts": (
        CHECKPOINT_EVERY
    ),

    "random_seed": (
        RANDOM_SEED
    ),

    "device": str(
        device
    ),

    "python_version": (
        sys.version
    ),

    "platform": (
        platform.platform()
    ),

    "torch_version": (
        torch.__version__
    ),

    "transformers_version": (
        transformers.__version__
    ),

    "number_posts_truncated": (
        number_truncated
    ),

    "percentage_posts_truncated": (
        percentage_truncated
    ),

    "manual_validation_sample_per_asset_year": (
        MANUAL_SAMPLE_PER_ASSET_YEAR
    ),

    "manual_validation_total_planned": (
        MANUAL_SAMPLE_PER_ASSET_YEAR
        * 2
        * 5
    ),

    "manual_validation_status": (
        manual_validation_status
    ),

    "stage_boundaries": {
        "daily_aggregation": False,
        "lag_creation": False,
        "market_merge": False,
        "forecasting": False,
    },
}


with open(
    MODEL_METADATA_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
    )


print(
    f"Saved:\n{MODEL_METADATA_FILE}"
)


# =============================================================================
# 48. FINAL QC TABLE
# =============================================================================

print_section(
    "FINAL STAGE 04 QC"
)


final_qc = pd.DataFrame(
    {
        "check": [

            "Input row count correct",

            "Final sentiment row count correct",

            "Unique post IDs",

            "Exact post-ID membership preserved",

            "No missing sentiment probabilities",

            "Probabilities within zero-one range",

            "Probabilities sum approximately to one",

            "Sentiment score within minus-one to plus-one",

            "Valid sentiment labels only",

            "BTC and ETH both present",

            "Years 2021 through 2025 present",

            "Truncation explicitly audited",

            "Manual validation sample created",

            "Stage 02 input remained read-only",
        ],

        "result": [

            len(df) == EXPECTED_ROWS,

            len(final_output) == EXPECTED_ROWS,

            not final_output[
                "post_id"
            ].duplicated().any(),

            set(
                final_output[
                    "post_id"
                ].astype(str)
            )
            ==
            set(
                df[
                    "post_id"
                ].astype(str)
            ),

            not final_output[
                [
                    "prob_negative",
                    "prob_neutral",
                    "prob_positive",
                ]
            ].isna().any().any(),

            bool(
                final_output[
                    [
                        "prob_negative",
                        "prob_neutral",
                        "prob_positive",
                    ]
                ]
                .ge(0)
                .all()
                .all()
                and
                final_output[
                    [
                        "prob_negative",
                        "prob_neutral",
                        "prob_positive",
                    ]
                ]
                .le(1)
                .all()
                .all()
            ),

            bool(
                np.allclose(
                    final_output[
                        [
                            "prob_negative",
                            "prob_neutral",
                            "prob_positive",
                        ]
                    ]
                    .sum(
                        axis=1
                    ),
                    1.0,
                    atol=1e-5,
                )
            ),

            bool(
                final_output[
                    "sentiment_score"
                ]
                .between(
                    -1,
                    1,
                    inclusive="both",
                )
                .all()
            ),

            set(
                final_output[
                    "sentiment_label"
                ]
                .astype(str)
                .unique()
            )
            ==
            VALID_SENTIMENT_LABELS,

            set(
                final_output[
                    "asset"
                ]
                .unique()
            )
            ==
            EXPECTED_ASSETS,

            (
                final[
                    "year"
                ].min()
                == EXPECTED_START_YEAR
                and
                final[
                    "year"
                ].max()
                == EXPECTED_END_YEAR
            ),

            True,

            MANUAL_TEMPLATE_FILE.exists(),

            True,
        ],
    }
)


save_and_reload_csv(
    final_qc,
    FINAL_QC_FILE,
)


failed_qc = (
    ~final_qc[
        "result"
    ]
)


if failed_qc.any():

    print(
        "\nFAILED FINAL QC CHECKS:"
    )

    print(
        final_qc[
            failed_qc
        ]
    )

    raise AssertionError(
        "Stage 04 final QC failed."
    )


print(
    "\nAll automated Stage 04 QC checks: PASS"
)


# =============================================================================
# 49. RELOAD PRIMARY FINAL DATASET AND VERIFY AGAIN
# =============================================================================

print_section(
    "FINAL FILE RELOAD / INTEGRITY CHECK"
)


reloaded_final = pd.read_csv(
    FINAL_SENTIMENT_FILE
)


if len(reloaded_final) != EXPECTED_ROWS:

    raise AssertionError(
        "Final saved sentiment file failed row-count reload validation."
    )


if reloaded_final[
    "post_id"
].duplicated().any():

    raise AssertionError(
        "Final saved sentiment file contains duplicate post IDs."
    )


if (
    set(
        reloaded_final[
            "post_id"
        ].astype(str)
    )
    !=
    set(
        df[
            "post_id"
        ].astype(str)
    )
):

    raise AssertionError(
        "Final saved sentiment file failed post-ID membership validation."
    )


print(
    "Final output reload validation: PASS"
)


# =============================================================================
# 50. DISPLAY MAIN RESULTS
# =============================================================================

print_section(
    "STAGE 04 AUTOMATED RESULTS"
)


print(
    f"Posts scored: "
    f"{len(reloaded_final):,}"
)


print(
    "\nSentiment label counts:"
)


print(
    reloaded_final[
        "sentiment_label"
    ]
    .value_counts()
)


print(
    "\nMean sentiment by asset:"
)


print(
    final
    .groupby(
        "asset"
    )[
        "sentiment_score"
    ]
    .mean()
)


print(
    "\nTruncated posts:"
    f" {number_truncated:,}"
    f" ({percentage_truncated:.4f}%)"
)


print(
    "\nManual sentiment validation status:"
    f" {manual_validation_status}"
)


# =============================================================================
# 51. LIST OUTPUT FILES
# =============================================================================

print_section(
    "STAGE 04 OUTPUT FILES"
)


output_files = [
    FINAL_SENTIMENT_FILE,
    OVERALL_SUMMARY_FILE,
    LABEL_DISTRIBUTION_FILE,
    BY_ASSET_FILE,
    BY_YEAR_FILE,
    BY_YEAR_ASSET_FILE,
    TRUNCATION_QC_FILE,
    LOW_CONFIDENCE_FILE,
    MODEL_METADATA_FILE,
    FINAL_QC_FILE,
    MANUAL_TEMPLATE_FILE,
    MANUAL_PREDICTIONS_FILE,
]


if MANUAL_RESULTS_FILE.exists():
    output_files.append(
        MANUAL_RESULTS_FILE
    )


if MANUAL_CLASS_METRICS_FILE.exists():
    output_files.append(
        MANUAL_CLASS_METRICS_FILE
    )


if MANUAL_CONFUSION_FILE.exists():
    output_files.append(
        MANUAL_CONFUSION_FILE
    )


for number, path in enumerate(
    output_files,
    start=1,
):

    print(
        f"{number}. {path.name}"
    )


# =============================================================================
# 52. FINAL METHODOLOGICAL STATUS
# =============================================================================

print_section(
    "STAGE 04 STATUS"
)


print(
    "AUTOMATED POST-LEVEL SENTIMENT ANALYSIS: COMPLETE"
)


print(
    "\nPrimary sentiment measure:"
)


print(
    "Sentiment_i = P(Positive)_i - P(Negative)_i"
)


print(
    "\nNo daily aggregation has been performed."
)


print(
    "No market variables have been merged."
)


print(
    "No forecasting lags have been constructed."
)


if manual_validation_status == "COMPLETE":

    print(
        "\nMANUAL SENTIMENT VALIDATION: COMPLETE"
    )

    print(
        "\nStage 04 may now be reviewed and formally frozen "
        "before proceeding to Stage 05."
    )

else:

    print(
        "\nMANUAL SENTIMENT VALIDATION: PENDING"
    )

    print(
        "\nNext action:"
    )

    print(
        "Open:"
    )

    print(
        MANUAL_TEMPLATE_FILE
    )

    print(
        "\nManually label all sampled posts as:"
    )

    print(
        "negative / neutral / positive"
    )

    print(
        "\nSave that SAME CSV without changing its filename."
    )

    print(
        "\nThen rerun this script."
    )

    print(
        "The existing model predictions will be reused from the checkpoint,"
        "\nso the 136,019 posts will NOT need to be scored again."
    )

    print(
        "\nThe script will then automatically calculate:"
    )

    print(
        "- accuracy"
        "\n- macro precision"
        "\n- macro recall"
        "\n- macro F1"
        "\n- Cohen's kappa"
        "\n- confusion matrix"
    )


print_section(
    "END OF STAGE 04"
)