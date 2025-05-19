import pandas as pd
import os
import sys
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath("../fsrs-optimizer/src/fsrs_optimizer/"))
import numpy as np
from sklearn.model_selection import train_test_split

# Review count: User id
# Median: 8798
# Most: 6810


def process_user(user_id):
    df = pd.read_parquet(
        "../anki-revlogs-10k/revlogs",
        filters=[("user_id", "=", user_id), ("rating", "in", [1, 2, 3, 4])],
        columns=["card_id", "day_offset", "rating", "elapsed_days", "duration"],
    )
    df["y"] = df["rating"].map({1: 0, 2: 1, 3: 1, 4: 1})
    df.size
    df_filtered = df[(df["elapsed_days"] > 0)]

    groups = df_filtered.groupby("rating")["duration"].median()

    df_filtered["review_average"] = df_filtered["rating"].map(groups)

    current_residual = df_filtered.groupby("rating").apply(
        lambda x: abs(x["duration"] - x["review_average"]).mean()
    )

    current_residual, groups

    columns = ["duration", "review_average", "y"]

    day_accuracies = df_filtered.groupby("day_offset")[columns]
    unique_days = df_filtered["day_offset"].unique()

    train_days, test_days = train_test_split(unique_days)

    train_df = df_filtered[df_filtered["day_offset"].isin(train_days)]
    test_df = df_filtered[df_filtered["day_offset"].isin(test_days)]

    train_groups = train_df.groupby("day_offset")[columns]
    test_groups = test_df.groupby("day_offset")[columns]

    def counts_sums(groups):
        # Compute count and sum
        counts = groups.count()
        sums = groups.sum()

        # Sort by count of 'duration' (or 'review_average' – whichever makes sense)
        sorted_indices = counts["duration"].sort_values().index

        # Reorder both sums and counts to match
        counts = counts.loc[sorted_indices]
        sums = sums.loc[sorted_indices]
        return counts, sums

    train_counts, train_sums = counts_sums(train_groups)
    test_counts, test_sums = counts_sums(test_groups)

    def loss(p: pd.Series):
        return abs(test_sums["duration"] - p).mean() / (60 * 60 * 24)

    def rating_medians():
        return {"loss": loss(test_sums["review_average"])}

    def mean_multiplier():
        multiplier = (train_sums["duration"] / train_sums["review_average"]).mean()
        return {
            "multiplier": multiplier,
            "loss": loss(test_sums["review_average"] * multiplier),
        }

    def median_multiplier():
        multiplier = (train_sums["duration"] / train_sums["review_average"]).median()
        return {
            "multiplier": multiplier,
            "loss": loss(test_sums["review_average"] * multiplier),
        }

    def review_trend():
        difference = train_sums["duration"] / train_sums["review_average"]
        c = np.polyfit(train_counts["review_average"], difference, 1)
        adjustment = np.polyval(c, test_counts["review_average"])
        return {"c": c, "loss": loss(test_counts["review_average"] * adjustment)}

    def true_retention_trend():
        retention = train_sums["y"] / train_counts["y"]
        c = np.polyfit(
            retention, train_sums["review_average"] / train_counts["review_average"], 1
        )
        retention = test_sums["y"] / test_counts["y"]
        return {
            "c": c,
            "loss": loss(test_counts["review_average"] * np.polyval(c, retention)),
        }

    return {
        function.__name__: function()
        for function in [
            rating_medians,
            mean_multiplier,
            median_multiplier,
            review_trend,
            true_retention_trend,
        ]
    }

print(process_user(1))