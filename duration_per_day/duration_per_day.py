from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import os
import sys
import json
from tqdm.auto import tqdm
from scipy.stats import siegelslopes

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
    df_filtered = df[(df["elapsed_days"] > 0)].copy()

    groups = df_filtered.groupby("rating")["duration"].median()

    df_filtered["review_average"] = df_filtered["rating"].map(groups)

    columns = ["duration", "review_average", "y"]

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
        return float(abs(test_sums["duration"] - p).mean() / (1000 * 60)) # Minutes

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

    default_fit = lambda x, y: np.polyfit(x, y, 1)

    def review_trend(fit=default_fit):
        difference = train_sums["duration"] / train_sums["review_average"]
        c = fit(train_sums["review_average"], difference)
        adjustment = np.polyval(c, test_sums["review_average"])
        return {"c": list(c), "loss": loss(test_sums["review_average"] * adjustment)}

    def true_retention_trend(fit=default_fit):
        retention = train_sums["y"] / train_counts["y"]
        c = fit(
            retention, train_sums["duration"] / train_sums["review_average"]
        )
        retention = test_sums["y"] / test_counts["y"]
        return {
            "c": list(c),
            "loss": loss(test_sums["review_average"] * np.polyval(c, retention)),
        }

    siegel = lambda x, y: siegelslopes(y, x)
    
    def review_trend_siegel():
        return review_trend(siegel)
    
    def true_retention_trend_siegel():
        return true_retention_trend(siegel)

    results = {
        function.__name__: function()
        for function in [
            rating_medians,
            mean_multiplier,
            median_multiplier,
            review_trend,
            true_retention_trend,
            review_trend_siegel,
            true_retention_trend_siegel
        ]
    }

    return {"user_id": user_id, "results": results}

OUTPUT_FILE = "duration_per_day/accuracies.jsonl"

processed_ids = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                processed_ids.add(data["user_id"])
            except json.JSONDecodeError:
                continue  # skip malformed lines

# Step 2: Determine which user_ids still need processing
all_ids = set(range(1, 10001))
remaining_ids = sorted(all_ids - processed_ids)

with open(OUTPUT_FILE, "a") as f:
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_user, user_id): user_id for user_id in remaining_ids}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            f.write(json.dumps(result) + "\n")
            f.flush()