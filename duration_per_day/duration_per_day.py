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
import math

# Review count: User id
# Median: 8798
# Most: 6810

def q_fast(array):
    n = len(array)
    if n < 2:
        raise Exception('n<2')

    # Same correction factors as original
    small_sample_correction_factors = {2: 0.399, 3: 0.994, 4: 0.512, 5: 0.844, 6: 0.611, 7: 0.857, 8: 0.669, 9: 0.872}
    if n <= 9:
        correction_factor = small_sample_correction_factors.get(n)
    else:
        if n % 2 == 0:
            correction_factor = n / (n + 3.8)
        else:
            correction_factor = n / (n + 1.4)

    const = correction_factor * 2.21914
    quartile = math.comb(math.floor(n / 2) + 1, 2) - 1

    # Sort the array first - O(n log n)
    sorted_array = sorted(array)

    # Binary search on the answer
    left, right = 0, sorted_array[-1] - sorted_array[0]

    def count_pairs_leq(target):
        """Count how many |x_i - x_j| <= target for i < j"""
        count = 0
        j = 1
        for i in range(n):
            # For sorted array, |x_i - x_j| = x_j - x_i for j > i
            # Find largest j such that x_j - x_i <= target
            while j < n and sorted_array[j] - sorted_array[i] <= target:
                j += 1
            # All pairs (i, k) where i < k < j satisfy the condition
            count += max(0, j - i - 1)
        return count

    # Binary search to find the exact value
    while right - left > 1e-10:
        mid = (left + right) / 2
        if count_pairs_leq(mid) <= quartile:
            left = mid
        else:
            right = mid

    return const * left


def huber(a):
    array = np.asarray(a)
    scale = q_fast(array) # estimator of scale has to be robust

    if scale == 0:
        if np.max(array) == np.min(array):  # all values are the same
            return np.mean(array)
        else:
            raise Exception('Scale=0')

    prev_val = np.median(array)  # initial guess, estimator of location also has to be robust
    maxiter = 1000
    tol = 1e-5
    c = 1.339  # roughly 95% asymptotic relative efficiency for a normal distribution

    def weighted_mean(array, prev_val):
        epsilon = 1e-15 # this is just to avoid division by 0 error
        weights = np.where(np.abs(array - prev_val) / scale <= c, 1,
                           np.abs(c / ((array - prev_val + epsilon) / scale)))  # this is faster than a for loop

        weighted_mean = np.average(array, weights=weights)
        return weighted_mean

    next_val = weighted_mean(array, prev_val)

    iter_count = 0
    while np.abs(prev_val - next_val) > tol and iter_count <= maxiter:
        prev_val = next_val
        next_val = weighted_mean(array, prev_val)
        iter_count += 1

    return next_val

def process_user(user_id):
    df = pd.read_parquet(
        "../anki-revlogs-10k/revlogs",
        filters=[("user_id", "=", user_id), ("rating", "in", [1, 2, 3, 4])],
        columns=["card_id", "day_offset", "rating", "elapsed_days", "duration"],
    )
    df["y"] = df["rating"].map({1: 0, 2: 1, 3: 1, 4: 1})
    df.size
    df_filtered = df[(df["elapsed_days"] > 0)].copy()

    columns = ["duration", "review_median", "review_mean", "review_huber", "y"]

    unique_days = df_filtered["day_offset"].unique()

    train_days, test_days = train_test_split(unique_days)

    train_df = df_filtered[df_filtered["day_offset"].isin(train_days)]
    test_df = df_filtered[df_filtered["day_offset"].isin(test_days)]

    median_groups = train_df.groupby("rating")["duration"].median()
    mean_groups = train_df.groupby("rating")["duration"].mean()
    huber_groups = train_df.groupby("rating")["duration"].apply(huber)

    df_filtered["review_median"] = df_filtered["rating"].map(median_groups)
    df_filtered["review_mean"] = df_filtered["rating"].map(mean_groups)
    df_filtered["review_huber"] = df_filtered["rating"].map(huber_groups)

    train_df = df_filtered[df_filtered["day_offset"].isin(train_days)]
    test_df = df_filtered[df_filtered["day_offset"].isin(test_days)]

    train_groups = train_df.groupby("day_offset")[columns]
    test_groups = test_df.groupby("day_offset")[columns]

    def counts_sums(groups):
        # Compute count and sum
        counts = groups.count()
        sums = groups.sum()

        # Sort by count of 'duration' (or 'review_median' – whichever makes sense)
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
        return {"loss": loss(test_sums["review_median"])}

    def rating_means():
        return {"loss": loss(test_sums["review_mean"])}
    
    def rating_huber():
        return {"loss": loss(test_sums["review_huber"])}

    def mean_multiplier():
        multiplier = (train_sums["duration"] / train_sums["review_median"]).mean()
        return {
            "multiplier": multiplier,
            "loss": loss(test_sums["review_median"] * multiplier),
        }

    def median_multiplier():
        multiplier = (train_sums["duration"] / train_sums["review_median"]).median()
        return {
            "multiplier": multiplier,
            "loss": loss(test_sums["review_median"] * multiplier),
        }

    default_fit = lambda x, y: np.polyfit(x, y, 1)

    def review_trend(fit=default_fit):
        difference = train_sums["duration"] / train_sums["review_median"]
        c = fit(train_sums["review_median"], difference)
        adjustment = np.polyval(c, test_sums["review_median"])
        return {"c": list(c), "loss": loss(test_sums["review_median"] * adjustment)}

    def true_retention_trend(fit=default_fit):
        retention = train_sums["y"] / train_counts["y"]
        c = fit(
            retention, train_sums["duration"] / train_sums["review_median"]
        )
        retention = test_sums["y"] / test_counts["y"]
        return {
            "c": list(c),
            "loss": loss(test_sums["review_median"] * np.polyval(c, retention)),
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
            rating_means,
            mean_multiplier,
            median_multiplier,
            review_trend,
            true_retention_trend,
            review_trend_siegel,
            true_retention_trend_siegel,
            rating_huber
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