import json
import pandas as pd
import numpy as np

# Define the path to your JSONL file
JSONL_FILE_PATH = "/home/luc/Programming/forks/fatigue-graph/duration_per_day/accuracies.jsonl"

# Define the methods we expect to see results for
METHODS = [
  "rating_medians",
  "mean_multiplier",
  "median_multiplier",
  "review_trend",
  "true_retention_trend"
]

# Initialize a dictionary to store win counts
# wins[method_a][method_b] will store how many users method_a was better than method_b
wins = {method_a: {method_b: 0 for method_b in METHODS} for method_a in METHODS}

# Counter for users processed with valid data for comparison
users_with_valid_data = 0

print(f"Reading results from {JSONL_FILE_PATH} and building superiority matrix...")

with open(JSONL_FILE_PATH, 'r') as f:
    for line in f:
        user_data = json.loads(line)
        user_id = user_data.get("user_id")
        method_results = user_data.get("results", {})

        # Skip users with no method results or empty results
        if not method_results:
            continue

        # Extract MAE values for the methods we care about, handling potential None/NaN
        current_maes = {k: v["loss"] for k, v in method_results.items()}
        users_with_valid_data += 1

        # Compare all pairs of methods for this user
        for method_a in current_maes:
            for method_b in current_maes:
                if method_a == method_b:
                    continue # Don't compare a method against itself

                # Method A is "superior" to Method B if its MAE is lower
                if current_maes[method_a] < current_maes[method_b]:
                    wins[method_b][method_a] += 1


# Convert the wins dictionary to a pandas DataFrame for display
superiority_matrix_df = pd.DataFrame(wins)

# Reorder columns and rows to match the defined METHODS list
superiority_matrix_df = superiority_matrix_df.reindex(index=METHODS, columns=METHODS)

print("\nSuperiority Matrix (Number of users where row method had lower MAE than column method):")
print(superiority_matrix_df)

print(f"\nProcessed results from {users_with_valid_data} users with sufficient data for comparison.")

import matplotlib.pyplot as plt

# Create a heatmap
fig, ax = plt.subplots(figsize=(8, 6))
heatmap = ax.imshow(superiority_matrix_df.values, cmap="Greens")

# Set labels
ax.set_xticks(np.arange(len(METHODS)))
ax.set_yticks(np.arange(len(METHODS)))
ax.set_xticklabels(METHODS, rotation=45, ha="right")
ax.set_yticklabels(METHODS)

# Add text annotations for percentages
for i in range(len(METHODS)):
    for j in range(len(METHODS)):
        if i == j: continue # Skip diagonal
        val = superiority_matrix_df.iloc[i, j] / users_with_valid_data
        text_color = "white" if val > 0.5 else "black" # Adjust text color based on cell brightness
        ax.text(j, i, f"{val:.1%}",
                       ha="center", va="center", color=text_color, fontsize=9)

# Add a colorbar
plt.colorbar(heatmap)

plt.title("Superiority Matrix Heatmap")
plt.tight_layout()
plt.savefig("/home/luc/Programming/forks/fatigue-graph/duration_per_day/superiority_matrix.png")
print("\nSuperiority matrix heatmap saved to /home/luc/Programming/forks/fatigue-graph/duration_per_day/superiority_matrix.png")
