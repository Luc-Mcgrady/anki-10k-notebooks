import json
import matplotlib.pyplot as plt

JSONL_FILE_PATH = "/home/luc/Programming/forks/fatigue-graph/duration_per_day/accuracies.jsonl"

def get_c(line):
    user_data = json.loads(line)
    return user_data["results"]["true_retention_trend_siegel"]["c"][0]

LIM = 10

with open(JSONL_FILE_PATH, 'r') as f:
    c_values = [get_c(line) for line in f]
    c_values = [v for v in c_values if abs(v) < LIM]

plt.xlim(-LIM, LIM)
plt.hist(c_values, bins=50)
plt.savefig("duration_per_day/true_retention_trend_siegel_c.png")