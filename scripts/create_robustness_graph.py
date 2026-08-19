import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/robustness_results.csv")

conditions = ["Original", "Paraphrase", "Noise"]

x = range(len(conditions))

plt.figure(figsize=(8,5))

plt.plot(
    conditions,
    df.iloc[0][conditions],
    marker="o",
    label="Qwen2.5-3B"
)

plt.plot(
    conditions,
    df.iloc[1][conditions],
    marker="o",
    label="Qwen2.5-1.5B"
)

plt.xlabel("Dataset Type")
plt.ylabel("Accuracy (%)")
plt.title("LLM Robustness Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/robustness_comparison.png"
)

print("Graph saved")