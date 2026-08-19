import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/benchmark_results.csv")

plt.figure(figsize=(8,5))

plt.bar(df["Model"], df["Overall"])

plt.title("Overall Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.xlabel("Model")

plt.tight_layout()

plt.savefig("figures/model_comparison.png")

print("Graph saved to figures/model_comparison.png")