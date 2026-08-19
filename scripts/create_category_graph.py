import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/benchmark_results.csv")

categories = [
    "Science",
    "Geography",
    "History",
    "Technology",
    "Mathematics"
]

x = range(len(categories))

plt.figure(figsize=(10,6))

for _, row in df.iterrows():
    plt.plot(
        categories,
        [row[c] for c in categories],
        marker="o",
        label=row["Model"]
    )

plt.title("Category-wise Accuracy")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()

plt.savefig("figures/category_comparison.png")

print("Saved figures/category_comparison.png")