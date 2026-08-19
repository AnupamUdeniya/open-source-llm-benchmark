import pandas as pd

df = pd.read_csv("results/robustness_results.csv")


results = []


for _, row in df.iterrows():

    model = row["Model"]

    original = row["Original"]

    paraphrase_drop = original - row["Paraphrase"]

    noise_drop = original - row["Noise"]


    results.append({
        "Model": model,
        "Paraphrase_Drop": paraphrase_drop,
        "Noise_Drop": noise_drop
    })


drop_df = pd.DataFrame(results)


drop_df.to_csv(
    "results/robustness_drop.csv",
    index=False
)


print(drop_df)

print("\nSaved: results/robustness_drop.csv")