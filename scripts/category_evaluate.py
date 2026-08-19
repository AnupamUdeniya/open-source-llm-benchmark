import pandas as pd

MODEL_FILE = "outputs/Qwen2.5-3B-Instruct_predictions.csv"

df = pd.read_csv(MODEL_FILE)

categories = df["Category"].unique()

print("\nCATEGORY RESULTS\n")

for category in categories:

    subset = df[df["Category"] == category]

    correct = 0

    for _, row in subset.iterrows():

        gt = str(row["GroundTruth"]).lower().strip()
        pred = str(row["Prediction"]).lower().strip()

        if gt in pred:
            correct += 1

    accuracy = correct / len(subset)

    print(
        f"{category}: {correct}/{len(subset)} = {accuracy:.2%}"
    )