import pandas as pd
import re

MODEL_FILE = "outputs/Qwen2.5-3B-Instruct_predictions.csv"

df = pd.read_csv(MODEL_FILE)
correct = 0
wrong = []

def normalize(text):
    text = str(text).lower()

    # Remove commas
    text = text.replace(",", "")

    # Remove quotes
    text = text.replace('"', "")
    text = text.replace("'", "")

    # Replace multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

for _, row in df.iterrows():

    gt = normalize(row["GroundTruth"])
    pred = normalize(row["Prediction"])

    if gt in pred:
        correct += 1
    else:
        wrong.append(row)

accuracy = correct / len(df)

print("Correct:", correct)
print("Total:", len(df))
print(f"Accuracy: {accuracy:.2%}")

pd.DataFrame(wrong).to_csv(
    "results/wrong_answers.csv",
    index=False
)

print("Wrong answers saved.")