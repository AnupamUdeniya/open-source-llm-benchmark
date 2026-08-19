import pandas as pd
import torch
from transformers import pipeline

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("Loading dataset...")
df = pd.read_csv("data/IndianLLMBenchmark_original_test.csv")

print("Questions:", len(df))

print("Loading model...")

pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

print("Model loaded!")
    
results = []

# ONLY FIRST 5 QUESTIONS FOR TESTING
for _, row in df.iterrows():

    question = row["Question"]

    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    output = pipe(
        messages,
        max_new_tokens=50,
        do_sample=False
    )

    answer = output[0]["generated_text"][-1]["content"]

    print("\nQuestion:", question)
    print("Prediction:", answer)

    results.append({
        "ID": row["ID"],
        "Category": row["Category"],
        "Difficulty": row["Difficulty"],
        "Question": question,
        "GroundTruth": row["Answer"],
        "Prediction": answer
    })

results_df = pd.DataFrame(results)

safe_name = MODEL_NAME.split("/")[-1]

results_df.to_csv(
    f"outputs/{safe_name}_original_test_predictions.csv",
    index=False
)

print(f"Saved to outputs/{safe_name}_original_test_predictions.csv")