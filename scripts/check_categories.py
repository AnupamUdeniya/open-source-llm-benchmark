import pandas as pd

df = pd.read_csv("data/IndianLLMBenchmark_v1.csv")

print(df["Category"].value_counts())