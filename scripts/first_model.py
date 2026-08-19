import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

messages = [
    {
        "role": "user",
        "content": "What is the capital of India?"
    }
]

result = pipe(
    messages,
    max_new_tokens=30,
    do_sample=False
)

answer = result[0]["generated_text"][-1]["content"]

print(answer)