import torch, pdb
from transformers import pipeline


model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "feature-extraction",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

data = pipe("this is a test")
pdb.set_trace()
print(data)
"""

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a financial advisor and CPA chatbot who always responds to help people!"},
    {"role": "user", "content": "If I am a visitor in the US holding a B visa, do I need to file and pay for taxes if I have some interest in come from having money in the bank?"},
]
outputs = pipe(
    messages,
    max_new_tokens=2560,
)
print(outputs[0]["generated_text"][-1])
"""
