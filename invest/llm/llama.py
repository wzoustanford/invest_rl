import torch, pdb
from transformers import pipeline

"""
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
    {"role": "system", "content": "You are a funny friend and always makes jokes with your baby cousin who is a university undergrad!"},
    {"role": "user", "content": "try to say this in a funny way to you baby couse: How is life and how is research? Are you working hard? Don't be cause your big bro wants you to be happy and spend more fun time with us in California."},
]
outputs = pipe(
    messages,
    max_new_tokens=2560,
)
print(outputs[0]["generated_text"][-1])
