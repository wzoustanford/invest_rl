import time
from vllm import LLM, SamplingParams

prompts = [
    "The best recipe for pasta is"
]
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=150)

loading_start = time.time()
llm = LLM(model="kaitchup/Qwen1.5-7B-awq-4bit", quantization="awq")
print("--- Loading time: %s seconds ---" % (time.time() - loading_start))

generation_time = time.time()
outputs = llm.generate(prompts, sampling_params)
print("--- Generation time: %s seconds ---" % (time.time() - generation_time))

for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)
    print('------')