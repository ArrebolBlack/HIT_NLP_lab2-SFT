# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams

# # model_name = "openbmb/MiniCPM4-0.5B"
# model_name = 'models/minicpm05b'
# prompt = [{"role": "user", "content": "Please recommend 5 tourist attractions in Beijing. "}]

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

# llm = LLM(
#     model=model_name,
#     trust_remote_code=True,
#     max_num_batched_tokens=32768, 
#     dtype="bfloat16", 
#     gpu_memory_utilization=0.8, 
# )
# sampling_params = SamplingParams(top_p=0.7, temperature=0.7, max_tokens=1024, repetition_penalty=1.02)

# outputs = llm.generate(prompts=input_text, sampling_params=sampling_params)

# print(outputs[0].outputs[0].text)

# 没有包装在if __name__ == "__main__":下，spawn报错



# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer

# MODEL_PATH = "models/minicpm05b"   # 或 HF 仓库名

# def main():
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
#     prompt = [{"role": "user", "content": "用三点解释Self-Attention"}]
#     input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

#     llm = LLM(
#         model=MODEL_PATH,
#         dtype="bfloat16",                 # 用 dtype，而不是 torch_dtype
#         gpu_memory_utilization=0.8,
#         max_num_batched_tokens=32768,
#         trust_remote_code=True
#     )

#     params = SamplingParams(top_p=0.7, temperature=0.7, max_tokens=256, repetition_penalty=1.02)
#     outputs = llm.generate(prompts=input_text, sampling_params=params)
#     print(outputs[0].outputs[0].text)

# if __name__ == "__main__":
#     # 可选：显式设为 spawn，防止被其它代码提前初始化 CUDA 后导致模式不一致
#     # import multiprocessing as mp; mp.set_start_method("spawn", force=True)
#     main()



'''
Also, you can start the inference server by running the following command:
Note: In vLLM's chat API, add_special_tokens is False by default. This means important special tokens—such as the beginning-of-sequence (BOS) token—will not be added automatically. To ensure the input prompt is correctly formatted for the model, you should explicitly set extra_body={"add_special_tokens": True}.
vllm serve openbmb/MiniCPM4-0.5B
vllm serve models/minicpm05b  # 这个本地仓库包含自定义代码，需要显式允许执行： # 给一个固定对外名

vllm serve /root/autodl-tmp/Exp2/models/minicpm05b \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --served-model-name minicpm05b

Then you can use the chat interface by running the following code:
'''

import openai

client = openai.Client(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="minicpm05b",
    messages=[
        {"role": "user", "content": "Write an article about Artificial Intelligence."},
    ],
    temperature=0.7,
    max_tokens=1024,
    extra_body=dict(add_special_tokens=True),  # Ensures special tokens are added for chat template
    
)

print(response.choices[0].message.content)


stream = client.chat.completions.create(
    model="minicpm05b",
    messages=[{"role":"user","content":"给我三条PyTorch入门建议"}],
    stream=True,
    extra_body={"add_special_tokens": True},
)
for chunk in stream:
    print((chunk.choices[0].delta.content or ""), end="", flush=True)
print()
