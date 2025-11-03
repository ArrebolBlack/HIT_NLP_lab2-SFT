from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

# path = 'openbmb/MiniCPM4-0.5B'
path = 'models/minicpm05b'
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

# User can directly use the chat interface
responds, history = model.chat(tokenizer, "Write an article about Artificial Intelligence.", temperature=0.7, top_p=0.7)
print(responds)

# User can also use the generate interface
# messages = [
#     {"role": "user", "content": "Write an article about Artificial Intelligence."},
# ]
# prompt_text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
# )
# model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

# model_outputs = model.generate(
#     **model_inputs,
#     max_new_tokens=1024,
#     top_p=0.7,
#     temperature=0.7
# )
# output_token_ids = [
#     model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs['input_ids']))
# ]

# responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
# print(responses)



from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch, threading

path = "models/minicpm05b"  # 或 "openbmb/MiniCPM4-0.5B"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    device_map="auto" if torch.cuda.is_available() else None
).eval()

messages = [{"role":"user","content":"用要点说明 SFT 与 DPO 的区别。"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok([prompt], return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k:v.to(model.device) for k,v in inputs.items()}

streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
gen_kwargs = dict(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, streamer=streamer)

thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()
print(">> ", end="", flush=True)
for text in streamer:
    print(text, end="", flush=True)
print()
