import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "/path/to/base_model"                     # 例: /root/.../minicpm05b
adapter_ckpt = "/path/to/checkpoint-10803"       # 你的checkpoint目录（含 adapter_model.safetensors）

tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()

# 叠加 LoRA 适配器
model = PeftModel.from_pretrained(model, adapter_ckpt)  # 会自动找 adapter_model.safetensors
# 如果是QLoRA/4bit部署，可在from_pretrained里加 quantization_config

prompt = "你好，帮我写一段小诗："
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0], skip_special_tokens=True))
