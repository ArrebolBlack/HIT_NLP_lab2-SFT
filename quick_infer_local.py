import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 本地目录或 HF 仓库名
MODEL_PATH = "models/minicpm05b"   # 或 "openbmb/MiniCPM4-0.5B"

def pick_dtype():
    if not torch.cuda.is_available():
        return None
    # 优先 bfloat16，若不支持再用 float16
    major, minor = torch.cuda.get_device_capability()
    if major >= 8:   # Ampere+ 一般OK
        return torch.bfloat16
    return torch.float16

def main():
    print("CUDA available:", torch.cuda.is_available())

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,   # MiniCPM 需要
        use_fast=False            # 保守：有些自定义tokenizer prefer slow
    )

    dtype = pick_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,                 # CPU 时为 None
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    # 准备一条对话（必须走 chat template）
    messages = [
        {"role": "user", "content": "用三句话解释什么是自注意力（Self-Attention）。"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # 让模型接着生成assistant
    )

    inputs = tokenizer([prompt], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05
        )

    # 只解码新增部分
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print("\n=== Model Output ===\n", text.strip())

if __name__ == "__main__":
    main()
