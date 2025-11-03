# -*- coding: utf-8 -*-
"""
chatml2conversation.py
将 ChatML 模板/三段式(instruction,input,output) 转为:
{
  "conversation_id": <int>,
  "category": "<str>",
  "conversation": [
    {"human": "<str>", "assistant": "<str>"},
    ...
  ]
}

输入每行通常是：
  1) {"text": "<ChatML文本>"}   # ChatML: <|beginofutterance|>系统 ... <|endofutterance|> 等
  2) {"instruction": "...", "input": "...", "output": "..."}  # 三段式

用法示例：
  python chatml2conversation.py \
    --in /root/autodl-tmp/Exp2/data/SFT/processed/hit_firefly_no_belle_300k_0.4.jsonl \
    --out /root/autodl-tmp/Exp2/data/SFT/processed/hit_firefly_no_belle_300k_0.4_conversation.jsonl \
    --default-category "Brainstorming" \
    --start-id 1
"""

import re
import json
import argparse
import sys

CHATML_BEGIN = "<|beginofutterance|>"
CHATML_END   = "<|endofutterance|>"
ROLE_SYSTEM  = "系统"
ROLE_USER    = "用户"
ROLE_ASSIST  = "智能助手"

def parse_chatml(text: str):
    if not isinstance(text, str) or CHATML_BEGIN not in text:
        return None

    pattern = re.compile(
        rf"{re.escape(CHATML_BEGIN)}\s*(?P<role>{re.escape(ROLE_SYSTEM)}|{re.escape(ROLE_USER)}|{re.escape(ROLE_ASSIST)})\s*\n(?P<content>.*?)\s*{re.escape(CHATML_END)}",
        flags=re.S
    )
    roles = pattern.findall(text)

    sys_text, user_text, asst_text = None, None, None
    for role, content in roles:
        c = content.strip()
        if role == ROLE_SYSTEM and sys_text is None:
            sys_text = c
        elif role == ROLE_USER and user_text is None:
            user_text = c
        elif role == ROLE_ASSIST and asst_text is None:
            asst_text = c
    
    # assistant必须
    if not asst_text:
        return None
    
    # user缺失时 fallback到 system（指令式数据）
    if not user_text:
        user_text = sys_text or ""

    return (sys_text or "", user_text, asst_text)

def parse_triplet(obj: dict):
    """
    解析三段式 {"instruction","input","output"} → (instruction, user, assistant)
    """
    if not isinstance(obj, dict):
        return None
    ins = obj.get("instruction")
    inp = obj.get("input")
    out = obj.get("output")
    if not isinstance(inp, str) or not isinstance(out, str):
        return None
    return (ins or "", inp, out)

def make_conversation_record(conv_id: int, category: str, user: str, assistant: str):
    return {
        "conversation_id": conv_id,
        "category": category,
        "conversation": [
            {"human": user, "assistant": assistant}
        ],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="输入 JSONL（每行含 text=ChatML 或 三段式）")
    ap.add_argument("--out", dest="out_path", required=True, help="输出 JSONL（每行为目标结构）")
    ap.add_argument("--default-category", type=str, default="General", help="默认类别字段值")
    ap.add_argument("--start-id", type=int, default=1, help="conversation_id 起始值")
    args = ap.parse_args()

    total, ok, bad = 0, 0, 0
    conv_id = args.start_id

    with open(args.in_path, "r", encoding="utf-8") as fin, \
         open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue

            # 优先走 ChatML
            rec = None
            if isinstance(obj, dict) and "text" in obj:
                parsed = parse_chatml(obj["text"])
                if parsed:
                    ins, user, asst = parsed
                    cat = obj.get("category") or args.default_category
                    rec = make_conversation_record(conv_id, cat, user, asst)

            # 其次尝试三段式
            if rec is None:
                parsed = parse_triplet(obj)
                if parsed:
                    ins, user, asst = parsed
                    cat = obj.get("category") or args.default_category
                    rec = make_conversation_record(conv_id, cat, user, asst)

            if rec is None:
                bad += 1
                continue

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            conv_id += 1
            ok += 1

    print(f"Done. total={total}, converted={ok}, skipped={bad}")
    if ok == 0:
        sys.exit(2)

if __name__ == "__main__":
    main()
