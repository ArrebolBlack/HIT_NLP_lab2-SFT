# -*- coding: utf-8 -*-
"""
异步批量生成“哈工大”相关中文 SFT 数据（按 ChatML 模板落盘为 JSONL 的 text 字段）
- 支持两类模型：chat（deepseek-chat 等）与 reasoner（deepseek-reasoner）
- 通过统一的 LLMCaller 封装，便于切换与复用
- 并发、重试、去重、质量校验、断点续跑
"""
"""
✅ 如何调用（两种模型随时切换）
1) 使用 chat（如 deepseek-chat）
export OPENAI_API_KEY=你的key
export OPENAI_BASE_URL=https://api.deepseek.com

python gen_SFT_data.py \
  --model deepseek-chat \
  --mode chat \
  --out data/raw/hit/train.jsonl \
  --n-per-topic 6 \
  --max-concurrency 50

2) 使用 reasoner（deepseek-reasoner）
export OPENAI_API_KEY=你的key
export OPENAI_BASE_URL=https://api.deepseek.com

python gen_SFT_data.py \
  --model deepseek-reasoner \
  --mode reasoner \
  --out data/raw/hit/train.jsonl \
  --n-per-topic 6 \
  --max-concurrency 50

deepseek-reasoner 的 思维链会在 reasoning_content 字段返回（代码里已记录到 debug 日志），
但落盘仍只用最终回答构建 ChatML 文本，满足你们的训练格式一致性。
"""

import os
import re
import sys
import json
import time
import hashlib
import random
import logging
import argparse
import asyncio
from typing import List, Dict, Any, Optional

from tqdm import tqdm
from openai import AsyncOpenAI

# =========================================================
# ==== CONFIG（可选，直接在这里写死；留空则不覆盖）====
# =========================================================
# 🔒 建议仅在个人环境使用，避免把密钥提交到仓库。
API_KEY_CODE = "sk-e3ae81b250c74be690f85e02a5a2a7b9"  # 例如 "sk-xxxxxxxxxx"; 留空则不用 # gen_SFT
BASE_URL_CODE = "https://api.deepseek.com"  # 例如 "https://api.deepseek.com"; 留空则不用

# 其它运行参数（留空/None则不覆盖命令行）
MODEL_CODE = "deepseek-chat"            # 例如 "deepseek-chat" 或 "deepseek-reasoner"
MODE_CODE = "chat"             # "chat" / "reasoner"
OUT_CODE = "E:/PE_Exam_2025_Autumn/code/gen_SFT_output/generated_data.jsonl"              # 例如 "data/raw/hit/train.jsonl"
TOPICS_FILE_CODE =  "E:/PE_Exam_2025_Autumn/code/gen_SFT_output/topics.txt"    # 例如 "topics.txt"
N_PER_TOPIC_CODE = 6      # 例如 6
MAX_CONCURRENCY_CODE = 50  # 例如 8
TEMPERATURE_CODE = 0.8      # 例如 0.8
# =========================================================

# =========================
# 常量与模板
# =========================

DEFAULT_TOPICS = [
    "哈工大计算机专业报考与准备建议",
    "哈工大人工智能专业课程体系与学习路径",
    "哈工大机械工程/自动化专业选择对比",
    "哈工大大一选课策略与GPA管理",
    "哈工大常见挂科科目与避免方法",
    "哈工大人工智能/机器人实验室申请流程与建议",
    "本科生参与科研（SRTP/大创/科创）指南",
    "哈工大机器人竞赛/电赛/数模如何准备",
    "保研路径与考核要点",
    "哈工大学生互联网实习/校招准备",
    "申请英美高校硕士/博士的建议与材料",
    "南区/一校区/二校区生活指南与食堂推荐",
    "哈尔滨冬季御寒与学习效率建议",
    "社团与时间管理平衡"
]

CHATML_BEGIN = "<|beginofutterance|>"
CHATML_END = "<|endofutterance|>"
SYSTEM_ROLE = "系统"
USER_ROLE = "用户"
ASSISTANT_ROLE = "智能助手"

def build_chatml_text(instruction: str, question: str, answer: str) -> str:
    return (
        f"{CHATML_BEGIN}{SYSTEM_ROLE}\n{instruction}\n{CHATML_END}\n"
        f"{CHATML_BEGIN}{USER_ROLE}\n{question}\n{CHATML_END}\n"
        f"{CHATML_BEGIN}{ASSISTANT_ROLE}\n{answer}\n{CHATML_END}"
    )

GEN_TEMPLATE = """你是一名严谨的中文教育数据标注员，负责为“哈工大相关主题”的中文对话微调（SFT）生成高质量样本。

【输出要求】
- **只输出 JSON**（UTF-8，无注释），格式为：一个数组，数组内每个元素是一个对象，字段为：
  - instruction：系统角色对助手的角色设定/任务说明（必须与哈工大主题相关，中文撰写，50~120字）
  - input：用户提问（中文，明确具体，30~120字，尽量结合哈工大真实场景）
  - output：助手回答（中文，专业、可信、结构化，200~500字，分点叙述，避免夸张/虚假信息）

【内容边界】
- 不生成违法、色情、仇恨、隐私内容
- 保持中立与专业，不捏造事实
- 尽可能结合“哈工大”真实语境（课程、实验室、科研、竞赛、生活、地点等）

【生成数量】
- 请为主题《{topic}》一次性生成 {n} 条高质量样本
- 只输出 JSON 数组，不要额外文字
"""

def construct_messages(topic: str, n: int) -> List[Dict[str, str]]:
    sys_prompt = "你是安全、专业的中文教育数据标注助手。"
    user_prompt = GEN_TEMPLATE.format(topic=topic, n=n)
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

# =========================
# .env / 环境加载 / 覆盖工具
# =========================

def load_dotenv(path: str = ".env") -> None:
    """极简 .env 解析（key=value），只设置当前未存在的环境变量。"""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except Exception:
        pass

def load_api_key() -> str:
    """
    API Key 获取优先级：
    1) API_KEY_CODE（代码常量）
    2) 命令行（在 amain 里覆盖）
    3) 环境变量 OPENAI_API_KEY
    4) .env 文件（会在 main 开始就加载）
    """
    if API_KEY_CODE:
        return API_KEY_CODE
    # 命令行在 parse_args 后处理
    env_key = os.getenv("OPENAI_API_KEY", "")
    return env_key

def load_base_url() -> str:
    """
    Base URL 获取优先级：
    1) BASE_URL_CODE（代码常量）
    2) 命令行（在 amain 里覆盖）
    3) 环境变量 OPENAI_BASE_URL 或默认 https://api.deepseek.com
    """
    if BASE_URL_CODE:
        return BASE_URL_CODE
    return os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")

def apply_code_overrides(args):
    """用代码常量覆盖命令行参数（仅当代码常量非空/非None时）"""
    if MODEL_CODE is not None:
        args.model = MODEL_CODE
    if MODE_CODE is not None:
        args.mode = MODE_CODE
    if OUT_CODE is not None:
        args.out = OUT_CODE
    if TOPICS_FILE_CODE is not None:
        args.topics_file = TOPICS_FILE_CODE
    if N_PER_TOPIC_CODE is not None:
        args.n_per_topic = N_PER_TOPIC_CODE
    if MAX_CONCURRENCY_CODE is not None:
        args.max_concurrency = MAX_CONCURRENCY_CODE
    if TEMPERATURE_CODE is not None:
        args.temperature = TEMPERATURE_CODE
    return args

# =========================
# 质量校验 & 去重
# =========================

def is_chinese_ratio_ok(text: str, min_ratio: float = 0.6) -> bool:
    if not text:
        return False
    zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return (zh / max(1, len(text))) >= min_ratio

def length_ok(text: str, min_len: int, max_len: int) -> bool:
    L = len(text.strip())
    return min_len <= L <= max_len

def validate_item(obj: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if not isinstance(obj, dict):
        return None
    for k in ("instruction", "input", "output"):
        if k not in obj or not isinstance(obj[k], str):
            return None
    ins, inp, out = obj["instruction"].strip(), obj["input"].strip(), obj["output"].strip()

    if not (is_chinese_ratio_ok(ins) and is_chinese_ratio_ok(inp) and is_chinese_ratio_ok(out)):
        return None
    if not length_ok(ins, 20, 200): return None
    if not length_ok(inp, 20, 300): return None
    if not length_ok(out, 80, 1200): return None

    bad = [r"仅供参考", r"免责声明", r"抱歉我不能", r"无法回答"]
    if any(re.search(p, out) for p in bad):
        return None

    return {"instruction": ins, "input": inp, "output": out}

# =========================
# 统一模型调用封装（chat / reasoner）
# =========================

class LLMCaller:
    """
    统一调用入口：
    - mode='chat'     → deepseek-chat（或其它兼容聊天模型）
    - mode='reasoner' → deepseek-reasoner（可取 reasoning_content 作日志）
    """

    def __init__(self, api_key: str, base_url: str, model: str, mode: str = "chat"):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        assert mode in ("chat", "reasoner")
        self.mode = mode

    async def generate_json_array(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.8,
        retries: int = 3,
        backoff: float = 1.8,
        timeout_s: int = 90,
    ) -> List[Dict[str, Any]]:
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                resp = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature if self.mode == "chat" else None,  # reasoner忽略采样参数
                    ),
                    timeout=timeout_s
                )
                if self.mode == "reasoner":
                    try:
                        rc = resp.choices[0].message.reasoning_content
                        if rc:
                            logging.debug("[reasoner reasoning_content]\n" + rc[:2000])
                    except Exception:
                        pass

                content = resp.choices[0].message.content
                m = re.search(r"(\[.*\])", content, flags=re.S)
                if not m:
                    raise ValueError("未在模型输出中定位到 JSON 数组")
                data = json.loads(m.group(1))
                if not isinstance(data, list):
                    raise ValueError("提取到的 JSON 不是数组")
                return data
            except Exception as e:
                last_err = e
                sleep_s = (backoff ** (attempt - 1)) * 0.8 + random.random() * 0.2
                await asyncio.sleep(sleep_s)
        raise RuntimeError(f"LLM 调用失败（已重试 {retries} 次）: {last_err}")

# =========================
# 主流程：并发生成 + ChatML 落盘
# =========================

async def generate_for_topics(
    topics: List[str],
    n_per_topic: int,
    caller: LLMCaller,
    out_jsonl: str,
    max_concurrency: int = 6,
    temperature: float = 0.8,
) -> None:
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    # 断点续跑：读取已存在记录
    seen = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                    text = obj.get("text", "")
                    fp = hashlib.md5(text.encode("utf-8")).hexdigest()
                    seen.add(fp)
                except Exception:
                    continue

    lock = asyncio.Lock()
    sem = asyncio.Semaphore(max_concurrency)

    async def worker(topic: str):
        async with sem:
            msgs = construct_messages(topic, n_per_topic)
            data = await caller.generate_json_array(
                messages=msgs,
                temperature=temperature,
                max_tokens=4096,
            )

            out_records = []
            for raw in data:
                item = validate_item(raw)
                if not item:
                    continue
                chatml = build_chatml_text(item["instruction"], item["input"], item["output"])
                fp = hashlib.md5(chatml.encode("utf-8")).hexdigest()
                if fp in seen:
                    continue
                out_records.append({"text": chatml})

            if out_records:
                async with lock:
                    with open(out_jsonl, "a", encoding="utf-8") as f:
                        for obj in out_records:
                            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            seen.add(hashlib.md5(obj["text"].encode("utf-8")).hexdigest())

    tasks = [asyncio.create_task(worker(t)) for t in topics]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
        try:
            await fut
        except Exception as e:
            logging.error(f"[Topic 失败] {e}")

# =========================
# CLI
# =========================

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def parse_args():
    ap = argparse.ArgumentParser(description="异步生成哈工大相关 SFT 数据（JSONL，text=ChatML 模板）")
    ap.add_argument("--api-key", type=str, default="", help="优先级低于代码常量，晚于 .env 和环境变量")
    ap.add_argument("--base-url", type=str, default="", help="优先级低于代码常量，晚于 .env 和环境变量")
    ap.add_argument("--model", type=str, default="deepseek-chat",
                    help="可用 deepseek-chat 或 deepseek-reasoner")
    ap.add_argument("--mode", type=str, default="chat", choices=["chat", "reasoner"],
                    help="chat / reasoner（reasoner 将记录 reasoning_content）")
    ap.add_argument("--out", type=str, default="data/raw/hit/train.jsonl")
    ap.add_argument("--topics-file", type=str, default="")
    ap.add_argument("--n-per-topic", type=int, default=6)
    ap.add_argument("--max-concurrency", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="对 reasoner 不生效，但保留参数兼容")
    return ap.parse_args()

async def amain():
    setup_logging()

    # 1) 先加载 .env（若存在）
    load_dotenv(".env")

    # 2) 解析命令行
    args = parse_args()

    # 3) 应用代码常量覆盖（你在文件头部写的 *_CODE）
    args = apply_code_overrides(args)

    # 4) API Key / Base URL 取值（考虑代码常量、命令行、环境变量、.env）
    api_key = API_KEY_CODE or (args.api_key if args.api_key else load_api_key())
    base_url = BASE_URL_CODE or (args.base_url if args.base_url else load_base_url())

    if not api_key:
        print("请提供 API Key：可在文件头 API_KEY_CODE 写死，或 --api-key 传入，或设置 OPENAI_API_KEY / .env", file=sys.stderr)
        sys.exit(2)

    topics = DEFAULT_TOPICS
    if args.topics_file:
        topics = read_lines(args.topics_file)
        if not topics:
            print(f"topics 文件为空：{args.topics_file}", file=sys.stderr)
            sys.exit(3)

    caller = LLMCaller(
        api_key=api_key,
        base_url=base_url,
        model=args.model,
        mode=args.mode,
    )

    logging.info(f"准备生成：{len(topics)} 个主题，每个 {args.n_per_topic} 条 → {args.out}")
    await generate_for_topics(
        topics=topics,
        n_per_topic=args.n_per_topic,
        caller=caller,
        out_jsonl=args.out,
        max_concurrency=args.max_concurrency,
        temperature=args.temperature,
    )
    logging.info("生成完成")

if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("用户中断。")
