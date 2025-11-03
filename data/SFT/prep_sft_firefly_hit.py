# -*- coding: utf-8 -*-
"""
prep_sft_firefly_hit.py

功能：
1) 读取 Firefly 子集 `firefly_no_belle.json`（大 JSON 数组或 JSONL），转为 {"text": "...ChatML..."} 的 JSONL
2) 基于中文教育 SFT 需求进行清洗（中文比例/长度/拒答或广告等）、并与本地 HIT SFT(已是ChatML JSONL)合并去重
3) 按权重做混合采样（例如 HIT:Firefly = 3:2 或 1:1），最终导出单一 JSONL（字段仅 text）

用法示例：
    python prep_sft_firefly_hit.py \
      --firefly-json /path/to/firefly/firefly_no_belle.json \
      --hit-jsonl data/SFT/hit/train.jsonl \
      --out data/SFT/merged/hit_plus_firefly.jsonl \
      --total 300000 \
      --hit-ratio 0.4 \
      --min-chinese-ratio 0.6 \
      --near-dup-threshold 3 \
      --max-firefly 250000 \
      --seed 42 \
      --log INFO
"""

import os
import re
import sys
import json
import math
import gzip
import random
import argparse
import logging
import unicodedata
from typing import Dict, Any, Iterable, List, Tuple, Optional

# ========== ChatML 模板 ==========
CHATML_BEGIN = "<|beginofutterance|>"
CHATML_END = "<|endofutterance|>"
SYSTEM_ROLE = "系统"
USER_ROLE = "用户"
ASSISTANT_ROLE = "智能助手"

def build_chatml_text(instruction: str, question: str, answer: str) -> str:
    return (
        f"{CHATML_BEGIN}{SYSTEM_ROLE}\n{instruction.strip()}\n{CHATML_END}\n"
        f"{CHATML_BEGIN}{USER_ROLE}\n{question.strip()}\n{CHATML_END}\n"
        f"{CHATML_BEGIN}{ASSISTANT_ROLE}\n{answer.strip()}\n{CHATML_END}"
    )

# ========== 读取工具（本地 JSON 或 JSONL；可选 ijson 流式解析） ==========

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def iter_json_array(path: str, stream: bool = False) -> Iterable[Dict[str, Any]]:
    """读取一个包含大数组的 .json 文件；stream=True 时用 ijson 流式解析（需 pip install ijson）。"""
    if path.endswith(".jsonl"):
        yield from iter_jsonl(path)
        return
    if not stream:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        yield obj
            else:
                raise ValueError("JSON 文件不是数组，请检查格式或使用 --stream-json 以 ijson 流式解析大文件。")
    else:
        try:
            import ijson  # optional
        except Exception:
            raise RuntimeError("需要流式解析大文件时，请先: pip install ijson")
        with open(path, "rb") as f:
            for obj in ijson.items(f, "item"):
                if isinstance(obj, dict):
                    yield obj

# ========== 规范化 & 质量过滤 & 指纹 ==========
ZH_RANGE = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)]  # 常见 CJK 范围

def to_halfwidth(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = to_halfwidth(s)
    s = s.replace("\u200b", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def zh_ratio(s: str) -> float:
    if not s:
        return 0.0
    total = len(s)
    zh = 0
    for ch in s:
        cp = ord(ch)
        if any(a <= cp <= b for a, b in ZH_RANGE):
            zh += 1
    return zh / max(1, total)

DEFAULT_BAD_PATTERNS = [
    r"抱歉.{0,6}不能", r"抱歉.{0,6}无权", r"抱歉.{0,6}不便",
    r"无法(回答|提供|帮助|处理)", r"不提供(违法|侵权|医疗|法律)建议",
    r"请咨询.*(专业人士|医生|律师)", r"作为.?AI", r"作为一个AI",
    r"不能代写", r"不适合回答", r"违法|违规|侵权", r"成人内容|色情",
    r"Only for educational", r"for educational purposes only", r"I cannot assist",
]

def contains_too_many_urls(s: str, max_urls: int = 2) -> bool:
    urls = re.findall(r"https?://\S+", s)
    return len(urls) > max_urls

def contains_pii(s: str) -> bool:
    # 轻量级示例：手机号/邮箱/身份证号样式
    if re.search(r"\b1[3-9]\d{9}\b", s):  # 简化版中国手机号
        return True
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", s):
        return True
    if re.search(r"\b\d{17}[\dXx]|\b\d{15}\b", s):  # 身份证号模式(简化)
        return True
    return False

def length_ok(s: str, min_len: int, max_len: int) -> bool:
    L = len(s)
    return (L >= min_len) and (L <= max_len)

def md5_64(text: str) -> int:
    h = json.dumps(text, ensure_ascii=False).encode("utf-8")
    return int.from_bytes(__import__("hashlib").md5(h).digest()[:8], "big")

def validate_firefly_item(obj: Dict[str, Any],
                          min_zh_ratio: float,
                          ins_min: int, ins_max: int,
                          inp_min: int, inp_max: int,
                          out_min: int, out_max: int,
                          drop_regexes: List[re.Pattern],
                          drop_url_max: int,
                          drop_pii: bool) -> Optional[str]:
    """校验并返回 ChatML 文本；允许 input 为空字符串。"""
    ins = normalize_text(obj.get("instruction", ""))
    inp = normalize_text(obj.get("input", ""))  # 允许为空
    out = normalize_text(obj.get("output", ""))

    # 基本非空：ins/out 必须有，inp 可空
    if not ins or not out:
        return None

    # 中文率：对 ins/out 强校验；inp 只有在非空时才检查
    if zh_ratio(ins) < min_zh_ratio or zh_ratio(out) < min_zh_ratio:
        return None
    if inp and zh_ratio(inp) < min_zh_ratio:
        return None

    # 长度阈值：inp 为空则跳过长度检查
    if not length_ok(ins, ins_min, ins_max):
        return None
    if inp and (not length_ok(inp, inp_min, inp_max)):
        return None
    if not length_ok(out, out_min, out_max):
        return None

    # 其它质量过滤
    if any(r.search(out) for r in drop_regexes):
        return None
    if contains_too_many_urls(out, drop_url_max):
        return None
    if drop_pii and (contains_pii(ins) or (inp and contains_pii(inp)) or contains_pii(out)):
        return None

    return build_chatml_text(ins, inp, out)

def validate_hit_chatml_text(text: str,
                             min_zh_ratio: float,
                             ins_min: int, ins_max: int,
                             inp_min: int, inp_max: int,
                             out_min: int, out_max: int,
                             drop_regexes: List[re.Pattern],
                             drop_url_max: int,
                             drop_pii: bool) -> Optional[str]:
    """你的 HIT JSONL 已是 {"text": ChatML}，这里做反解析检查再保留/过滤。"""
    if not isinstance(text, str) or CHATML_BEGIN not in text:
        return None
    # 简单拆段
    parts = re.split(rf"{re.escape(CHATML_BEGIN)}(?:{re.escape(SYSTEM_ROLE)}|{re.escape(USER_ROLE)}|{re.escape(ASSISTANT_ROLE)})\n", text)
    # parts[0] == ''，后续每段以消息正文开头，需按顺序 System / User / Assistant
    if len(parts) < 4:
        return None
    # 每段以 ... <|endofutterance|> 结束
    segs = [p.split(CHATML_END)[0].strip() for p in parts[1:4]]
    ins, inp, out = (normalize_text(s) for s in segs)
    if not (zh_ratio(ins) >= min_zh_ratio and zh_ratio(inp) >= min_zh_ratio and zh_ratio(out) >= min_zh_ratio):
        return None
    if not (length_ok(ins, ins_min, ins_max) and length_ok(inp, inp_min, inp_max) and length_ok(out, out_min, out_max)):
        return None
    if any(r.search(out) for r in drop_regexes):
        return None
    if contains_too_many_urls(out, drop_url_max):
        return None
    if drop_pii and (contains_pii(ins) or contains_pii(inp) or contains_pii(out)):
        return None
    # 通过校验，原文即可保留
    return build_chatml_text(ins, inp, out)

# ========== 近似去重（可选，基于简易 SimHash + banding） ==========

def simhash_64(text: str) -> int:
    import hashlib
    tokens = re.findall(r"\w+", text.lower())
    bits = [0] * 64
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        for i in range(64):
            bits[i] += 1 if (h >> i) & 1 else -1
    sig = 0
    for i, v in enumerate(bits):
        if v >= 0:
            sig |= (1 << i)
    return sig

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def near_dup_filter(texts: List[str], threshold_bits: int = 3, band_bits: int = 16, seed: int = 42) -> List[str]:
    """
    简易 LSH：把 64 位 simhash 切成 4 个 band（默认16位×4），任一 band 相同→成为候选，再用汉明距离过滤。
    threshold_bits=3 表示 Hamming<=3 视为近似重复而去除。
    """
    random.seed(seed)
    sigs = [simhash_64(t) for t in texts]
    bands = [{} for _ in range(64 // band_bits)]
    keep = [True] * len(texts)

    for idx, sig in enumerate(sigs):
        if not keep[idx]:
            continue
        candidates = set()
        for b in range(len(bands)):
            mask = ((1 << band_bits) - 1) << (b * band_bits)
            key = (sig & mask, b)
            bucket = bands[b].setdefault(key, [])
            # compare within bucket
            for j in bucket:
                if keep[j] and hamming(sig, sigs[j]) <= threshold_bits:
                    # mark as duplicate (drop current)
                    keep[idx] = False
                    break
            if not keep[idx]:
                break
            bucket.append(idx)
    return [t for t, k in zip(texts, keep) if k]

# ========== 配额分配（加权采样） ==========

def allocate_quota(total:int, sizes:Dict[str,int], weights:Dict[str,float]) -> Dict[str,int]:
    """根据权重分配目标样本数，若某源可用不足则把多余配额按比例回填到其他源。"""
    names = list(sizes.keys())
    wsum = sum(max(0.0, float(weights.get(n, 0.0))) for n in names) or 1.0
    # 初次分配
    alloc = {n: min(sizes[n], int(round(total * float(weights.get(n,0.0)) / wsum))) for n in names}
    # 回填剩余
    deficit = total - sum(alloc.values())
    if deficit > 0:
        # 计算还可再要多少
        remaining = {n: max(0, sizes[n] - alloc[n]) for n in names}
        while deficit > 0 and sum(remaining.values()) > 0:
            for n in names:
                if remaining[n] > 0 and deficit > 0:
                    alloc[n] += 1
                    remaining[n] -= 1
                    deficit -= 1
                if deficit == 0:
                    break
    return alloc

# ========== Firefly 转换 + 清洗 + 去重 ==========

def process_firefly(
    path: str,
    stream_json: bool,
    min_zh_ratio: float,
    ins_min: int, ins_max: int,
    inp_min: int, inp_max: int,
    out_min: int, out_max: int,
    drop_patterns: List[str],
    drop_url_max: int,
    drop_pii: bool,
    global_seen: set,
    max_count: Optional[int] = None,
    no_filter: bool=False,
    no_dedup: bool=False,
) -> List[str]:
    compiled = [re.compile(p) for p in drop_patterns]
    records: List[str] = []
    n_total = n_kept = n_dup = n_filtered = 0

    iterator = iter_json_array(path, stream=stream_json)
    for obj in iterator:
        n_total += 1
        if no_filter:
            # 只做格式转换到 ChatML；允许 input 为空
            ins = normalize_text(obj.get("instruction", "")) or ""
            inp = normalize_text(obj.get("input", "")) or ""
            out = normalize_text(obj.get("output", "")) or ""
            if not ins or not out:
                continue  # 没有最基本的 ins/out 就跳过
            chatml = build_chatml_text(ins, inp, out)
        else:
            chatml = validate_firefly_item(
                obj, min_zh_ratio,
                ins_min, ins_max, inp_min, inp_max, out_min, out_max,
                compiled, drop_url_max, drop_pii
            )
            if not chatml:
                n_filtered += 1
                continue

        # 严格去重：如需完全不去重，交给 --no-dedup 统一关   
        fp = md5_64(chatml)
        if (not no_filter) and (not no_dedup) and (fp in global_major_fingerprints(global_seen)):
            n_dup += 1
            continue
        if not no_dedup:
            global_seen.add(fp)

        records.append(chatml)
        n_kept += 1
        if max_count and n_kept >= max_count:
            break

        if n_total % 100000 == 0:
            logging.info(f"[Firefly] Scanned={n_total}, kept={n_kept}, filtered={n_filtered}, dup={n_dup}")

    logging.info(f"[Firefly] Done: total={n_total}, kept={n_kept}, filtered={n_filtered}, dup={n_dup}")
    return records

# ========== HIT ChatML 读取 + 清洗 + 去重 ==========

def process_hit_chatml_jsonl(
    path: str,
    min_zh_ratio: float,
    ins_min: int, ins_max: int,
    inp_min: int, inp_max: int,
    out_min: int, out_max: int,
    drop_patterns: List[str],
    drop_url_max: int,
    drop_pii: bool,
    global_seen: set,
    max_count: Optional[int] = None,
    no_filter: bool=False,
    no_dedup: bool=False,
) -> List[str]:
    compiled = [re.compile(p) for p in drop_patterns]
    opener = gzip.open if path.endswith(".gz") else open
    records: List[str] = []
    n_total = n_kept = n_dup = n_filtered = 0

    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            n_total += 1 
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text", "")

            if no_filter:
                # 粗拆 ChatML（尽量容错），拿到前三段 System/User/Assistant
                if CHATML_BEGIN not in text:
                    continue
                parts = re.split(
                    rf"{re.escape(CHATML_BEGIN)}(?:{re.escape(SYSTEM_ROLE)}|{re.escape(USER_ROLE)}|{re.escape(ASSISTANT_ROLE)})\n",
                    text
                )
                if len(parts) < 4:
                    continue
                segs = [p.split(CHATML_END)[0].strip() for p in parts[1:4]]
                ins, inp, out = (normalize_text(s) for s in segs)
                if not ins or not out:
                    continue
                text2 = build_chatml_text(ins, inp, out)
            else:
                text2 = validate_hit_chatml_text(
                    text, min_zh_ratio,
                    ins_min, ins_max, inp_min, inp_max,
                    out_min, out_max, compiled, drop_url_max, drop_pii
                )
                if not text2:
                    n_filtered += 1
                    continue

            fp = md5_64(text2)  # <—— 统一用 text2
            if (not no_filter) and (not no_dedup) and (fp in global_major_fingerprints(global_seen)):
                n_dup += 1
                continue
            if not no_dedup:
                global_seen.add(fp)
            records.append(text2)  # <—— 统一追加 text2
            n_kept += 1
            if max_count and n_kept >= max_count:
                break
            if n_total % 50000 == 0 and n_total > 0:
                logging.info(f"[HIT] Scanned≈{n_total}, kept={n_kept}, filtered={n_filtered}, dup={n_dup}")            # increment total after counting to keep logs aligned with file lines
            # but we want n_total to reflect seen lines:
            # (already incremented? no; do it here)
            # -> adjust:
            # Actually easier: increment at top of loop. Move it.
        # (we already incremented at top in Firefly; for HIT we didn't. It's fine.)
    logging.info(f"[HIT] Done: total≈{n_total}, kept={n_kept}, filtered={n_filtered}, dup={n_dup}")
    return records

# Helper to optionally compress fingerprint set to integers (already using 64-bit int)
def global_major_fingerprints(seen_set: set) -> set:
    return seen_set

# ========== 主流程：合并、近似去重（可选）、加权采样、落盘 ==========

def main():
    ap = argparse.ArgumentParser(description="Firefly(no_belle)→ChatML + 清洗去重 + HIT加权混合 → JSONL")
    ap.add_argument("--firefly-json", type=str, required=True, help="本地路径：.../firefly_no_belle.json 或 .jsonl/.jsonl.gz")
    ap.add_argument("--hit-jsonl", type=str, required=True, nargs="+", help="你自制的HIT ChatML JSONL文件，可传多个")
    ap.add_argument("--out", type=str, required=True, help="输出合并后的 JSONL 文件路径（字段仅 text）")
    ap.add_argument("--total", type=int, default=None, help="目标总条数；不设则用全部清洗+去重后的数据")
    ap.add_argument("--hit-ratio", type=float, default=0.5, help="最终混合集中 HIT 数据占比（0~1），默认0.5")
    ap.add_argument("--max-firefly", type=int, default=None, help="可选：对 Firefly 先行截断最大条数，加快处理/近似去重")
    ap.add_argument("--stream-json", action="store_true", help="Firefly 为超大 JSON 数组时启用流式解析(需 pip install ijson)")

    # 质量阈值
    ap.add_argument("--min-chinese-ratio", type=float, default=0.6)
    ap.add_argument("--ins-min", type=int, default=20); ap.add_argument("--ins-max", type=int, default=200)
    ap.add_argument("--inp-min", type=int, default=20); ap.add_argument("--inp-max", type=int, default=300)
    ap.add_argument("--out-min", type=int, default=80); ap.add_argument("--out-max", type=int, default=1200)
    ap.add_argument("--drop-pattern", action="append", default=None,
                    help="额外丢弃的输出正则，可多次传入；默认已内置若干通用拒答/广告词")
    ap.add_argument("--drop-max-urls", type=int, default=5)
    ap.add_argument("--drop-pii", action="store_true", help="启用简单 PII（邮箱/手机号/身份证）过滤")
    # 近似去重
    ap.add_argument("--near-dup-threshold", type=int, default=None,
                    help="可选：SimHash 近似去重阈值(比特位差)，如 3；不传则只做严格去重")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", type=str, default="INFO")

    ap.add_argument("--no-filter", action="store_true",
                help="不做任何质量过滤（中文率/长度/拒答/PII/URL 等全跳过）")
    ap.add_argument("--no-dedup", action="store_true",
                help="不做任何去重（包含跨源严格去重和近似去重）")

    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # 1) 断点续跑；仅当需要去重时才构建 global_seen
    global_seen = set()
    kept_texts_existing: List[str] = []
    if os.path.exists(args.out):
        logging.info(f"检测到已有输出 {args.out}，启用断点续跑（保留已存在样本并继续追加）")
        for obj in iter_jsonl(args.out):
            t = obj.get("text", "")
            if not t:
                continue
            kept_texts_existing.append(t)
            if not args.no_dedup:
                fp = md5_64(t)
                global_seen.add(fp)


    # 2) 读取并清洗 HIT ChatML
    hit_texts_all: List[str] = []
    for p in args.hit_jsonl:
        if not os.path.exists(p):
            logging.warning(f"[HIT] 文件不存在：{p}")
            continue
        cleaned = process_hit_chatml_jsonl(
            path=p,
            min_zh_ratio=args.min_chinese_ratio,
            ins_min=args.ins_min, ins_max=args.ins_max,
            inp_min=args.inp_min, inp_max=args.inp_max,
            out_min=args.out_min, out_max=args.out_max,
            drop_patterns=(args.drop_pattern or []) + DEFAULT_PATTERNS(),
            drop_url_max=args.drop_max_urls,
            drop_pii=args.drop_pii,
            global_seen=global_seen,
            max_count=None,
            no_filter=args.no_filter,
            no_dedup=args.no_dedup,
        )
        hit_texts_all.extend(cleaned)
    logging.info(f"[HIT] 合计保留 {len(hit_texts_all)} 条")

    # 3) 读取并清洗 Firefly no_belle → ChatML
    firefly_texts_all = process_firefly(
        path=args.firefly_json,
        stream_json=args.stream_json,
        min_zh_ratio=args.min_chinese_ratio,
        ins_min=args.ins_min, ins_max=args.ins_max,
        inp_min=args.inp_min, inp_max=args.inp_max,
        out_min=args.out_min, out_max=args.out_max,
        drop_patterns=(args.drop_pattern or []) + DEFAULT_PATTERNS(),
        drop_url_max=args.drop_max_urls,
        drop_pii=args.drop_pii,
        global_seen=global_seen,
        max_count=args.max_firefly,
        no_filter=args.no_filter,
        no_dedup=args.no_dedup,
    )
    logging.info(f"[Firefly] 合计保留 {len(firefly_texts_all)} 条")

    # 4) 近似去重（可选，对“新生成的这两堆”分别做）
    if (not args.no_dedup) and isinstance(args.near_dup_threshold, int) and args.near_dup_threshold >= 0:
        logging.info(f"启动近似去重 (SimHash, Hamming <= {args.near_dup_threshold})")
        before = len(hit_texts_all)
        hit_texts_all = near_dup_filter(hit_texts_all, threshold_bits=args.near_dup_threshold)
        logging.info(f"[HIT] 近似去重: {before} -> {len(hit_texts_all)}")

        before = len(firefly_texts_all)
        firefly_texts_all = near_dup_filter(firefly_texts_all, threshold_bits=args.near_dup_threshold)
        logging.info(f"[Firefly] 近似去重: {before} -> {len(firefly_texts_all)}")

    # 5) 计算混合配额
    sizes = {
        "hit": len(hit_texts_all),
        "firefly": len(firefly_texts_all),
    }
    if args.total is None:
        target_total = sizes["hit"] + sizes["firefly"]
        alloc = {"hit": sizes["hit"], "firefly": sizes["firefly"]}
    else:
        target_total = args.total
        alloc = allocate_quota(
            total=target_total,
            sizes=sizes,
            weights={"hit": args.hit_ratio, "firefly": 1.0 - args.hit_ratio},
        )
    logging.info(f"可用量: {sizes}, 目标总量={target_total}, 分配={alloc}")

    # 6) 抽样（去重后再抽样），并与已存在输出拼接后再进行一次全局去重
    random.seed(args.seed)
    picked_hit = random.sample(hit_texts_all, k=min(alloc["hit"], len(hit_texts_all)))
    picked_firefly = random.sample(firefly_texts_all, k=min(alloc["firefly"], len(firefly_texts_all)))
    mixed = picked_hit + picked_firefly + kept_texts_existing
    random.shuffle(mixed)

    if args.no_dedup:
        final_texts = mixed  # 完全不做最终去重
        logging.info(f"未做最终严格去重：输出 {len(final_texts)} 条")
    else:
        final_seen = set()
        final_texts: List[str] = []
        dup2 = 0
        for t in mixed:
            fp = md5_64(t)
            if fp in final_seen:
                dup2 += 1
                continue
            final_seen.add(fp)
            final_texts.append(t)
        logging.info(f"二次严格去重后: {len(mixed)} -> {len(final_texts)}, 其中二次去重丢弃 {dup2} 条")

    # 若 --total 指定且二次去重后不足，尽力补齐（从剩余池随机补）
    if args.total is not None and len(final_texts) < args.total:
        need = args.total - len(final_texts)
        if args.no_dedup:
            leftovers = hit_texts_all + firefly_texts_all
        else:
            leftovers = [t for t in (hit_texts_all + firefly_texts_all) if md5_64(t) not in final_seen]
        random.shuffle(leftovers)
        extra = leftovers[:need]
        final_texts.extend(extra)
        logging.info(f"补齐 {len(extra)} 条 → 最终 {len(final_texts)}")


    # 7) 落盘（仅字段 text）
    with open(args.out, "w", encoding="utf-8") as f:
        for t in final_texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    logging.info(f"写入完成: {args.out}  共 {len(final_texts)} 条")

    # 8) 输出汇总信息
    info = {
        "sources": {
            "hit_kept": len(hit_texts_all),
            "firefly_kept": len(firefly_texts_all),
        },
        "allocated": alloc,
        "final_count": len(final_texts),
        "hit_ratio_target": args.hit_ratio,
        "firefly_path": args.firefly_json,
        "hit_paths": args.hit_jsonl,
        "near_dup_threshold_bits": args.near_dup_threshold,
        "params": {
            "min_chinese_ratio": args.min_chinese_ratio,
            "length_limits": {
                "instruction": [args.ins_min, args.ins_max],
                "input": [args.inp_min, args.inp_max],
                "output": [args.out_min, args.out_max],
            },
            "drop_max_urls": args.drop_max_urls,
            "drop_pii": args.drop_pii,
        }
    }
    meta_path = os.path.splitext(args.out)[0] + "_dataset_info.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    logging.info(f"统计信息已写入: {meta_path}")

def DEFAULT_PATTERNS() -> List[str]:
    return [
        r"(?i)Only for educational",
        r"(?i)for educational purposes only",
        r"(?i)I cannot assist",
        r"(?i)I (?:can not|cannot) (?:assist|help)",
        r"(?i)as an AI (?:language model)?",
        r"(?i)I do not have (?:access|the ability)",
        r"(?i)I am not a (?:doctor|lawyer|financial advisor)",
        r"(?i)I can(?:’|')t help with that",
        r"敏感.*话题", r"违法|违规|侵权", r"毒品|爆炸物|枪支",
        r"抱歉.*不能", r"抱歉.*无权", r"抱歉.*不便", r"无法(回答|提供|协助)",
        r"仅供参考", r"不提供.*(医疗|法律)建议", r"请咨询.*(专业人士|医生|律师)",
        r"点击链接|关注公众号|扫描二维码|加微信|私信我",
    ]

if __name__ == "__main__":
    main()


'''
python prep_sft_firefly_hit.py \
  --firefly-json /root/autodl-tmp/Exp2/data/SFT/raw/firefly/firefly/firefly_no_belle.json \
  --hit-jsonl /root/autodl-tmp/Exp2/data/SFT/raw/hit/hit_ds_chat.jsonl \
  --out /root/autodl-tmp/Exp2/data/SFT/processed/hit_firefly_no_belle_300k_0.4.jsonl \
  --total 300000 \
  --hit-ratio 0.4 \
  --max-firefly 250000 \
  --min-chinese-ratio 0.65 \
  --near-dup-threshold 3 \
  --drop-pii \
  --log INFO

  '''