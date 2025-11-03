# HIT NLP Course 2025 Autumn - Experiment 2: LLM SFT with Local Data

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

哈尔滨工业大学自然语言处理课程2025秋季学期实验二：构建本地数据并对大语言模型进行监督微调（SFT）。

## 📋 项目概述

本项目构建了一个完整的LLM监督微调pipeline，包含数据生成、预处理、模型训练和推理全流程。通过结合本地生成的领域特定数据和开源数据集，对MiniCPM4基座模型进行监督微调，探索SFT对模型性能的提升效果。

**核心特性**：
- 🔄 完整的SFT训练pipeline
- 📊 混合数据集策略（本地生成 + 开源数据）
- 🚀 支持多种训练方式（Full、LoRA、QLoRA）
- 🤖 多框架推理支持（Transformers、vLLM、SGLang）
- ⚡ 高效的异步数据生成

## 🚀 快速开始

### 环境配置
```bash
pip install -r requirements.txt
```

### 数据与模型下载
- **模型、代码、数据、实验报告**: [百度网盘](https://pan.baidu.com/s/1DheDYLMr1WUl1tcN_gpKuw?pwd=hr3k)
- **基座模型**: [MiniCPM4](https://github.com/OpenBMB/MiniCPM)

## 📊 数据集

### 1. 本地生成数据 (HIT Dataset)
使用LLM生成高质量的SFT数据：
- **数据生成流程**:
  1. 使用GPT-5联网搜索生成相关主题，写入`topics.txt`
  2. 异步调用DeepSeek模型生成SFT数据

### 2. 开源数据集
从[开源SFT数据集整理](https://github.com/chaoswork/sft_datasets)中选取以下数据
- [firefly_no_belle.json](https://github.com/chaoswork/sft_datasets)
- [Alpaca-CoT/firefly_no_belle](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/blob/main/firefly/firefly_no_belle.json)

## 🛠️ 数据生成

### API配置
支持三种灵活的API配置方式：

```bash
# 1. 环境变量（推荐）
export OPENAI_API_KEY=你的key
export OPENAI_BASE_URL=https://api.deepseek.com

# 2. 参数传入
--api-key 你的key --base-url https://api.deepseek.com

# 3. 代码内修改（懒人必备）
# 修改 gen_SFT_data.py 文件开头
```

### 数据生成命令

**DeepSeek-Chat模型（推荐）**:
```bash
python gen_SFT_data.py \
  --model deepseek-chat \
  --mode chat \
  --out data/raw/hit/train.jsonl \
  --n-per-topic 6 \
  --max-concurrency 50
```

**DeepSeek-Reasoner模型**:
```bash
python gen_SFT_data.py \
  --model deepseek-reasoner \
  --mode reasoner \
  --out data/raw/hit/train.jsonl \
  --n-per-topic 6 \
  --max-concurrency 50
```

### ChatML格式
```xml
<|beginofutterance|>系统
（Instruction）
<|endofutterance|>
<|beginofutterance|>用户
（Question）
<|endofutterance|>
<|beginofutterance|>智能助手
（Answer）
<|endofutterance|>
```

## 🔧 数据预处理

### 主要处理步骤
- 格式转换：转换为ChatML格式
- 数据清洗与去重
- 数据集合并与采样
- 格式转换：转换为Firefly训练所需的conversation格式

**注意**：加权采样功能当前未完全实现，需要混合数据集调整采样的用户可以自行修改相关代码。当前版本保证清洗去重和直接合并功能稳定。

### 预处理命令
经过充分测试的配置（清洗 + 去重 + 合并）：
```bash
python prep_sft_firefly_hit.py \
  --firefly-json /path/to/firefly_no_belle.json \
  --hit-jsonl /path/to/hit_ds_chat.jsonl \
  --out /path/to/output.jsonl \
  --total 300000 \
  --hit-ratio 0.4 \
  --max-firefly 250000 \
  --min-chinese-ratio 0.65 \
  --near-dup-threshold 3 \
  --drop-pii \
  --log INFO
```
**典型输出**：约171,739条firefly数据和1,100条hit数据，建议根据实际需求调整数据配方。

### 格式转换
转换为Firefly框架SFT训练所需的conversation格式：
```bash
python chatml2conversation.py \
  --in /path/to/input.jsonl \
  --out /path/to/output.jsonl \
  --default-category "Brainstorming" \
  --start-id 1
```

## 🎯 模型训练

使用[Firefly训练框架](https://github.com/yangjianxin1/Firefly)，OpenBMB的开源工程师在PR中提供了详细的MiniCPM4配置。

[Firefly](https://github.com/yangjianxin1/Firefly)有非常详细的中文的**参数说明**以及**训练示例**，建议参考

### 训练方法对比
| 方法 | 显存占用 | 训练速度 | 推荐度 |
|------|----------|----------|--------|
| Full Fine-tuning | 高 | 慢 | ⭐⭐ |
| LoRA | 中 | 中 | ⭐⭐⭐ |
| **QLoRA** | **低** | **快** | **⭐⭐⭐⭐⭐** |

### QLoRA训练步骤（LoRA，Full同理，注意Full可开启deepspeed，命令头加入deepspeed --num_gpus={num_gpus} ）
1. **配置修改**:
   ```bash
   # 编辑配置文件
   vim Firefly/train_args/sft/qlora/minicpm4-0.5b-sft-qlora.json
   ```

2. **开始训练**:
   ```bash
   cd Firefly
   python train.py --train_args_file train_args/sft/qlora/minicpm4-0.5b-sft-qlora.json
   ```

### 其他训练方式
1. **LoRA**: 类似QLoRA，修改对应配置文件
2. **全量参数(Full)**: 可开启DeepSpeed优化，num_gpus替换为显卡数量
   ```bash
   deepspeed --num_gpus={num_gpus} train.py --train_args_file train_args/sft/full/bloom-1b1-sft-full.json
   ```

### 二阶段训练
本项目采用二阶段训练策略：
- **第一阶段**: 使用大规模通用数据（firefly_no_belle）建立基础能力
- **第二阶段**: 使用小规模私有数据（HIT数据）进行领域风格强化
如需单阶段混合训练，请相应调整数据配方比例。

### 模型合并
QLoRA/LoRA训练完成后需要合并权重，便于下一阶段训练：
```bash
python Firefly/script/merge_lora.py
```

## 🔍 模型推理

基于[MiniCPM4官方实现](https://github.com/OpenBMB/MiniCPM)，复现了三种主流推理框架：

### 快速测试
```bash
python quick_infer_local.py
```

### Transformers推理（兼容性好）
```bash
python inference_with_transformers.py
```

### vLLM推理（高吞吐量）
```bash
python inference_with_vLLM.py
```

### SGLang推理（优化推理流程）
```bash
python inference_with_SGLang.py
```

## 📁 项目结构
```
├── LLM_data_gen/           # 数据生成脚本
├── data/
│   ├── raw/               # 原始数据
│   └── processed/         # 处理后的数据
├── scripts/               # 训练和推理脚本
├── configs/               # 训练配置文件
└── models/                # 训练好的模型
```

## ⚠️ 注意事项

1. **数据采样**: 当前版本加权采样功能尚未实现，需要自定义采样策略的用户请修改prep_sft_firefly_hit.py中的相关逻辑
2. **API限制**: 数据生成时注意API调用频率限制，合理设置并发数，max_concurrency=50在实际测试中表现稳定
3. **硬件要求**: 
  - QLoRA训练MiniCPM4-0.5B：最低2.2GB显存
  - 实验环境：16核CPU + 单卡RTX 4090
  - 训练时间：QLoRA约4小时完成训练
4. **模型合并**: QLoRA/LoRA训练完成后请使用`merge_lora.py`合并模型, 方便第二阶段训练和推理


## 📈 实验结果

详细实验结果和分析请参考实验报告（百度网盘中提供）。

## 🙏 致谢

- [MiniCPM](https://github.com/OpenBMB/MiniCPM) - 优秀的开源基座模型型
- [Firefly](https://github.com/yangjianxin1/Firefly) - 易用高效的训练框架
- 我的钱包 - 为API调用和算力提供资金支持

## 📄 许可证

本项目基于MIT许可证开源。
