from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="QingyiSi/Alpaca-CoT",
    repo_type="dataset",
    local_dir="./data/SFT/raw/firefly",
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["firefly/*"],   # 只下载 firefly 目录
)
