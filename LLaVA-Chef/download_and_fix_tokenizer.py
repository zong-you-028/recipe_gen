import os
import json
from huggingface_hub import hf_hub_download

# === 自定儲存位置 ===
local_dir = "./LLaVA-Chef-Local"
os.makedirs(local_dir, exist_ok=True)

# === 必要檔案 ===
files_to_download = [
    "tokenizer.model",              # LLaMA tokenizer 的核心
    "tokenizer_config.json",        # 包含 pad_token 設定
    "special_tokens_map.json",      # 包含 <image> 等特殊符號
]

# === 下載所有 tokenizer 相關檔案 ===
for fname in files_to_download:
    print(f"📥 Downloading {fname} ...")
    hf_hub_download(
        repo_id="mohbattharani/LLaVA-Chef",
        filename=fname,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

# === 修正 pad_token ===
tokenizer_config_path = os.path.join(local_dir, "tokenizer_config.json")
print("🔧 Fixing tokenizer_config.json ...")

with open(tokenizer_config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

if isinstance(config.get("pad_token"), int):
    config["pad_token"] = "[PAD]"
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("✅ 修正完成 pad_token 為 '[PAD]'")
else:
    print("✅ 不需修正 pad_token")

# === 提示 ===
print("\n📂 請在主程式中設定以下路徑：")
print(f"model_path = r\"{local_dir}\"")
