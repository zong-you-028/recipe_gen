import torch
from PIL import Image
from transformers import LlamaTokenizer, CLIPImageProcessor
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images

# === 設定模型路徑與硬體 ===
model_path = r"./LLaVA-Chef-Local"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 手動載入 tokenizer，避免錯誤 pad_token 類型 ===
print("🔍 Manually loading tokenizer from:", model_path)
tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.pad_token = tokenizer.unk_token or "[PAD]"
print(f"✅ pad_token set to: {tokenizer.pad_token}")

# === 載入模型與圖片處理器 ===
print("📦 Loading model...")
model = LlavaLlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device).eval()

# === 調整模型 embedding 大小（以支援新增的 special token）===
model.resize_token_embeddings(len(tokenizer))

image_processor = CLIPImageProcessor.from_pretrained(model_path)

# === 載入輸入圖片與食材列表 ===
image_path = r"D:\master_degree\database\Samsung Food\images\000006_1070c6ea3ffef8941498a8f1cd4b0d49f17_image.png"
image = Image.open(image_path).convert("RGB")
ingredients = [
    "ranch dressing", "mozzarella cheese", "cilantro", "avocado",
    "lemon juice", "Lettuce", "tortillas", "hot sauce", "chicken breasts"
]

# === 處理圖片 ===
model_name = get_model_name_from_path(model_path)
image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).to(device)

# === 組裝 prompt ===
ingredients_text = ", ".join(ingredients)
prompt = f"Write a recipe based on the ingredients: {ingredients_text}. Include title, ingredients, and instructions."

# === 加入 image token 到 prompt ===
DEFAULT_IMAGE_TOKEN = "<image>"
image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
prompt_with_img = DEFAULT_IMAGE_TOKEN + "\n" + prompt
print(f"📝 Prompt: {prompt}")

# === 編碼輸入 ===
input_ids = tokenizer_image_token(
    prompt_with_img,
    tokenizer,
    image_token_index=image_token_id,
    return_tensors='pt'
).unsqueeze(0).to(device)

# === 模型推論 ===
print("🚀 Generating...")
with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=512,
    )

# === 解碼輸出 ===
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\n===== 🍳 Generated Recipe =====\n")
print(output_text)
