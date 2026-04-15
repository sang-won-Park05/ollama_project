from pathlib import Path

from train.utils.model_loader import load_model_with_adapter, load_tokenizer

base_model_id = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
adapter_path = "outputs/adapters/exaone-security-lora-m4"
merged_out = "outputs/merged/exaone-security-lora-m4"

Path(merged_out).mkdir(parents=True, exist_ok=True)

print("Loading tokenizer...")
tokenizer = load_tokenizer(
    base_model_id,
    trust_remote_code=True,
)

print("Loading base model + LoRA adapter...")
model = load_model_with_adapter(
    model_name_or_path=base_model_id,
    adapter_path=adapter_path,
    trust_remote_code=True,
    load_in_4bit=False,
    use_bf16=False,
    device_preference="mps",
    dtype_name="float16",
    use_mps=True,
)

print("Merging adapter into base model...")
merged_model = model.merge_and_unload()

print("Saving merged model...")
merged_model.save_pretrained(merged_out)
tokenizer.save_pretrained(merged_out)

print(f"Done. Merged model saved to: {merged_out}")