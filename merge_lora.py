# merge_lora.py

import os
import torch
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_best_checkpoint_path(run_name):
    # eval_checkpoints.py'in yazdığı dosya
    checkpoint_path = os.path.join(config.CHECKPOINT_ROOT,run_name)
    best_path_file = os.path.join(checkpoint_path, "best_checkpoint.txt")
    with open(best_path_file, "r") as f:
        best_ckpt = f.read().strip()
    return best_ckpt

def main(run_name,dataset_tag: str = "deep"):
    best_ckpt = load_best_checkpoint_path(run_name)
    print("Merge edilecek en iyi checkpoint:", best_ckpt)

    # Base model'i tam prec/yarım prec olarak yükle (4-bit değil!)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, best_ckpt)

    # LoRA ağırlıklarını base model ile birleştir
    print("LoRA adapter merge ediliyor...")
    merged_model = model.merge_and_unload()

    # Kaydedilecek klasör
    merged_dir = os.path.join(
        config.CHECKPOINT_ROOT,
        f"merged_{dataset_tag}_best3"
    )
    os.makedirs(merged_dir, exist_ok=True)

    print("Merged model kaydediliyor →", merged_dir)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print("✅ Merge tamamlandı.")

if __name__ == "__main__":
    # deep eğitimi için:
    #main(run_name = "20251208-175019_deep_full_e3_deneme3" ,dataset_tag="deep")

    # diverse için ayrı run'da:
    main(run_name = "20251208-225644_diverse_full_e3_deneme3" ,dataset_tag="diverse")
