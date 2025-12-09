# push_merged_to_hf.py

import os
import config
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(best_merged_file_name):
    # Örneğin deep için merge edilen klasör
    merged_dir = os.path.join(
        config.CHECKPOINT_ROOT,
        best_merged_file_name
    )

    # repo ismini değiştir, kullanıcı adını kontrol et
    repo_name = "MehmetDORA/qwen2.5-coder-1.5b-diverse-lora-merged-deneme3" 

    model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        merged_dir,
        trust_remote_code=True,
    )

    print("Model HuggingFace Hub'a yükleniyor:", repo_name)
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    print("✅ Yükleme tamamlandı.")

if __name__ == "__main__":

    # ismini değiştir
    main(best_merged_file_name = "merged_diverse_best3")
