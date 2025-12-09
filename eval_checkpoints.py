# eval_checkpoints.py

import os
import torch
import config
from datasets import Dataset
from data_utils import get_test_dataset, formatting_func
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel

def get_base_model_and_tokenizer():
    print(f"Base model yükleniyor → {config.MODEL_NAME}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False

    return base_model, tokenizer

def prepare_test_ds(dataset_kind: str, tokenizer, max_test_samples=None):
    test_ds = get_test_dataset(dataset_kind=dataset_kind, max_test_samples=max_test_samples)

    def tokenize(example):
        text = formatting_func(example)
        return tokenizer(
            text,
            truncation=True,
            max_length=config.MAX_SEQ_LEN,
            padding=False,
        )

    test_tokenized = test_ds.map(tokenize, remove_columns=test_ds.column_names)
    return test_tokenized

def evaluate_checkpoint(ckpt_dir: str, dataset_kind: str, test_ds: Dataset):
    print(f"\n▶ Checkpoint değerlendiriliyor: {ckpt_dir}")

    base_model, tokenizer = get_base_model_and_tokenizer()

    # LoRA adapter'ını base model'e yükle
    model = PeftModel.from_pretrained(base_model, ckpt_dir)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    eval_args = TrainingArguments(
        output_dir=os.path.join(ckpt_dir, "eval_tmp"),
        per_device_eval_batch_size=config.BATCH_SIZE,
        dataloader_num_workers=2,
        bf16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    metrics = trainer.evaluate()
    print(f"Eval metrics: {metrics}")
    return metrics

def main(run_name , dataset_kind: str = "deep", max_test_samples=None):
    # 1) Test dataset'i hazırla (bir kez tokenize etmek yeter)
    base_model, tokenizer = get_base_model_and_tokenizer()
    test_ds = prepare_test_ds(dataset_kind, tokenizer, max_test_samples=max_test_samples)

    # 2) Checkpoint klasörlerini bul
    ckpt_root = os.path.join(config.CHECKPOINT_ROOT, run_name) # son yapılan eğitime ait checkpoint'lerin yolu
    checkpoint_dirs = [
        os.path.join(ckpt_root, d)
        for d in os.listdir(ckpt_root)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(ckpt_root, d))
    ]
    checkpoint_dirs = sorted(
        checkpoint_dirs,
        key=lambda p: int(p.split("-")[-1])  # checkpoint-400 → 400'a göre sırala
    )

    print("Bulunan checkpoint'ler:")
    for d in checkpoint_dirs:
        print(" -", d)

    best_ckpt = None
    best_loss = float("inf")
    all_results = {}

    # 3) Her checkpoint için evaluate
    for ckpt_dir in checkpoint_dirs:
        metrics = evaluate_checkpoint(ckpt_dir, dataset_kind, test_ds)
        eval_loss = metrics.get("eval_loss", None)
        all_results[ckpt_dir] = metrics

        if eval_loss is not None and eval_loss < best_loss:
            best_loss = eval_loss
            best_ckpt = ckpt_dir

    print("\n==== ÖZET ====")
    for ckpt_dir, metrics in all_results.items():
        print(f"{ckpt_dir} → eval_loss = {metrics.get('eval_loss')}")

    print("\n⭐ En iyi checkpoint:")
    print("PATH:", best_ckpt)
    print("BEST eval_loss:", best_loss)

    # En iyi checkpoint yolunu bir txt'e kaydedilir. 
    best_path_file = os.path.join(ckpt_root, "best_checkpoint.txt")
    with open(best_path_file, "w") as f:
        f.write(best_ckpt)
    print("Best checkpoint path kaydedildi →", best_path_file)

if __name__ == "__main__":

    # run_name -> checkpoints altındaki eğitim dosyasının adı
    # deep run'ı için:
    #main(run_name = "20251208-175019_deep_full_e3_deneme3",dataset_kind="deep", max_test_samples=None)

    # diverse için ayrı bir run'da:
    main(run_name = "20251208-225644_diverse_full_e3_deneme3", dataset_kind="diverse", max_test_samples=None)
