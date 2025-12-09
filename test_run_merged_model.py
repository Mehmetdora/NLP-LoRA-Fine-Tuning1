import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_merged_model(local_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

def build_prompt(problem_text: str) -> str:
    system_prompt = (
        "You are an expert Python programmer. "
        "Please read the problem carefully before writing any Python code."
    )
    text = (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{problem_text.strip()}\n"
        f"<|assistant|>\n"
    )
    return text

def main(best_merged_file_name):
    # merged modelin klasörünün yolu
    local_dir = f'/content/drive/MyDrive/lora-nlp-homework1/checkpoints/{best_merged_file_name}'
    model, tokenizer = load_merged_model(local_dir)

    problem = """
    Given an integer n, write a function that returns True if n is prime,
    and False otherwise.
    """

    prompt = build_prompt(problem)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("=== MODEL CEVABI ===")
    print(generated)

if __name__ == "__main__":
    main(best_merged_file_name = "merged_diverse_best3")
