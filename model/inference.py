import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = "lora-qlora-greenbuilding"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()

    return tokenizer, model

def ask(question):
    tokenizer, model = load_model()

    prompt = f"用户提问：{question}\n回答："
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    q = input("请输入您的问题：")
    print("\n模型回答：\n")
    print(ask(q))
