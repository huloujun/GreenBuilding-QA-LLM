from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "outputs/lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True
)

# 加载 LoRA 微调权重
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

def build_prompt(question):
    return (
        "<|im_start|>user\n"
        f"{question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

@torch.no_grad()
def ask(question):
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n🤖 模型回答：", answer)

while True:
    q = input("\n请输入你的问题：")
    ask(q)
