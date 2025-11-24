import gradio as gr
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

tokenizer, model = load_model()

def answer(q):
    prompt = f"用户提问：{q}\n回答："
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3
        )

    ans = tokenizer.decode(output[0], skip_special_tokens=True)
    return ans

interface = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="输入你的建筑问题"),
    outputs=gr.Textbox(label="模型回答"),
    title="绿建问答小模型（Qwen2.5-0.5B + LoRA）",
    description="一个通过少量数据微调得到的建筑领域专业问答模型"
)

if __name__ == "__main__":
    interface.launch()
