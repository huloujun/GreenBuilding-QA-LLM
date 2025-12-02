import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = "outputs/lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# ★ 全部使用 CPU
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map={"": "cpu"},
)

model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.to("cpu")     # 再次确保在 CPU
model.eval()


def chat(message, history):
    prompt = f"用户：{message}\n助手："

    # ★ 强制输入张量放在 CPU
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    if "助手：" in answer:
        answer = answer.split("助手：")[-1].strip()

    return answer


with gr.Blocks(title="GreenBuilding-QA") as demo:
    gr.Markdown("# 🌱 GreenBuilding-QA ChatBot（CPU 版本）")

    gr.ChatInterface(fn=chat)

demo.launch(server_name="0.0.0.0", server_port=7860)

