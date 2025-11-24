import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = "outputs/lora"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# 加载基础模型（4bit）
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 加载 LoRA 微调权重
model = PeftModel.from_pretrained(base_model, LORA_PATH)

def chatbot(question, history):
    prompt = f"用户：{question}\n助手："
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.3,
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer

# Gradio 界面
with gr.Blocks(title="GreenBuilding-QA-LLM") as demo:
    gr.Markdown("# 🌱 GreenBuilding-QA 智能问答系统\n基于 Qwen2.5-0.5B + QLoRA 微调")

    chat_interface = gr.ChatInterface(chatbot)

demo.launch(server_name="0.0.0.0", server_port=7860)
