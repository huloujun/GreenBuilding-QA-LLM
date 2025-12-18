import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "./outputs/checkpoint-384"   # 你已经确认可用的 LoRA 路径


# --------------------------
# Tokenizer（保持与你终端脚本一致）
# --------------------------
def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True
    )
    # 关键：补 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# --------------------------
# Model + LoRA
# --------------------------
def load_model():
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(
        base,
        LORA_PATH,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model


# --------------------------
# Streamlit 缓存加载
# --------------------------
st.title("绿色建筑问答机器人（Qwen2.5-7B-Instruct + LoRA）")

@st.cache_resource
def load_all():
    tokenizer = load_tokenizer(BASE_MODEL)
    model = load_model()
    return tokenizer, model


with st.spinner("正在加载模型（首次需要较长时间）..."):
    tokenizer, model = load_all()


# --------------------------
# Prompt（关键：Instruct 对齐）
# --------------------------
def build_prompt(query: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{query}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# --------------------------
# 推理函数（等价于你终端能跑的版本）
# --------------------------
@torch.no_grad()
def chat(query):
    prompt = build_prompt(query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    # 只返回 assistant 的内容
    return text.split("assistant")[-1].strip()


# --------------------------
# UI
# --------------------------
user_input = st.text_area("请输入你的问题：")

if st.button("发送"):
    if user_input.strip():
        with st.spinner("正在生成回答..."):
            answer = chat(user_input)
        st.success(answer)


