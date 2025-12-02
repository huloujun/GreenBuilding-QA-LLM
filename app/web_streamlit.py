import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-0.5B"        # 和你训练时一致
LORA_PATH = "./output/checkpoint-final" # 你的 LoRA 路径（按你的实际情况修改）


# --------------------------
# 强制使用 slow tokenizer（兼容旧模型）
# --------------------------
def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,   # 🔥 禁用 fast tokenizer
        trust_remote_code=True
    )
    return tokenizer


# --------------------------
# 兼容旧版 Qwen2 + LoRA 的模型加载
# --------------------------
def load_model():
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载 LoRA，兼容旧 peft==0.4 - 0.6
    model = PeftModel.from_pretrained(
        base,
        LORA_PATH,
        torch_dtype=torch.float16,
    ).eval()

    return model


# --------------------------
# Streamlit APP
# --------------------------
st.title("绿色建筑问答机器人（Streamlit | 旧模型兼容版）")

@st.cache_resource
def load_all():
    tok = load_tokenizer(BASE_MODEL)
    model = load_model()
    return tok, model


with st.spinner("正在加载模型...（首次加载需要 20 秒左右）"):
    tokenizer, model = load_all()


# --------------------------
# 聊天逻辑
# --------------------------
def chat(query):
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# --------------------------
# UI
# --------------------------
user_input = st.text_area("请输入你的问题：")

if st.button("发送"):
    if user_input.strip():
        with st.spinner("正在生成回答..."):
            answer = chat(user_input)
        st.success(answer)


