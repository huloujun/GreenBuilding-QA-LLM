import json
import time
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api.openai.com/v1"  # 或兼容 API
)

PROMPT_PATH = "prompts/A_technical.txt"
OUTPUT_PATH = "data/gb_qa_A_technical.jsonl"

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    base_prompt = f.read()

def generate_batch(batch_id):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 或其他
        messages=[
            {"role": "system", "content": "你是一个数据生成引擎"},
            {"role": "user", "content": base_prompt}
        ],
        temperature=0.8
    )
    return response.choices[0].message.content

for i in range(10):  # 10 次 × 200 条 = 2000 条
    print(f"Generating batch {i+1}...")
    text = generate_batch(i)

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for line in text.splitlines():
            if line.strip().startswith("{"):
                f.write(line.strip() + "\n")

    time.sleep(5)  # 防止触发限流
