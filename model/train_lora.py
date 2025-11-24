import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# ----------------------
# 1. 基本参数
# ----------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "data/gb_qa.jsonl"
OUTPUT_DIR = "lora-qlora-greenbuilding"
MAX_LENGTH = 512

# ----------------------
# 2. 数据加载
# ----------------------
print("正在加载数据……")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# ----------------------
# 3. 加载 tokenizer 和模型
# ----------------------
print("正在加载模型……")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# ----------------------
# 4. LoRA 配置
# ----------------------
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("LoRA 参数添加完毕")

# ----------------------
# 5. 数据处理函数
# ----------------------
def format_example(example):
    instruction = example["instruction"]
    output = example["output"]
    text = f"用户提问：{instruction}\n回答：{output}"
    return tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

train_dataset = dataset.map(format_example)

# ----------------------
# 6. 训练参数
# ----------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    warmup_ratio=0.05,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# ----------------------
# 7. Trainer 开始训练
# ----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

trainer.train()

# ----------------------
# 8. 保存 LoRA 模型
# ----------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n🎉 LoRA 微调完成！模型已保存到：", OUTPUT_DIR)
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# ----------------------
# 1. 基本参数
# ----------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "data/gb_qa.jsonl"
OUTPUT_DIR = "lora-qlora-greenbuilding"
MAX_LENGTH = 512

# ----------------------
# 2. 数据加载
# ----------------------
print("正在加载数据……")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# ----------------------
# 3. 加载 tokenizer 和模型
# ----------------------
print("正在加载模型……")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# ----------------------
# 4. LoRA 配置
# ----------------------
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("LoRA 参数添加完毕")

# ----------------------
# 5. 数据处理函数
# ----------------------
def format_example(example):
    instruction = example["instruction"]
    output = example["output"]
    text = f"用户提问：{instruction}\n回答：{output}"
    return tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

train_dataset = dataset.map(format_example)

# ----------------------
# 6. 训练参数
# ----------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    warmup_ratio=0.05,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# ----------------------
# 7. Trainer 开始训练
# ----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

trainer.train()

# ----------------------
# 8. 保存 LoRA 模型
# ----------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n🎉 LoRA 微调完成！模型已保存到：", OUTPUT_DIR)
