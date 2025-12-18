import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from transformers import Trainer
from peft import LoraConfig, get_peft_model
import torch

# -------------------------
# 1. 模型与数据路径
# -------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "data/gb_qa.jsonl"

# -------------------------
# 2. 加载数据
# -------------------------
dataset = load_dataset("json", data_files=DATA_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 构建 prompt 模式，适配 instruction + output
def format_example(example):
    return (
        "<|im_start|>user\n"
        f"{example['instruction']}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{example['output']}\n"
        "<|im_end|>"
    )

def tokenize(example):
    text = format_example(example)
    return tokenizer(text, truncation=True, max_length=1024)

tokenized_dataset = dataset.map(tokenize)

# -------------------------
# 3. 加载模型（4bit QLoRA）
# -------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# LoRA 配置
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# -------------------------
# 4. 训练参数
# -------------------------
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_steps=200,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=False,
    bf16=True,  
    optim="paged_adamw_8bit",
    report_to="none",
)

# -------------------------
# 5. Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# -------------------------
# 6. 开始训练
# -------------------------
trainer.train()

# 保存LoRA权重
model.save_pretrained("outputs/lora")
print("🎉 训练完成！LoRA 权重已保存到 outputs/lora")
