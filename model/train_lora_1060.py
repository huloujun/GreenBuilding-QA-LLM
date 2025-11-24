import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from transformers import Trainer
from peft import LoraConfig, get_peft_model
import torch

# -------------------------
# 1. 模型与数据路径
# -------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "data/gb_qa.jsonl"

# -------------------------
# 2. 加载数据
# -------------------------
dataset = load_dataset("json", data_files=DATA_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 构建 prompt 模式，适配 instruction + output
def format_example(example):
    return (
        f"用户：{example['instruction']}\n"
        f"助手：{example['output']}"
    )

def tokenize(example):
    text = format_example(example)
    return tokenizer(text, padding="max_length", truncation=True, max_length=512)

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
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# -------------------------
# 4. 训练参数
# -------------------------
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    save_steps=100,
    num_train_epochs=5,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,  # GTX1060 不支持 bf16
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
