#  GreenBuilding-QA-LLM
**基于 Qwen2.5-0.5B + QLoRA 微调的绿色建筑问答大模型**

本项目旨在构建一个轻量、高效、可在普通消费级 GPU（如 GTX1060 6GB）上运行的专业绿色建筑问答模型。  
模型基于 `Qwen2.5-0.5B`，使用 `QLoRA` 技术进行低显存微调，并可通过本地 Web 界面进行交互。

---

##  项目特点

### 绿色建筑领域专属知识  
使用自建数据集（如体形系数、窗墙比、U 值、SHGC 等）进行强化学习，模型对绿建设计问题具有强识别与回答能力。

###  轻量 / 本地运行  
仅使用 **GTX1060 6GB** + QLoRA 即可完成训练与推理。

###  可复现的完整训练流程  
提供训练脚本、推理脚本、Web UI 全流程。

---

##  目录结构

GreenBuilding-QA-LLM/
 │
 ├── data/
 │   └── gb_qa.jsonl             # 训练数据
 │
 ├── model/
 │   ├── train_lora_1060.py      # QLoRA 微调脚本
 │   ├── infer_lora.py           # 推理脚本
 │
 ├── app/
 │   └── web_ui.py               # 网页端 ChatBot
 │
 ├── outputs/
 │   └── lora/                   # LoRA 微调后的权重
 │
 └── README.md

---

##  1. 训练数据格式（JSONL）

```json
{"instruction": "建筑体形系数对能耗有什么影响？", "output": "..."}
{"instruction": "夏热冬冷地区外窗U值一般控制在多少？", "output": "..."}
```



##  2. 环境配置

```
conda env create -f llm_env.yaml
conda activate llm
```

##  3. 微调模型

python model/train_lora_1060.py

微调后权重自动保存至：

outputs/lora/

##  4. 本地推理

python model/infer_lora.py

输入：

```
建筑体形系数对能耗有什么影响？
```

即可看到微调模型的回答。

------



------

##  示例效果

**输入：**

```
热冬冷地区外窗 U 值一般控制在多少？
```

**模型回答：**

```
一般控制在 2.0~2.5 W/m²·K，以平衡保温性能和舒适性。
```

------

##  技术栈

- Qwen2.5-0.5B
- LoRA / QLoRA
- Transformers
- BitsAndBytes（4bit 量化）
- Gradio
- HuggingFace Datasets

------

##  作者

中国建筑第五工程局 · 绿色建筑与数智技术方向
胡楼君

------

## 📄 License

MIT License