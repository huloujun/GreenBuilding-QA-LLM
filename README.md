#  GreenBuilding-QA-LLM
**基于 Qwen2.5-0.5B + QLoRA 微调的绿色建筑问答大模型**

**基于 Qwen2.5-7B + QLoRA 微调的绿色建筑问答大模型**

本项目旨在构建一个轻量、高效、可在普通消费级 GPU（如 GTX1060 6GB、RTX4090 24GB）上运行的专业绿色建筑问答模型。  
模型基于 `Qwen2.5-0.5B  `或`Qwen2.5-0.5B  `，使用 `QLoRA` 技术进行低显存微调，并可通过本地 Web 界面进行交互。

---

##  项目特点

### 绿色建筑领域专属知识  
使用自建数据集（如体形系数、窗墙比、U 值、SHGC 等）进行强化学习，模型对绿建设计问题具有强识别与回答能力。

###  轻量 / 本地运行  
仅使用 **GTX1060 6GB** 或**RTX4090 24GB**+ QLoRA 即可完成训练与推理。

###  可复现的完整训练流程  
提供训练脚本、推理脚本、Web UI 全流程。

---

##  1. 训练数据格式（JSONL）

```json
{"instruction": "建筑体形系数对能耗有什么影响？", "output": "..."}
{"instruction": "夏热冬冷地区外窗U值一般控制在多少？", "output": "..."}
```



##  2. 环境配置

### 1. 创建环境
```bash
conda create -n llm python=3.10 -y
conda activate llm
```

### 2. 安装 PyTorch（根据显卡选择 CUDA 版本）

```
# RTX 40/30/20 系列（推荐）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# GTX 10 系列（1060/1080）
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. 安装项目依赖

```
pip install transformers datasets accelerate peft bitsandbytes safetensors
pip install huggingface-hub
```



##  3. 微调模型

python model/train_lora1060.py

python model/train_lora4090.py

微调后权重自动保存至：

outputs/lora/

##  4. 本地推理

python model/infer_lora1060.py

python model/infer_lora4090.py

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
- Qwen2.5-7B 
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