# 🧠 AI-Powered Paraphrasing Tool  
Transformer-Based Text Rewriting with Evaluation Metrics

A production-oriented NLP application that generates high-quality paraphrased text using modern large language models (LLMs).  
The project demonstrates model experimentation, evaluation-driven development, and cross-environment deployment (Colab + local Apple Silicon).

---

## 🚀 Project Overview

This project implements an AI-powered paraphrasing system using transformer models.  
It focuses on:

- Meaning preservation
- Controlled lexical variation
- Evaluation-driven generation
- Cross-platform deployment

The project includes two implementations:

| Environment | File | Model | Purpose |
|-------------|------|-------|---------|
| Google Colab (CUDA GPU) | `Paraphrasing_LLaMA2.ipynb` | LLaMA 2 (7B Chat) | Experimentation & evaluation |
| Local VSCode (Apple MPS) | `paraphraser.py` | Mistral-7B-Instruct | Production-ready CLI tool |

---

## 🧠 Why Two Implementations?

### 📓 Notebook (LLaMA 2 – Colab)

- Used for experimentation and metric analysis
- Runs with 4-bit quantization (bitsandbytes)
- Leverages CUDA GPU for efficient large-model inference
- Designed for iterative prompt tuning and evaluation benchmarking

**Reasoning:**  
LLaMA 2 performs well but is heavy. Colab’s CUDA environment enables efficient experimentation with quantized large models.

---

### 🖥 Production Script (Mistral – Local VSCode)

- Runs locally on Apple Silicon (MPS)
- Console-based input/output
- Deterministic and optimized for stability
- Designed as a deployable CLI tool

**Reasoning:**  
Mistral-7B-Instruct provides strong instruction-following performance with better local compatibility.  
It runs smoothly on Apple M-series hardware without CUDA dependency.

---

## 🎯 Key Engineering Decisions
### 🔹 LLaMA 2 for Experimentation

✔ Strong instruction tuning

✔ CUDA-optimized via quantization

✔ Ideal for benchmarking and evaluation research

### 🔹 Mistral for Production

✔ Efficient on Apple Silicon (MPS)

✔ Stable decoding

✔ Strong instruction adherence

✔ Cleaner local deployment

### 🔹 Deterministic Decoding for Stability

✔ Reduced randomness to ensure:

✔ Consistent semantic similarity

✔ Reduced hallucinations

✔ Production reliability

## 🏗 Architecture
<img width="50%" height="50%" alt="image" src="https://github.com/user-attachments/assets/0c4532d8-508b-4166-b4b7-1d24eaa3e152" />



## 📊 Evaluation Strategy

Rather than relying on lexical overlap alone, the system uses multiple evaluation metrics:

- **ROUGE** → Token-level overlap
- **BLEU / SacreBLEU** → Structural similarity
- **Semantic Similarity (MiniLM embeddings)** → Meaning preservation
- **Grammar correction layer** → Fluency assurance

This ensures paraphrases are:

✔ Meaning-preserving  
✔ Structurally valid  
✔ Grammatically sound  
✔ Lexically diverse  

---
## 📓 Colab Notebook (Paraphrasing_LLaMA2.ipynb)

Designed for:

- Prompt experimentation
- Hyperparameter tuning
- Multi-metric evaluation
- GPU-based quantized inference

Uses: meta-llama/Llama-2-7b-chat-hf

With 4-bit quantization for efficient execution.

**Sample Colab Output**
<img width="1636" height="266" alt="image" src="https://github.com/user-attachments/assets/79d210aa-3722-4e1b-82a7-4db29ace0fe3" />


## 💻 Local CLI Version (`paraphraser.py`)

### Run:

```bash
conda activate llama-env
pip install requirements.txt
python paraphraser.py
```
Sample Terminal Output
<img width="1123" height="478" alt="image" src="https://github.com/user-attachments/assets/7c5a4031-00e9-46b9-8fbb-34c44ff4257f" />

## 📦 Requirements

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- LanguageTool
- Rouge-score
- SacreBLEU
- NLTK
(See requirements.txt)

