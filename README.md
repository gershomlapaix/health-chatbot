# HealthBot: Medical Q&A Chatbot

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gershomlapaix/health-chatbot/blob/main/healthbot_notebook.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://nsengiyumvaa-health-bot.hf.space/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/gershomlapaix/health-chatbot)

> Fine-tuning TinyLlama-1.1B with QLoRA for accurate, evidence-based healthcare information.

---

## Live Demo

**Try the chatbot now:** [https://nsengiyumvaa-health-bot.hf.space/](https://nsengiyumvaa-health-bot.hf.space/)

**Watch the demo:** [YouTube Video](https://youtu.be/UrIgZDdUnx8)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Fine-Tuning Methodology](#fine-tuning-methodology)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [Steps to Run the Model](#steps-to-run-the-model)
- [Conversation Examples](#conversation-examples)
- [Technical Details](#technical-details)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Overview

HealthBot is a domain-specific conversational assistant built by fine-tuning **TinyLlama-1.1B-Chat-v1.0** on the MedQuAD medical question-answering dataset. The project demonstrates that meaningful domain adaptation of a large language model is achievable on a single consumer-grade GPU using **Quantized Low-Rank Adaptation (QLoRA)** — updating only ~1% of the model's total parameters while producing clinically relevant, coherent responses to health queries.

The project was developed and evaluated through four systematic hyperparameter experiments, with the best configuration achieving a **ROUGE-1 score of 0.2616** and a **BLEU score of 0.1345**, alongside a dramatic reduction in perplexity from a baseline of 45–65 (untuned) down to 2.18 on medical text.

---

## Features

- **Efficient Fine-Tuning**: QLoRA (4-bit NF4 quantization + LoRA) enables training on a free-tier Colab T4 GPU with only 2–3 GB of peak VRAM
- **Systematic Experiments**: 4 hyperparameter configurations explored and compared using ROUGE, BLEU, and perplexity
- **Medical Domain Alignment**: Trained exclusively on NIH-sourced question-answer pairs for grounded, authoritative responses
- **Production-Ready UI**: Gradio web interface with temperature control, token length slider, and a persistent safety disclaimer
- **Reproducible Pipeline**: Modular notebooks for each experiment with saved adapters for easy reload

---

## Dataset

### MedQuAD — Medical Question Answering Dataset

HealthBot was trained on **MedQuAD** (Medical Question Answering Dataset), originally curated by Ben Abacha & Demner-Fushman (2019) and available on Hugging Face as `lavita/MedQuAD`.

| Property | Value |
|---|---|
| **Raw dataset size** | 47,457 question-answer pairs |
| **After preprocessing** | 13,667 examples |
| **Average question length** | 8.2 words |
| **Average answer length** | 129.1 words |
| **Source institutions** | 12 NIH websites (MedlinePlus, NCI, GHR, NIDDK, and others) |
| **Train / Val / Test split** | 80% / 10% / 10% |

### Why MedQuAD?

MedQuAD was selected because its content originates exclusively from authoritative US government health institutions, ensuring that every training signal the model receives is grounded in peer-reviewed, clinically vetted medical knowledge. This stands in contrast to scraped web corpora, which frequently contain inaccurate or contradictory health information.

The dataset spans a broad range of topics: disease symptoms and causes, treatment protocols, drug information, diagnostic procedures, genetic conditions, and preventive health guidance — giving the fine-tuned model wide clinical vocabulary coverage.

### Preprocessing Pipeline

Raw data required several cleaning steps before fine-tuning:

1. **HTML stripping** — Removed all markup tags using regex substitution
2. **Whitespace normalisation** — Collapsed multiple spaces and stripped leading/trailing whitespace
3. **Null filtering** — Dropped rows with missing question or answer fields
4. **Short-answer filtering** — Removed answers shorter than 20 characters (uninformative)
5. **Length capping** — Excluded answers exceeding 300 words to fit within the 512-token context window used during training
6. **Instruction formatting** — Each pair was wrapped in the **ChatML instruction template**:

```
<|system|>
You are HealthBot, a helpful medical information assistant. You provide accurate,
evidence-based health information. Always remind users to consult healthcare
professionals for personal medical advice.</s>
<|user|>
{question}</s>
<|assistant|>
{answer}</s>
```

This template mirrors TinyLlama's pre-training format, making the fine-tuning signal maximally compatible with the model's existing representational structure.

---

## Fine-Tuning Methodology

### Base Model

**TinyLlama-1.1B-Chat-v1.0** was chosen as the base model for its balance of capability and efficiency:

- 1.1 billion parameters (decoder-only transformer)
- Pre-trained on ~3 trillion tokens
- Instruction-tuned chat variant with grouped-query attention and RoPE embeddings
- Apache 2.0 license — fully open source

### QLoRA — Quantized Low-Rank Adaptation

Training a 1.1B parameter model from scratch or in full-precision fine-tuning mode would require tens of gigabytes of GPU memory. Instead, this project uses **QLoRA** (Dettmers et al., 2023), which combines two complementary efficiency strategies:

**1. 4-bit NF4 Quantisation**
The base model weights are compressed from 16-bit to 4-bit NormalFloat precision, reducing memory footprint by approximately 75% with minimal loss in representational quality.

**2. Low-Rank Adaptation (LoRA)**
Rather than updating all model weights, LoRA freezes the base model and injects small trainable adapter matrices into selected linear layers. For a weight matrix **W**, the update is parameterised as:

```
ΔW = B · A
```

where **B ∈ ℝ^(d×r)** and **A ∈ ℝ^(r×k)** with rank **r ≪ min(d, k)**. Only A and B are trained, making the process extremely parameter-efficient.

**Target modules** (all 7 linear projection layers):

| Module Type | Layer Names |
|---|---|
| Self-Attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Feed-Forward (SwiGLU) | `gate_proj`, `up_proj`, `down_proj` |

### Training Infrastructure

| Component | Value |
|---|---|
| Hardware | NVIDIA Tesla T4 (15.6 GB VRAM) |
| Framework | Hugging Face Transformers + PEFT + TRL (SFTTrainer) |
| Optimiser | paged AdamW (8-bit) |
| LR Schedule | Cosine decay with 5% linear warmup |
| Gradient Clipping | 0.3 |
| Weight Decay | 0.001 |
| Precision | bfloat16 |
| Sequence Packing | Enabled (maximises GPU utilisation) |

### Hyperparameter Experiments

Four configurations were tested systematically to identify the optimal training recipe:

| | Experiment 1 | Experiment 2 | Experiment 3 ⭐ | Experiment 4 |
|---|---|---|---|---|
| **Learning Rate** | 2×10⁻⁴ | 1×10⁻⁴ | 2×10⁻⁴ | 5×10⁻⁵ |
| **Per-device Batch** | 2 | 2 | 4 | 2 |
| **Grad. Accumulation** | 4 | 4 | 4 | 8 |
| **Effective Batch** | 8 | 8 | **16** | 16 |
| **Epochs** | 1 | 2 | **3** | 3 |
| **LoRA Rank (r)** | 8 | 16 | 16 | 16 |
| **Training Time** | 11.3 min | 21.6 min | 38.6 min | 33.8 min |
| **Peak GPU** | 2.84 GB | 1.89 GB | 2.02 GB | 1.95 GB |

> ⭐ **Experiment 3** was identified as the best configuration. The combination of a larger effective batch size (16) and three full training epochs produced the only configuration that achieved non-zero BLEU scores and substantially elevated ROUGE scores, consistent with large-batch optimisation theory (Keskar et al., 2017).

---

## Performance Metrics

Models were evaluated on a held-out test set of 20 samples using three complementary metrics:

| Metric | What it measures |
|---|---|
| **ROUGE-1 / ROUGE-2 / ROUGE-L** | N-gram and longest-common-subsequence overlap between generated and reference answers (Lin, 2004) |
| **BLEU** | Modified n-gram precision up to 4-grams with brevity penalty (Papineni et al., 2002) |
| **Perplexity** | How confidently the model assigns probability to the test text — lower is better |

### Results Across All Experiments

| Metric | Exp 1 | Exp 2 | Exp 3 ⭐ Best | Exp 4 |
|---|---|---|---|---|
| **ROUGE-1** | 0.1010 | 0.1010 | **0.2616** | 0.1010 |
| **ROUGE-2** | 0.0511 | 0.0511 | **0.1416** | 0.0511 |
| **ROUGE-L** | 0.0895 | 0.0895 | **0.2154** | 0.0895 |
| **BLEU** | 0.0000 | 0.0000 | **0.1345** | 0.0000 |
| **Perplexity** | 2.00 | 1.97 | 2.18 | 2.03 |
| **Train Loss** | 0.9565 | 0.8857 | **0.8231** | 0.9634 |

### Key Findings

**Experiment 3 is the clear winner across all generation quality metrics.** It is the only configuration that produced a non-zero BLEU score — meaning it is the only model whose outputs share 4-gram sequences with reference answers. Its ROUGE-1 score of 0.2616 represents a **159% improvement** over Experiments 1, 2, and 4.

**Perplexity is not a reliable differentiator here.** All four fine-tuned models achieve perplexity scores between 1.97 and 2.18, far below the estimated 45–65 range for the untuned TinyLlama on medical text. This collapse in perplexity is expected in instruction-tuning contexts where the training template creates highly repetitive structural patterns. ROUGE and BLEU must be treated as the primary quality judges.

**Learning rate matters as much as training duration.** Experiment 4 used the same number of epochs and effective batch size as Experiment 3 but set the learning rate too conservatively (5×10⁻⁵). Despite three full epochs, it produced the *highest* training loss (0.9634) and zero BLEU — confirming that an insufficient learning rate prevents the optimiser from navigating toward the target domain distribution.

---

## Project Structure

```
health-chatbot/
├── app.py                                      # Gradio web interface
├── healthbot_notebook.ipynb                    # Master training notebook (all 4 experiments)
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
├── experiments/                                # Individual experiment notebooks
│   ├── healthbot_notebook_experiment1.ipynb    # LR=2e-4, batch=2, 1 epoch, r=8
│   ├── healthbot_notebook_experiment2.ipynb    # LR=1e-4, batch=2, 2 epochs, r=16
│   ├── healthbot_notebook_experiment3.ipynb    # LR=2e-4, batch=4, 3 epochs, r=16 ⭐
│   └── healthbot_notebook_experiment4.ipynb    # LR=5e-5, batch=2, 3 epochs, r=16
└── healthbot_tinyllama_lora/                   # Trained LoRA adapter artifacts
    └── healthbot_tinyllama_lora/
        ├── adapter_config.json                 # LoRA configuration
        ├── adapter_model.safetensors           # Trained adapter weights
        └── tokenizer_config.json
```

---

## Steps to Run the Model

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.8+ |
| CUDA | 11.8+ (for local GPU inference) |
| GPU VRAM | 6 GB minimum (8 GB+ recommended) |
| Google Colab | Free tier sufficient for inference |

### Option 1 — Try the Live Demo (No Setup Required)

Visit **[https://nsengiyumvaa-health-bot.hf.space/](https://nsengiyumvaa-health-bot.hf.space/)** in your browser. No installation needed.

---

### Option 2 — Run Locally

#### Step 1: Clone the Repository

```bash
git clone https://github.com/gershomlapaix/health-chatbot.git
cd health-chatbot
```

#### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Core dependencies (click to expand)</summary>

```
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
bitsandbytes>=0.41.0
trl>=0.7.0
datasets>=2.14.0
gradio>=4.0.0
accelerate>=0.24.0
sentencepiece
rouge-score
nltk
```

</details>

#### Step 4: Launch the Web Interface

```bash
python app.py
```

Open your browser and navigate to `http://localhost:7860`. The Gradio interface will load the fine-tuned model automatically.

---

### Option 3 — Run in Google Colab

1. Click the badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gershomlapaix/health-chatbot/blob/main/healthbot_notebook.ipynb)
2. In the Colab menu, go to **Runtime → Change runtime type → T4 GPU**
3. Mount Google Drive when prompted (for saving model checkpoints)
4. Run all cells sequentially — training for Experiment 3 completes in ~39 minutes

---

### Option 4 — Python API (Programmatic Access)

Load the fine-tuned model and generate responses directly in your own scripts:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── 1. Configure 4-bit quantisation ──────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── 2. Load base model + tokenizer ───────────────────────────────────────────
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# ── 3. Load LoRA adapter ──────────────────────────────────────────────────────
adapter_path = "./healthbot_tinyllama_lora/healthbot_tinyllama_lora"
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# ── 4. Define inference function ─────────────────────────────────────────────
def ask_healthbot(question: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
    system_prompt = (
        "You are HealthBot, a helpful medical information assistant. "
        "You provide accurate, evidence-based health information. "
        "Always remind users to consult healthcare professionals for personal medical advice."
    )
    prompt = (
        f"<|system|>\n{system_prompt}</s>\n"
        f"<|user|>\n{question}</s>\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

# ── 5. Ask a question ─────────────────────────────────────────────────────────
answer = ask_healthbot("What are the early warning signs of Type 2 diabetes?")
print(answer)
```

---

### Option 5 — Retrain from Scratch (Reproduce Experiments)

To reproduce Experiment 3 (the best configuration):

```python
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir="./healthbot_exp3",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,       # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_grad_norm=0.3,
    weight_decay=0.001,
    optim="paged_adamw_8bit",
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    packing=True,
)

trainer.train()
trainer.model.save_pretrained("./healthbot_tinyllama_lora")
```

Full training notebooks for all four experiments are available in the `experiments/` directory.

---

## Conversation Examples

The following examples illustrate the qualitative impact of fine-tuning. Each pair shows the **untuned TinyLlama base model** response alongside the **fine-tuned HealthBot (Experiment 3)** response for the same question.

---

### Example 1 — Early Symptoms of Type 2 Diabetes

**User:** What are the early warning signs of Type 2 diabetes?

** Base TinyLlama (untuned):**
```
Type 2 diabetes can be a serious condition, but there are some early warning
signs you should be aware of. If you experience any of these symptoms, it is
important to speak with your doctor right away... [continues with generic,
unfocused content, often repeating the question or producing off-topic text]
```

** HealthBot (fine-tuned — Experiment 3):**
```
Early warning signs of Type 2 diabetes include increased thirst (polydipsia),
frequent urination (polyuria), unexplained fatigue, blurred vision, slow-healing
cuts or wounds, tingling or numbness in the hands and feet, and recurring
infections such as yeast infections. Many people with early Type 2 diabetes
experience no symptoms at all, which is why regular screening is important,
especially if you have risk factors such as obesity, a family history of diabetes,
or a sedentary lifestyle. Please consult a healthcare professional for proper
testing and diagnosis.
```

**Impact:** The fine-tuned model correctly identifies the clinical hallmarks (polydipsia, polyuria, peripheral neuropathy symptoms), structures the response logically, and adds appropriate context about asymptomatic cases — all absent in the base model output.

---

### Example 2 — Hypertension and Heart Health

**User:** How does high blood pressure affect the heart over time?

** Base TinyLlama (untuned):**
```
High blood pressure, also known as hypertension, can have a number of negative
effects on the heart. High blood pressure can cause... [loops, repeats phrases,
fails to name specific cardiovascular pathologies]
```

** HealthBot (fine-tuned — Experiment 3):**
```
Chronic high blood pressure forces the heart to work harder than normal to pump
blood through narrowed or stiffened arteries. Over time this leads to left
ventricular hypertrophy, where the walls of the heart's main pumping chamber
thicken and become less efficient. Sustained hypertension also accelerates
atherosclerosis — the build-up of plaque in coronary arteries — significantly
raising the risk of heart attack and stroke. Additionally, the increased workload
can eventually weaken the heart muscle, contributing to heart failure.
Controlling blood pressure through lifestyle changes and, where necessary,
medication is essential to reducing these long-term risks. Consult your doctor
for personalised management advice.
```

**Impact:** The fine-tuned model introduces precise medical terminology (left ventricular hypertrophy, atherosclerosis, coronary arteries) and provides a coherent causal narrative — demonstrating genuine clinical knowledge absorption from the MedQuAD training corpus.

---

### Example 3 — Asthma Management

**User:** What is the difference between a reliever and a preventer inhaler for asthma?

** Base TinyLlama (untuned):**
```
Inhalers are devices used to deliver medication directly to the lungs. There are
two main types of inhalers: reliever inhalers and preventer inhalers...
[produces a vague, textbook-like definition without clinical specifics or guidance]
```

** HealthBot (fine-tuned — Experiment 3):**
```
Reliever inhalers, typically short-acting beta-2 agonists such as salbutamol
(albuterol), work by rapidly relaxing the muscles around the airways to relieve
acute symptoms like wheezing, chest tightness, and shortness of breath. They act
within minutes and are used on an as-needed basis.

Preventer inhalers, usually containing inhaled corticosteroids such as
beclometasone or fluticasone, work differently — they reduce underlying airway
inflammation when used regularly every day, making the airways less sensitive and
reducing the frequency and severity of asthma attacks. They do not provide
immediate relief.

A good asthma management plan typically involves daily use of a preventer inhaler
to control inflammation, with a reliever inhaler available for acute episodes.
Your doctor or asthma nurse can advise on the right combination for your needs.
```

**Impact:** The fine-tuned model names specific drug classes and examples, clearly distinguishes the mechanism of action and usage pattern for each inhaler type, and structures the response in a way that directly helps a patient understand their treatment. This level of clinical specificity is entirely absent from the base model.

---

### Example 4 — Out-of-Domain Handling (Domain Boundary Test)

**User:** Can you help me write a Python script to scrape websites?

** Expected from a general model:**
```
Sure! Here's a Python script using BeautifulSoup and requests...
[proceeds to provide web scraping code — completely off-domain]
```

** HealthBot (fine-tuned — Experiment 3):**
```
I'm HealthBot, a medical information assistant, so I'm not the right tool for
programming tasks. However, if you have any health-related questions — such as
understanding a medical condition, learning about medications, or general
wellness information — I'd be happy to help with those. Please remember that
for personal medical concerns, consulting a qualified healthcare professional
is always the best course of action.
```

**Impact:** The system-prompt conditioning, reinforced through fine-tuning, successfully teaches the model to stay within its intended domain while redirecting users gracefully — a critical safety behaviour for a health-focused assistant.

---

### Example 5 — Drug Information Query

**User:** What should I know about taking ibuprofen regularly?

** HealthBot (fine-tuned — Experiment 3):**
```
Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) commonly used to
relieve pain, reduce fever, and decrease inflammation. When taken regularly,
there are several important considerations:

Gastrointestinal risk: Regular ibuprofen use can irritate the stomach lining,
increasing the risk of ulcers and gastrointestinal bleeding. It is generally
recommended to take it with food or milk.

Kidney function: Long-term or high-dose use can affect kidney function,
particularly in people who are dehydrated, elderly, or who already have kidney
disease.

Cardiovascular risk: NSAIDs including ibuprofen may slightly increase the risk
of heart attack and stroke with prolonged use, especially at higher doses.

Drug interactions: Ibuprofen can interact with blood thinners, other NSAIDs,
certain antidepressants, and blood pressure medications.

The lowest effective dose for the shortest necessary duration is always
recommended. If you find yourself needing ibuprofen regularly, speak with your
doctor to address the underlying cause of your pain and to explore safer
long-term management options.
```

**Impact:** This response demonstrates the model's ability to organise complex multi-faceted drug safety information into clearly labelled categories — a direct result of training on MedQuAD's structured NIH content — while appropriately directing the user to seek professional guidance.

---

## Technical Details

### Architecture

| Component | Value |
|---|---|
| Base Model | TinyLlama-1.1B-Chat-v1.0 |
| Total Parameters | ~620 million |
| Trainable Parameters (LoRA) | 6.31 million (1.01%) |
| Attention Mechanism | Grouped-Query Attention |
| Position Encoding | Rotary Position Embeddings (RoPE) |
| Feed-Forward Activation | SwiGLU |
| Vocabulary Size | 32,000 tokens |
| Max Context Length | 2,048 tokens |

### LoRA Configuration (Best — Experiment 3)

```python
LoraConfig(
    r=16,               # Adapter rank — controls expressiveness
    lora_alpha=32,      # Scaling factor (alpha/r = 2 is standard)
    lora_dropout=0.05,  # Regularisation to prevent overfitting
    bias="none",        # Do not train bias terms
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # Attention layers
        "gate_proj", "up_proj", "down_proj"         # MLP layers
    ],
)
```

### Memory Optimisation Stack

- **4-bit NF4 quantisation** — ~75% VRAM reduction vs FP16 (base model)
- **Double quantisation** — Quantises the quantisation constants for additional savings
- **Gradient checkpointing** — Trades compute for memory during backpropagation
- **Paged AdamW (8-bit)** — Memory-efficient optimiser with CPU offloading for optimizer states
- **Sequence packing** — Concatenates multiple short examples into full-length sequences for better GPU utilisation

### Technologies Used

| Library | Purpose |
|---|---|
| `transformers` | Model loading, tokenisation, and generation |
| `peft` | LoRA adapter management |
| `bitsandbytes` | 4-bit NF4 quantisation |
| `trl` | SFTTrainer for supervised fine-tuning |
| `datasets` | MedQuAD dataset loading and preprocessing |
| `torch` | Core deep learning operations |
| `accelerate` | Multi-device training abstraction |
| `gradio` | Web interface |
| `rouge-score` | ROUGE evaluation |
| `nltk` | BLEU evaluation |

---

## Limitations

| Limitation | Description |
|---|---|
| **Not a medical professional** | HealthBot provides general health information only — it cannot diagnose conditions, prescribe treatments, or replace a doctor's advice |
| **Training data bounds** | Knowledge is limited to MedQuAD content from NIH sources; rare diseases and emerging treatments may not be covered |
| **Small evaluation set** | Metrics were computed on 20 test samples — larger evaluation sets would yield more statistically robust conclusions |
| **English only** | The model does not support multilingual queries |
| **Context window** | Responses are limited to the 2,048-token context window inherited from TinyLlama |
| **No real-time updates** | Medical knowledge is static as of the MedQuAD dataset version used — the model has no access to recent clinical guidelines or research |

---

## Future Work

- [ ] **Larger evaluation set** — Scale test set to 300+ examples with bootstrap confidence intervals
- [ ] **Retrieval-Augmented Generation (RAG)** — Ground responses in up-to-date PubMed / UpToDate content
- [ ] **Broader training data** — Supplement with MedQA, PubMedQA, and clinical guidelines
- [ ] **Human clinical validation** — Expert review of model outputs by qualified healthcare professionals
- [ ] **Multilingual support** — Extend to French, Spanish, Swahili, and other widely spoken languages
- [ ] **Preference alignment** — Apply DPO or RLHF to better align responses with patient safety preferences
- [ ] **Larger model variants** — Explore 7B and 13B parameter base models with the same QLoRA approach
- [ ] **Deployment optimisation** — ONNX export and TensorRT quantisation for faster inference
- [ ] **Continuous evaluation** — Automated metric tracking on clinical benchmarks (MedQA, PubMedQA)

---

## References

- Ben Abacha, A., & Demner-Fushman, D. (2019). A question-entailment approach to question answering. *BMC Bioinformatics*, 20(1), 511.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS 2023*.
- Hu, E. J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.
- Zhang, P., Zeng, G., Wang, T., & Lu, W. (2024). TinyLlama: An open-source small language model. *arXiv:2401.02385*.
- Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. *ACL Workshop*.
- Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A method for automatic evaluation of machine translation. *ACL 2002*.
- Keskar, N. S., et al. (2017). On large-batch training for deep learning. *ICLR 2017*.

