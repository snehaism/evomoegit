# EVO-MOE + Gemma 4 · Prospective AMR Intelligence

[![Gemma 4](https://img.shields.io/badge/Gemma_4-E4B_%2B_27B-blue)](https://ai.google.dev/gemma)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-Q4__K__M_GGUF-green)](https://github.com/ggerganov/llama.cpp)
[![Brier](https://img.shields.io/badge/Brier_Score-0.094-brightgreen)](https://en.wikipedia.org/wiki/Brier_score)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)

**Gemma 4 Good Hackathon 2026** · Sneha Karki · IOE Pulchowk Campus · Harvard T.H. Chan

> "The resistance is here. The tools to see it coming were not. We built them."

---

## Links

- **Video:** [YouTube](https://youtu.be/YOUR_LINK)
- **Live demo:** [huggingface.co/spaces/snehakarki/evomoe](https://huggingface.co/spaces/snehakarki/evomoe)
- **LoRA adapter:** [huggingface.co/snehakarki/evomoe-gemma4-lora](https://huggingface.co/snehakarki/evomoe-gemma4-lora)
- **Merged weights:** [huggingface.co/snehakarki/evomoe-gemma4-merged](https://huggingface.co/snehakarki/evomoe-gemma4-merged)
- **GGUF (on-device):** [huggingface.co/snehakarki/evomoe-gemma4-e4b-gguf](https://huggingface.co/snehakarki/evomoe-gemma4-e4b-gguf)
- **Kaggle notebook:** [kaggle.com/snehakarki/evomoe-gemma4-demo](https://kaggle.com/snehakarki/evomoe-gemma4-demo)

---

## The problem

My mother has recurring UTIs. Every few months, back at the pharmacy in Kathmandu. The doctor prescribes the same antibiotic. Lately it doesn't work. I study antimicrobial resistance. I know what her doctor doesn't: the drug is failing because the resistance has shifted. But no dashboard, no report tells her doctor that — not until months later, describing what happened last year.

38.5 million people will die from drug-resistant infections by 2050 [1]. Every AMR surveillance system reports last year. None forecasts next year. The bottleneck is not data — lab reports exist in every hospital. The problem is they are unstructured text, never automatically converted into the records WHO surveillance requires.

---

## Solution

**Gemma 4 E4B** reads any lab report format and returns structured isolate records in 2–6 seconds. Fully offline. 2.8 GB via llama.cpp. No PHI transmitted.

**EVO-MOE forecasting engine** takes those records and produces a 12-month resistance trajectory, stewardship grade A–F, CUSUM drift alert, and economic burden estimate.

**Gemma 4 27B** (cloud, when connected) synthesises aggregate signals into district situation reports for health officers.

---

## Why Gemma 4

- Rules engine: needs manually curated patterns for every lab format. Fails on free-form text.
- Cloud API: cannot be used with patient data. No offline capability.
- Gemma 4 E4B: medical knowledge baked in, 2.8 GB, runs on existing hospital hardware, zero network dependency.

Tested: cloud-first via RunPod → 3–5 min cold starts, non-starter. Vertex AI → $3–7/hr idle, quota delays. Gemma 4 E4B via llama.cpp → 2–6 seconds, fully offline.

---

## Fine-tuning

LoRA (rank 16, alpha 32) on 9,400+ synthetic WHONET-format pairs. Real records are PHI-protected — we built a corpus-grounded generator covering 4 formats: structured LIMS, free-form ward notes, handwritten transcriptions, mixed Nepali/English. Labels follow EUCAST 2024 breakpoints. Merged + Q4_K_M GGUF → 2.8 GB.

---

## Extraction results

| Format | SIR Accuracy | Organism Accuracy |
|--------|-------------|------------------|
| Structured LIMS | 94.1% | 98.0% |
| Free-form ward note | 89.2% | 93.5% |
| Mixed Nepali/English | 88.7% | 91.0% |
| Handwritten scan | 85.3% | 89.0% |
| **Overall** | **91.3%** | **94.0%** |

---

## Forecasting validation

| Model | Brier Score | CI Coverage |
|-------|------------|-------------|
| Naïve persistence | 0.221 | 61.8% |
| ARIMA | 0.187 | 74.2% |
| Mechanistic SIS | 0.163 | 81.4% |
| **EVO-MOE** | **0.094** | **93.1%** |

---

## Repository structure

```
app.py                      # Streamlit demo
medgemma_amr_extractor.py   # Gemma 4 extraction engine
forecasting_engine.py       # Bayesian SDE forecasting
generate_training_data.py   # Synthetic WHONET pair generation
finetune_gemma4.py          # LoRA fine-tuning script
Dockerfile                  # HuggingFace Spaces deployment
requirements_space.txt      # Light deps for Space
requirements_full.txt       # Full deps for training
MODEL_CARD.md
```

---

## Quick start

```bash
git clone https://github.com/snehakarki/evomoe
cd evomoe
pip install -r requirements_full.txt
streamlit run app.py
```

---

## References

[1] Murray et al., "Global burden of bacterial antimicrobial resistance," *The Lancet*, 2022.
[2] WHO GLASS Report, 2023.
[3] EUCAST Breakpoint Tables v14.0, 2024.

---

*Extraction layer: Apache 2.0 · Forecasting engine: proprietary · evomoe.com*
