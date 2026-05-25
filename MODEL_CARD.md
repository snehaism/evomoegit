---
license: apache-2.0
base_model: google/gemma-4-e4b-it
tags:
  - medical
  - amr
  - antimicrobial-resistance
  - whonet
  - gguf
  - llama-cpp
  - gemma4
language:
  - en
  - ne
pipeline_tag: text-generation
---

# EVO-MOE · Gemma 4 E4B · AMR Extractor

Fine-tuned Gemma 4 E4B for clinical antimicrobial resistance extraction.
Part of the EVO-MOE prospective AMR forecasting platform.

## What it does

Reads any lab report format → structured WHO GLASS-ready isolate records.

**Output per antibiotic tested:**
```json
{
  "organism_eucast": "Klebsiella pneumoniae",
  "antibiotic": "meropenem",
  "sir_result": "R",
  "mic_value": 8.0,
  "ward": "ICU",
  "specimen_type": "blood",
  "collection_date": "2024-06-15",
  "patient_id": "MRN-00441"
}
```

## Performance

| Metric | Score |
|--------|-------|
| SIR accuracy | 91.3% |
| Organism accuracy | 94.0% |
| MIC within 1 doubling dilution | 85.7% |

## Fine-tuning details

- **Base:** google/gemma-4-e4b-it
- **Method:** LoRA (rank 16, alpha 32)
- **Training data:** 9,400 synthetic WHONET-format pairs
- **Labels:** EUCAST 2024 breakpoints
- **Formats:** structured LIMS · free-form · mixed Nepali/English · handwritten
- **Quantization:** Q4_K_M GGUF via llama.cpp

## Usage

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="snehakarki/evomoe-gemma4-e4b-gguf",
    filename="*Q4_K_M*",
    n_ctx=4096,
)
```

## Links

- [Source code](https://github.com/snehakarki/evomoe)
- [Live demo](https://huggingface.co/spaces/snehaiism/EvoMoE)
- [API](https://snehaiism-evomoe-api.hf.space/docs)

**Clinical disclaimer:** Population-level stewardship planning only. NOT patient-level clinical decision support.
