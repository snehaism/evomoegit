"""
finetune_gemma4.py
EVO-MOE · Gemma 4 E4B Fine-tuning for AMR Extraction

LoRA fine-tune on 9,400+ synthetic WHONET-format training pairs.
Produces the on-device model (merged + GGUF) used in the EVO-MOE pipeline.

Requirements:
    pip install transformers peft trl datasets torch accelerate bitsandbytes

Run:
    python finetune_gemma4.py
"""

import os
import json
import torch
from dotenv import load_dotenv
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

load_dotenv()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_ID = "google/gemma-4-e4b-it"
TOKEN = os.getenv("HUGGINGFACE_TOKEN")
OUTPUT_DIR = "./evomoe-gemma4-lora"
MERGED_DIR = "./evomoe-gemma4-merged"

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=0,
)


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, token=TOKEN, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=TOKEN,
    )

    return model, tokenizer


# ─────────────────────────────────────────────
# Format training pairs
# ─────────────────────────────────────────────
def format_example(report_text: str, isolate_records: list) -> str:
    """
    Format one training example as a Gemma 4 instruction/response pair.

    Input:  unstructured lab report text
    Output: JSON array of structured isolate records
    """
    return (
        f"<start_of_turn>user\n"
        f"Extract all antibiotic susceptibility results from this lab report.\n"
        f"Return a JSON array. Each object must have: organism_eucast, antibiotic, "
        f"sir_result (S/I/R/SDD), mic_value (number or null), ward, specimen_type, "
        f"collection_date (YYYY-MM-DD or null), patient_id (string or null).\n"
        f"Return ONLY the JSON array. No explanation.\n\n"
        f"Lab report:\n{report_text}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{json.dumps(isolate_records, ensure_ascii=False)}\n"
        f"<end_of_turn>"
    )


# ─────────────────────────────────────────────
# Load training data
# ─────────────────────────────────────────────
def load_training_data(data_path: str = "data/training_pairs.json") -> Dataset:
    """
    Load synthetic WHONET training pairs.
    Generate them first with: python generate_training_data.py
    """
    with open(data_path) as f:
        raw = json.load(f)

    examples = []
    for item in raw:
        text = format_example(item["report"], item["records"])
        examples.append({"text": text})

    dataset = Dataset.from_list(examples)
    print(f"Loaded {len(dataset)} training examples")
    return dataset


# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────
def train():
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = load_model()

    print("Applying LoRA...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    print("Loading training data...")
    dataset = load_training_data()

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    print("Training...")
    trainer = SFTTrainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=tokenized,
        dataset_text_field="input_ids",
        tokenizer=tokenizer,
        max_seq_length=1024,
    )

    trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done. Next steps:")
    print("  1. Merge adapter:  python merge_adapter.py")
    print("  2. Convert to GGUF: llama.cpp/convert_hf_to_gguf.py")
    print("  3. Quantize:       llama-quantize model.gguf Q4_K_M")

    return trainer


# ─────────────────────────────────────────────
# Merge adapter into base weights
# ─────────────────────────────────────────────
def merge_adapter():
    from peft import PeftModel

    print(f"Loading base model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        token=TOKEN,
    )

    print(f"Loading LoRA adapter from {OUTPUT_DIR}...")
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_DIR}...")
    model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

    print(f"Merged model saved. Convert to GGUF with:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {MERGED_DIR} --outtype f16")
    print(f"  ./llama-quantize {MERGED_DIR}/model.gguf Q4_K_M")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge_adapter()
    else:
        train()
