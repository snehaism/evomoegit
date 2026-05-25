"""
medgemma_amr_extractor.py
EVO-MOE — Gemma 4 E4B AMR extraction engine
Uses Llama.from_pretrained — no manual model path management needed.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False


@dataclass
class AMRRecord:
    organism_eucast: str
    antibiotic:      str
    sir_result:      str
    mic_value:       Optional[float]
    ward:            str
    specimen_type:   str
    collection_date: str
    patient_id:      str
    confidence:      float
    source:          str = "medgemma"


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
KNOWN_DRUGS = [
    "meropenem", "imipenem", "ertapenem", "doripenem",
    "ciprofloxacin", "levofloxacin",
    "ceftriaxone", "cefepime", "ceftazidime",
    "vancomycin", "teicoplanin",
    "colistin", "polymyxin",
    "linezolid", "daptomycin",
    "gentamicin", "amikacin",
    "piperacillin", "ampicillin", "aztreonam",
]

KNOWN_ORGANISMS = [
    {"eucast": "Klebsiella pneumoniae",    "patterns": ["klebsiella pneumoniae", "k. pneumoniae", "kp ", " kp,", "kleb"]},
    {"eucast": "Acinetobacter baumannii",  "patterns": ["acinetobacter baumannii", "a. baumannii", " ab ", "aci"]},
    {"eucast": "Pseudomonas aeruginosa",   "patterns": ["pseudomonas aeruginosa", "p. aeruginosa", " pa ", "psa"]},
    {"eucast": "Staphylococcus aureus",    "patterns": ["staphylococcus aureus", "s. aureus", "staph aureus", "mrsa"]},
    {"eucast": "Enterococcus faecium",     "patterns": ["enterococcus faecium", "e. faecium", "efm"]},
    {"eucast": "Enterobacter cloacae",     "patterns": ["enterobacter cloacae", "e. cloacae", "ecl"]},
    {"eucast": "Escherichia coli",         "patterns": ["escherichia coli", "e. coli", "ecoli"]},
    {"eucast": "Streptococcus pneumoniae", "patterns": ["streptococcus pneumoniae", "s. pneumoniae", "pneumococcus"]},
]

SIR_MAP = {
    "resistant": "R", "resistance": "R",
    "sensitive": "S", "susceptible": "S",
    "intermediate": "I",
    "sdd": "SDD",
    "प्रतिरोधी": "R",
    "संवेदनशील": "S",
}

SPECIMEN_MAP = {
    "blood": "blood", "bld": "blood", "bc": "blood",
    "urine": "urine", "ur": "urine", "mssu": "urine",
    "sputum": "sputum", "sput": "sputum", "bal": "sputum",
    "wound": "wound", "swab": "wound", "pus": "wound",
    "csf": "csf", "cerebrospinal": "csf",
}

WARD_MAP = {
    "icu": "ICU", "intensive care": "ICU", "critical care": "ICU",
    "hdu": "HDU", "high dependency": "HDU",
    "ed": "ED", "emergency": "ED",
    "surgical": "surgical", "surgery": "surgical",
    "paediatric": "paediatric", "pediatric": "paediatric", "nicu": "paediatric",
    "general": "general", "medicine": "general",
}


class MedGemmaAMRExtractor:

    MODEL_REPO = "ggml-org/gemma-4-E4B-it-GGUF"
    MODEL_PATTERN = "gemma-4-E4B-it-Q4_K_M.gguf"

    def __init__(self):
        self.llm = None
        if LLAMA_AVAILABLE:
            try:
                print(f"Loading Gemma 4 E4B from {self.MODEL_REPO}...")
                self.llm = Llama.from_pretrained(
                    repo_id=self.MODEL_REPO,
                    filename=self.MODEL_PATTERN,
                    cache_dir="/data",
                    n_ctx=4096,
                    n_threads=2,
                    verbose=True,
                    token=os.getenv("HF_TOKEN"),
                )
                print("Gemma 4 E4B loaded successfully.")
            except Exception as e:
                print(f"Model load failed: {e} — using rule-based fallback.")

    # ─────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────
    def extract(self, report_text: str) -> List[AMRRecord]:
        if self.llm is not None:
            records = self._gemma_extract(report_text)
            if records:
                return records
        return self._rule_based_extract(report_text)

    # ─────────────────────────────────────────────
    # Gemma 4 extraction
    # ─────────────────────────────────────────────
    def _gemma_extract(self, report_text: str) -> List[AMRRecord]:
        prompt = (
            "<start_of_turn>system\n"
            "You are a medical data extraction assistant. Extract antimicrobial "
            "susceptibility data from lab reports and return a JSON array. "
            "Each object must have: organism_eucast, antibiotic, sir_result (S/I/R/SDD), "
            "mic_value (float or null), ward, specimen_type, "
            "collection_date (YYYY-MM-DD or null), patient_id. "
            "Output ONLY valid JSON, no other text."
            "<end_of_turn>\n"
            "<start_of_turn>user\n"
            f"Extract data from:\n{report_text}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        try:
            response = self.llm(prompt, max_tokens=2048, temperature=0.1, stop=["<end_of_turn>"])
            output = response["choices"][0]["text"].strip()
            start = output.find("[")
            end = output.rfind("]") + 1
            if start == -1 or end == 0:
                return []
            data = json.loads(output[start:end])
            return self._parse_json_records(data, source="medgemma")
        except Exception as e:
            print(f"Gemma extraction error: {e}")
            return []

    def _parse_json_records(self, data: list, source: str) -> List[AMRRecord]:
        records = []
        for item in data:
            if not isinstance(item, dict):
                continue
            sir = str(item.get("sir_result", "")).upper().strip()
            if sir not in {"S", "I", "R", "SDD"}:
                continue
            mic = None
            mic_raw = item.get("mic_value")
            if mic_raw is not None:
                try:
                    mic = float(str(mic_raw).replace(">", "").replace("<", "").strip())
                    if not (0.001 <= mic <= 2048):
                        mic = None
                except (ValueError, TypeError):
                    mic = None
            records.append(AMRRecord(
                organism_eucast=str(item.get("organism_eucast", "")).strip(),
                antibiotic=str(item.get("antibiotic", "")).strip().lower(),
                sir_result=sir,
                mic_value=mic,
                ward=str(item.get("ward", "general")).strip(),
                specimen_type=str(item.get("specimen_type", "other")).strip(),
                collection_date=str(item.get("collection_date", "")).strip(),
                patient_id=str(item.get("patient_id", "")).strip(),
                confidence=0.9,
                source=source,
            ))
        return records

    # ─────────────────────────────────────────────
    # Rule-based fallback
    # ─────────────────────────────────────────────
    def _rule_based_extract(self, report_text: str) -> List[AMRRecord]:
        text_lower = report_text.lower()

        organism = "unknown"
        for org in KNOWN_ORGANISMS:
            if any(p in text_lower for p in org["patterns"]):
                organism = org["eucast"]
                break
        if organism == "unknown":
            return []

        specimen = next((v for k, v in SPECIMEN_MAP.items() if k in text_lower), "other")
        ward = next((v for k, v in WARD_MAP.items() if k in text_lower), "general")
        date = self._extract_date(report_text)
        patient_id = self._extract_patient_id(report_text)

        records = []
        for drug in KNOWN_DRUGS:
            pattern = (
                rf'\b{re.escape(drug)}\b[\s:=,|]*'
                r'(?:(?:MIC|mic)[\s:=]*(?:[><=]?\s*[\d.]+)\s*(?:mg/L|mg/l|µg/mL)?\s*)?'
                r'([SsRrIi])\b'
                r'|'
                rf'\b{re.escape(drug)}\b[\s:=,|]*'
                r'(resistant|sensitive|susceptible|intermediate|प्रतिरोधी|संवेदनशील)'
            )
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                sir_raw = (match.group(1) or match.group(2) or "").strip().lower()
                sir = SIR_MAP.get(sir_raw, sir_raw.upper())
                if sir_raw in {"r"}: sir = "R"
                elif sir_raw in {"s"}: sir = "S"
                elif sir_raw in {"i"}: sir = "I"
                if sir not in {"S", "I", "R", "SDD"}:
                    continue
                mic = self._extract_mic_near_drug(report_text, drug)
                records.append(AMRRecord(
                    organism_eucast=organism,
                    antibiotic=drug,
                    sir_result=sir,
                    mic_value=mic,
                    ward=ward,
                    specimen_type=specimen,
                    collection_date=date,
                    patient_id=patient_id,
                    confidence=0.65,
                    source="fallback",
                ))
                break
        return records

    def _extract_date(self, text: str) -> str:
        for p in [r"(\d{4}-\d{2}-\d{2})", r"(\d{2}/\d{2}/\d{4})"]:
            m = re.search(p, text)
            if m:
                try:
                    import pandas as pd
                    return pd.to_datetime(m.group(1)).strftime("%Y-%m-%d")
                except Exception:
                    return m.group(1)
        return ""

    def _extract_patient_id(self, text: str) -> str:
        m = re.search(
            r"(?:patient[\s_-]*(?:id|no|number)|MRN|HN|accession)[:\s#]+([A-Z0-9\-]+)",
            text, re.IGNORECASE,
        )
        return m.group(1).strip() if m else ""

    def _extract_mic_near_drug(self, text: str, drug: str) -> Optional[float]:
        idx = text.lower().find(drug.lower())
        if idx == -1:
            return None
        window = text[idx:idx + 80]
        m = re.search(r"(?:MIC[\s:=]*)?([><=]?\s*[\d.]+)\s*(?:mg/L|mg/l|µg/mL)?", window)
        if m:
            try:
                val = float(m.group(1).replace(">","").replace("<","").replace("=","").strip())
                return val if 0.001 <= val <= 2048 else None
            except (ValueError, TypeError):
                return None
        return None
