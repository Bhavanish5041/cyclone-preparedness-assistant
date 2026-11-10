"""Phase 2: Tokenizer Audit for CycloneLM v0.2 Vocabulary.

Evaluates BPETokenizer (vocab_size=1,024) across:
1. Common English sentences
2. Technical cyclone terminology
3. Numerical values
4. JSON strings
5. Community names
6. Tool schemas

Measures:
- Tokens per word (subword fragmentation)
- Character-to-token compression ratio
- Unknown token (<unk>) count and frequency
- Subword decomposition analysis
- Vocabulary coverage across the domain corpus
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.cyclonelm.tokenizer.bpe_tokenizer import BPETokenizer

TEST_CASES = {
    "common_english": [
        "The quick brown fox jumps over the lazy dog.",
        "How can I help you prepare for the upcoming storm season?",
        "Disaster management requires coordination between local hospitals and emergency shelters.",
        "Please provide a clear and simple explanation of the safety protocol.",
        "Residents living along the coastline must follow evacuation instructions immediately.",
    ],
    "cyclone_terminology": [
        "Very Severe Cyclonic Storm",
        "Extremely Severe Cyclonic Storm",
        "Super Cyclonic Storm",
        "Maximum Sustained Surface Wind Speed",
        "Analytic Hierarchy Process",
        "Consistency Ratio strictly less than 0.10",
        "Multimodal early warning dissemination",
    ],
    "numerical_values": [
        "61.06",
        "65.04",
        "50.06",
        "1,879",
        "622",
        "118 to 166 km/h",
        "0.10",
        "40.0",
        "59.99",
        "1257",
        "369",
        "253",
    ],
    "json_syntax": [
        '{"tool": "lookup_community", "parameters": {"query": "Ullal", "district": "Dakshina Kannada"}}',
        '{"district_a": "Dakshina Kannada", "district_b": "Udupi"}',
        '{"identifier": "Tallur"}',
        '{"priority_band": "BAND_1_CRITICAL", "limit": 5}',
        '{"archetype_id": 3}',
    ],
    "community_names": [
        "Ullal",
        "Tallur",
        "Bengre",
        "Edmoge",
        "Malpe",
        "Someshwara",
        "Paduvari",
        "Hebri",
        "Bhatkal",
        "Kundapura",
        "Mangaluru",
        "Karwar",
        "Kumta",
    ],
    "tool_schemas": [
        "lookup_community",
        "analyze_infrastructure_and_exposure",
        "explain_preparedness_assessment",
        "get_intervention_recommendations",
        "compare_districts",
        "filter_and_rank_communities",
        "explain_vulnerability_archetype",
    ],
}


def audit_tokenizer() -> Dict[str, Any]:
    print("=" * 70)
    print("CYCLONELM PHASE 2: TOKENIZER COMPREHENSIVE AUDIT")
    print("=" * 70)

    tok_path = Path("models/cyclonelm/v0.2/tokenizer/vocab.json")
    assert tok_path.exists(), f"Tokenizer missing at {tok_path}"
    tokenizer = BPETokenizer.load(tok_path)

    vocab_size = tokenizer.vocab_size
    print(f"Loaded Tokenizer: {tok_path} | Vocab Size: {vocab_size}")

    audit_report = {
        "vocab_size": vocab_size,
        "special_tokens_count": len(tokenizer.special_tokens),
        "merges_count": len(tokenizer.merges),
        "categories": {},
    }

    total_words_all = 0
    total_tokens_all = 0
    total_chars_all = 0
    total_unk_all = 0

    for category, examples in TEST_CASES.items():
        cat_words = 0
        cat_tokens = 0
        cat_chars = 0
        cat_unk = 0
        item_details = []

        for ex in examples:
            enc = tokenizer.encode(ex, add_bos=False, add_eos=False)
            token_ids = enc["input_ids"]
            decoded_tokens = [tokenizer.id_to_token.get(tid, "<unk>") for tid in token_ids]

            words = ex.split()
            word_count = max(1, len(words))
            token_count = len(token_ids)
            char_count = len(ex)
            unk_count = sum(1 for tid in token_ids if tid == tokenizer.unk_id)

            tokens_per_word = round(token_count / word_count, 2)
            chars_per_token = round(char_count / max(1, token_count), 2)

            cat_words += word_count
            cat_tokens += token_count
            cat_chars += char_count
            cat_unk += unk_count

            item_details.append({
                "text": ex,
                "token_count": token_count,
                "word_count": word_count,
                "tokens_per_word": tokens_per_word,
                "chars_per_token": chars_per_token,
                "tokens": decoded_tokens,
                "unk_count": unk_count,
            })

        avg_tpw = round(cat_tokens / max(1, cat_words), 2)
        avg_cpt = round(cat_chars / max(1, cat_tokens), 2)

        total_words_all += cat_words
        total_tokens_all += cat_tokens
        total_chars_all += cat_chars
        total_unk_all += cat_unk

        audit_report["categories"][category] = {
            "example_count": len(examples),
            "total_words": cat_words,
            "total_tokens": cat_tokens,
            "avg_tokens_per_word": avg_tpw,
            "avg_chars_per_token": avg_cpt,
            "unk_count": cat_unk,
            "unk_rate": round(cat_unk / max(1, cat_tokens) * 100.0, 2),
            "examples": item_details,
        }

        print(f"[{category}] Words: {cat_words} | Tokens: {cat_tokens} | Avg Tokens/Word: {avg_tpw} | Unk: {cat_unk}")

    audit_report["overall"] = {
        "total_words": total_words_all,
        "total_tokens": total_tokens_all,
        "total_chars": total_chars_all,
        "overall_avg_tokens_per_word": round(total_tokens_all / max(1, total_words_all), 2),
        "overall_avg_chars_per_token": round(total_chars_all / max(1, total_tokens_all), 2),
        "total_unk": total_unk_all,
        "unk_rate": round(total_unk_all / max(1, total_tokens_all) * 100.0, 3),
    }

    # Inspect corpus vocabulary coverage
    train_corpus_file = Path("data/cyclonelm/v0.2/domain_corpus/train_combined_corpus.txt")
    with open(train_corpus_file, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    corpus_enc = tokenizer.encode(corpus_text, add_bos=False, add_eos=False)
    corpus_tokens = corpus_enc["input_ids"]
    corpus_unk = sum(1 for tid in corpus_tokens if tid == tokenizer.unk_id)
    unique_token_ids = len(set(corpus_tokens))

    audit_report["corpus_coverage"] = {
        "corpus_total_tokens": len(corpus_tokens),
        "corpus_unique_tokens_used": unique_token_ids,
        "vocab_utilization_percent": round(unique_token_ids / vocab_size * 100.0, 1),
        "corpus_unk_tokens": corpus_unk,
        "corpus_unk_rate": round(corpus_unk / max(1, len(corpus_tokens)) * 100.0, 3),
    }
    print(f"\n[Corpus Coverage] Tokens: {len(corpus_tokens):,} | Unique IDs used: {unique_token_ids}/{vocab_size} ({audit_report['corpus_coverage']['vocab_utilization_percent']}%) | UNK rate: {audit_report['corpus_coverage']['corpus_unk_rate']}%")

    out_path = Path("reports/cyclonelm_tokenizer_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"Saved tokenizer audit report to {out_path}")

    return audit_report


if __name__ == "__main__":
    audit_tokenizer()
