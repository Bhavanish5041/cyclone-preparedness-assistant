"""Unit tests for CycloneLM BPE Tokenizer."""

import json
from pathlib import Path

import pytest

from src.cyclonelm.tokenizer.bpe_tokenizer import DEFAULT_SPECIAL_TOKENS, BPETokenizer


@pytest.fixture
def sample_corpus():
    return [
        "Tropical cyclones in the Arabian Sea impact Coastal Karnataka.",
        "Analytical Hierarchy Process evaluates composite community preparedness.",
        "Operational priority bands allocate emergency disaster relief.",
        "Ullal in Dakshina Kannada exhibits severe storm surge exposure.",
    ]


def test_tokenizer_initialization():
    tok = BPETokenizer()
    assert tok.vocab_size == len(DEFAULT_SPECIAL_TOKENS)
    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.bos_id == 2
    assert tok.eos_id == 3


def test_tokenizer_train_and_encode_decode(sample_corpus):
    tok = BPETokenizer()
    tok.train(sample_corpus, vocab_size=180, min_frequency=1)
    assert tok.vocab_size <= 180

    text = "Tropical cyclones in Karnataka"
    enc = tok.encode(text, add_bos=True, add_eos=True)
    assert enc["input_ids"][0] == tok.bos_id
    assert enc["input_ids"][-1] == tok.eos_id

    dec = tok.decode(enc["input_ids"], skip_special_tokens=True)
    assert dec == text


def test_tokenizer_special_tokens_preservation(sample_corpus):
    tok = BPETokenizer()
    tok.train(sample_corpus, vocab_size=200)

    tool_text = '<tool_call>{"tool": "lookup_community", "parameters": {"query": "Ullal"}}</tool_call>'
    enc = tok.encode(tool_text)
    dec = tok.decode(enc["input_ids"], skip_special_tokens=False)
    assert dec == tool_text


def test_tokenizer_padding_and_truncation(sample_corpus):
    tok = BPETokenizer()
    tok.train(sample_corpus, vocab_size=150)

    text = "Short text"
    enc = tok.encode(text, max_length=16, pad_to_max_length=True)
    assert len(enc["input_ids"]) == 16
    assert len(enc["attention_mask"]) == 16
    assert enc["input_ids"].count(tok.pad_id) > 0
    assert enc["attention_mask"].count(0) > 0

    # Truncation check
    enc_trunc = tok.encode(sample_corpus[0], max_length=5, pad_to_max_length=False)
    assert len(enc_trunc["input_ids"]) == 5


def test_tokenizer_save_load(sample_corpus, tmp_path):
    tok = BPETokenizer()
    tok.train(sample_corpus, vocab_size=160)

    save_file = tmp_path / "test_vocab.json"
    tok.save(save_file)
    assert save_file.exists()

    reloaded = BPETokenizer.load(save_file)
    assert reloaded.vocab_size == tok.vocab_size

    test_str = "Coastal Karnataka preparedness"
    enc1 = tok.encode(test_str)["input_ids"]
    enc2 = reloaded.encode(test_str)["input_ids"]
    assert enc1 == enc2


def test_all_special_tags_atomic_representation():
    tok = BPETokenizer()
    expected_tags = [
        "<pad>", "<unk>", "<bos>", "<eos>",
        "<tool_call>", "</tool_call>",
        "<tool_result>", "</tool_result>",
        "<user>", "</user>",
        "<assistant>", "</assistant>",
        "<system>", "</system>",
        "<evidence>", "</evidence>",
    ]
    for tag in expected_tags:
        assert tag in tok.token_to_id
        enc = tok.encode(tag)
        assert len(enc["input_ids"]) == 1, f"Tag {tag} is not atomic: encoded to {enc['input_ids']}"
        assert tok.decode(enc["input_ids"]) == tag


def test_tokenizer_roundtrip_with_tool_tags():
    v02_path = Path("models/cyclonelm/v0.2/tokenizer/vocab.json")
    if v02_path.exists():
        tok = BPETokenizer.load(v02_path)
    else:
        tok = BPETokenizer()
        corpus = [
            "<user>Look up Ullal in Dakshina Kannada</user>",
            "<assistant><tool_call>{\"tool\": \"lookup_community\", \"parameters\": {\"query\": \"Ullal\", \"district\": \"Dakshina Kannada\"}}</tool_call></assistant>",
            "<tool_result>{\"community_name\": \"Ullal (TMC)\", \"preparedness_score\": 60.44}</tool_result>",
            "<assistant><evidence>Ullal (TMC) AHP score: 60.44 (High)</evidence>Ullal exhibits high preparedness.</assistant>",
        ]
        tok.train(corpus, vocab_size=256, min_frequency=1)

    complex_text = (
        "<user>Look up Ullal in Dakshina Kannada</user>"
        "<assistant><tool_call>{\"tool\": \"lookup_community\", \"parameters\": {\"query\": \"Ullal\", \"district\": \"Dakshina Kannada\"}}</tool_call></assistant>"
        "<tool_result>{\"community_name\": \"Ullal (TMC)\", \"preparedness_score\": 60.44}</tool_result>"
        "<assistant><evidence>Ullal (TMC) AHP score: 60.44 (High)</evidence>Ullal exhibits high preparedness.</assistant>"
    )
    enc = tok.encode(complex_text)
    dec = tok.decode(enc["input_ids"], skip_special_tokens=False)
    assert dec == complex_text


def test_v01_checkpoint_incompatible_with_v02_vocabulary():
    from src.cyclonelm.inference.engine import CycloneLMEngine

    v01_ckpt = Path("models/cyclonelm/v0.1/checkpoints/best_model.pt")
    v02_tok = Path("models/cyclonelm/v0.2/tokenizer/vocab.json")

    if v01_ckpt.exists() and v02_tok.exists():
        with pytest.raises(ValueError, match="Incompatible vocabulary"):
            CycloneLMEngine(
                checkpoint_path=v01_ckpt,
                tokenizer_path=v02_tok,
                device="cpu",
            )


def test_tokenizer_roundtrip_actual_training_text():
    """Verify lossless round-trip encoding/decoding on actual CycloneLM training text."""
    v02_tok_path = Path("models/cyclonelm/v0.2/tokenizer/vocab.json")
    v01_tok_path = Path("models/cyclonelm/v0.1/tokenizer/vocab.json")

    # Test v0.2 tokenizer with v0.2 training corpus
    if v02_tok_path.exists():
        tok_v02 = BPETokenizer.load(v02_tok_path)
        corpus_path = Path("data/cyclonelm/v0.2/domain_corpus/train_combined_corpus.txt")
        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()][:50]
            for line in lines:
                encoded = tok_v02.encode(line, add_bos=False, add_eos=False)
                decoded = tok_v02.decode(encoded["input_ids"], skip_special_tokens=False)
                assert decoded == line, f"Roundtrip failed for line: {line[:50]}..."

        # Test v0.2 instruction text
        inst_path = Path("data/cyclonelm/v0.2/instructions/train.jsonl")
        if inst_path.exists():
            with open(inst_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()][:25]
            for r in records:
                text = f"<user>{r['instruction']}</user><assistant>{r['response']}</assistant>"
                encoded = tok_v02.encode(text)
                decoded = tok_v02.decode(encoded["input_ids"], skip_special_tokens=False)
                assert decoded == text, f"Roundtrip failed for instruction: {text[:50]}..."

    # Test v0.1 tokenizer with v0.1 training corpus
    if v01_tok_path.exists():
        tok_v01 = BPETokenizer.load(v01_tok_path)
        v01_corpus_path = Path("data/cyclonelm/pretraining/cyclonelm_domain_corpus.txt")
        if v01_corpus_path.exists():
            with open(v01_corpus_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()][:50]
            for line in lines:
                encoded = tok_v01.encode(line, add_bos=False, add_eos=False)
                decoded = tok_v01.decode(encoded["input_ids"], skip_special_tokens=False)
                assert decoded == line, f"Roundtrip failed for v0.1 line: {line[:50]}..."


