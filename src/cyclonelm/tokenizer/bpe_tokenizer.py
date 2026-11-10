"""Byte-Pair Encoding (BPE) Tokenizer built from scratch for CycloneLM.

Features:
- Pure-Python BPE training algorithm with no external C-extensions.
- Deterministic tokenization preserving whitespace, punctuation, and domain terms.
- Built-in special tokens for conversational turns, tool calling, and padding.
- JSON serialization and deserialization.
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

DEFAULT_SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<tool_call>",
    "</tool_call>",
    "<tool_result>",
    "</tool_result>",
    "<user>",
    "</user>",
    "<assistant>",
    "</assistant>",
    "<system>",
    "</system>",
    "<evidence>",
    "</evidence>",
]


class BPETokenizer:
    """Byte-Pair Encoding (BPE) subword tokenizer for CycloneLM."""

    def __init__(
        self,
        special_tokens: Optional[Sequence[str]] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ) -> None:
        self.special_tokens = list(special_tokens or DEFAULT_SPECIAL_TOKENS)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Vocab mappings
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}

        # Initialize special tokens
        for idx, tok in enumerate(self.special_tokens):
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok

        # Pre-compiled regex for special tokens, sorted by length descending so longer tags match first
        self._special_pattern = re.compile(
            "(" + "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)) + ")"
        )

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    def _split_into_characters(self, word: str) -> List[str]:
        """Split a word into characters, prepending whitespace indicator for non-first words if needed."""
        return list(word)

    def train(
        self,
        corpus: Sequence[str],
        vocab_size: int = 1024,
        min_frequency: int = 2,
    ) -> None:
        """Train BPE merges on a corpus of text strings."""
        if vocab_size <= len(self.special_tokens):
            raise ValueError(
                f"vocab_size {vocab_size} must be larger than number of special tokens ({len(self.special_tokens)})"
            )

        # 1. Frequency of pre-tokenized words
        word_freqs: Dict[Tuple[str, ...], int] = {}
        # Base characters set - initialize with all printable ASCII characters, space, tab, newline
        base_chars: Set[str] = set(string.printable)

        for text in corpus:
            # Segment text into words, numbers, punctuation and spaces
            tokens = re.findall(r"\w+|[^\w\s]|\s+", text, re.UNICODE)
            for tok in tokens:
                if tok in self.special_tokens:
                    continue
                char_tuple = tuple(tok)
                word_freqs[char_tuple] = word_freqs.get(char_tuple, 0) + 1
                base_chars.update(char_tuple)

        # 2. Add base characters to vocab
        for ch in sorted(base_chars):
            if ch not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[ch] = new_id
                self.id_to_token[new_id] = ch

        # 3. Iterative BPE merges
        num_merges = vocab_size - len(self.token_to_id)
        merges: List[Tuple[str, str]] = []

        for _ in range(max(0, num_merges)):
            pairs: Dict[Tuple[str, str], int] = {}
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)  # type: ignore[arg-type]
            if pairs[best_pair] < min_frequency and len(merges) > 50:
                # Stop if no frequent pairs remain
                break

            merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[new_token] = new_id
                self.id_to_token[new_id] = new_token

            # Update word frequencies with merge
            new_word_freqs: Dict[Tuple[str, ...], int] = {}
            for word, freq in word_freqs.items():
                i = 0
                new_word = []
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs

        self.merges = merges
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

    def _tokenize_word(self, word: str) -> List[str]:
        """Apply BPE merges to a single word/symbol."""
        if not word:
            return []
        symbols = list(word)
        if len(symbols) <= 1:
            return symbols

        while len(symbols) > 1:
            # Find the best merge pair among adjacent symbols
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            mergeable_pairs = [
                (pair, self.bpe_ranks[pair])
                for pair in pairs
                if pair in self.bpe_ranks
            ]
            if not mergeable_pairs:
                break

            best_pair = min(mergeable_pairs, key=lambda x: x[1])[0]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_symbols.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        pad_to_max_length: bool = False,
    ) -> Dict[str, List[int]]:
        """Encode text to token ids with optional special tokens and padding."""
        segments = self._special_pattern.split(text)
        token_ids: List[int] = []

        if add_bos:
            token_ids.append(self.bos_id)

        for segment in segments:
            if not segment:
                continue
            if segment in self.token_to_id:
                token_ids.append(self.token_to_id[segment])
            else:
                words = re.findall(r"\w+|[^\w\s]|\s+", segment, re.UNICODE)
                for w in words:
                    subwords = self._tokenize_word(w)
                    for sw in subwords:
                        token_ids.append(self.token_to_id.get(sw, self.unk_id))

        if add_eos:
            token_ids.append(self.eos_id)

        # Truncate if requested
        if max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        attention_mask = [1] * len(token_ids)

        # Pad if requested
        if pad_to_max_length and max_length is not None:
            pad_len = max_length - len(token_ids)
            if pad_len > 0:
                token_ids.extend([self.pad_id] * pad_len)
                attention_mask.extend([0] * pad_len)

        return {"input_ids": token_ids, "attention_mask": attention_mask}

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode sequence of token ids back into text string."""
        tokens = []
        for tid in token_ids:
            tok = self.id_to_token.get(tid, self.unk_token)
            if skip_special_tokens and tok in self.special_tokens:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def save(self, filepath: str | Path) -> None:
        """Serialize tokenizer state to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "special_tokens": self.special_tokens,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> BPETokenizer:
        """Load tokenizer state from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(
            special_tokens=data.get("special_tokens"),
            unk_token=data.get("unk_token", "<unk>"),
            pad_token=data.get("pad_token", "<pad>"),
            bos_token=data.get("bos_token", "<bos>"),
            eos_token=data.get("eos_token", "<eos>"),
        )
        tok.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        tok.merges = [tuple(m) for m in data.get("merges", [])]  # type: ignore[misc]
        tok.bpe_ranks = {pair: i for i, pair in enumerate(tok.merges)}
        tok._special_pattern = re.compile(
            "(" + "|".join(re.escape(t) for t in sorted(tok.special_tokens, key=len, reverse=True)) + ")"
        )
        return tok
