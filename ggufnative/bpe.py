"""Byte-level BPE tokenizer read straight from a GGUF file (GPT-2 style, as used by Qwen2/Qwen3).

No tokenizer library: the vocabulary, merge ranks and special tokens all come from the model file,
so a model dropped into drock/models brings its own tokenizer. Text is split into words by the
model's pre-tokenizer pattern, each word's UTF-8 bytes are mapped to GPT-2's printable byte
alphabet, and adjacent pairs are merged by rank until no known merge applies.
"""
from __future__ import annotations

import regex

# Pre-tokenizer patterns by tokenizer.ggml.pre (from llama.cpp's LLAMA_VOCAB_PRE_TYPE_* table).
_PRE = {
    "qwen2": (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
              r"|\s*[\r\n]+|\s+(?!\S)|\s+"),
    # LLaMA-3's split, which differs from qwen2 only in grouping digits in threes. BitNet b1.58
    # 2B4T uses it (its tokenizer.json), although its GGUF carries no tokenizer.ggml.pre at all.
    "llama-bpe": (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                  r"|\s*[\r\n]+|\s+(?!\S)|\s+"),
}
# MiniCPM5-2B's GGUF names its own pre-tokenizer ("minicpm5") where the 1B names "llama-bpe". Checked
# rather than assumed (2026-10-22): for eight probe strings chosen to separate the two patterns, the
# llama-bpe split gives exactly the token count the reference runtime reports for the same raw text,
# including the digit cases that distinguish it from qwen2 ("1234567890" -> 4 tokens, threes, not 10;
# "ABCdef 007 3.14159" -> 9, not 14). qwen2 matched 4 of 8, llama-bpe 8 of 8.
_PRE["minicpm5"] = _PRE["llama-bpe"]
_CONTROL, _USER_DEFINED = 3, 4
# Words kept in the BPE cache. drock's prompts repeat the same system text, so a few thousand words
# cover almost every hit; clearing past this bound keeps memory flat in a long session.
_CACHE_MAX = 50_000


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible map from the 256 byte values to printable characters."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BPE:
    def __init__(self, tokens: list[str], merges: list[str], token_types: list[int], pre: str = "qwen2"):
        if pre not in _PRE:
            raise NotImplementedError(f"pre-tokenizer {pre!r} is not supported yet")
        self.tokens = tokens
        self.id_of = {t: i for i, t in enumerate(tokens)}
        self.ranks = {tuple(m.split(" ", 1)): i for i, m in enumerate(merges)}
        self.byte_enc = _bytes_to_unicode()
        self.byte_dec = {c: b for b, c in self.byte_enc.items()}
        self.pat = regex.compile(_PRE[pre])
        # Control and user-defined tokens (<|im_start|>, <think>, ...) are matched verbatim, never split.
        self.special = {t: i for i, (t, ty) in enumerate(zip(tokens, token_types)) if ty in (_CONTROL, _USER_DEFINED)}
        self.special_ids = set(self.special.values())
        self.special_re = regex.compile("|".join(regex.escape(s) for s in sorted(self.special, key=len, reverse=True)))
        self._cache: dict[str, list[int]] = {}
        self._pieces: dict[int, str] = {}

    def _bpe(self, word: str) -> list[int]:
        ids = self._cache.get(word)
        if ids is not None:
            return ids
        parts = list(word)
        while len(parts) > 1:
            best, at = None, -1
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if best is None:
                break
            parts[at:at + 2] = [parts[at] + parts[at + 1]]
        ids = [self.id_of[p] for p in parts]
        if len(self._cache) >= _CACHE_MAX:  # a long session sees unbounded distinct words (file contents)
            self._cache.clear()
        self._cache[word] = ids
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        out: list[int] = []
        for w in self.pat.findall(text):
            out += self._bpe("".join(self.byte_enc[b] for b in w.encode("utf-8")))
        return out

    def encode(self, text: str, special: bool = True) -> list[int]:
        """Token ids for `text`; with special=True, <|im_start|>-style tokens in the text are recognised."""
        if not special or not self.special:
            return self._encode_plain(text)
        ids, pos = [], 0
        for m in self.special_re.finditer(text):
            ids += self._encode_plain(text[pos:m.start()])
            ids.append(self.special[m.group()])
            pos = m.end()
        return ids + self._encode_plain(text[pos:])

    def token_bytes(self, i: int) -> bytes:
        t = self.tokens[i]
        if i in self.special_ids:
            return t.encode("utf-8")
        return bytes(self.byte_dec[c] if c in self.byte_dec else ord(c) & 0xFF for c in t)

    def decode(self, ids) -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", "replace")

    def piece(self, i: int) -> str:
        """One token's text ("\\ufffd" inside means it is part of a multi-byte character)."""
        p = self._pieces.get(i)
        if p is None:
            p = self._pieces[i] = self.token_bytes(i).decode("utf-8", "replace")
        return p
