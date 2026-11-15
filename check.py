"""Standalone check: prints got versus expected for each case and exits 1 on any mismatch."""
import sys

BAD = 0


def same(name, got, expected):
    global BAD
    ok = got == expected
    BAD += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r} expected {expected!r}")

from pathlib import Path

from ggufnative import gguf, native

if len(sys.argv) < 2:
    print("usage: python check.py path/to/model.gguf   (qwen3, a ChatML llama-family model, or bitnet-b1.58)")
    sys.exit(2)
path = Path(sys.argv[1]).resolve()
g = gguf.load(path)
print("architecture:", g.get("general.architecture"), "| tensors:", len(g.tensors))
same("file parses with tensors", len(g.tensors) > 0, True)
eng = native.Engine(path, 1024)
text, meta = eng.chat([{"role": "user", "content": "What is the capital of France? One word."}], max_tokens=12, temperature=0.0)
print("reply:", text.strip()[:80], "|", {k: meta[k] for k in list(meta)[:6]})
same("reply names Paris", "paris" in text.lower(), True)

sys.exit(1 if BAD else 0)
