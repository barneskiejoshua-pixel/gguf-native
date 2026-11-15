# gguf-native

A GGUF model file in, text out, in Python and numpy. No llama.cpp, no runtime underneath.

- `gguf.py` reads GGUF v3 lazily over mmap.
- `bpe.py` builds the byte-level BPE tokenizer from the vocabulary inside the model file.
- `native.py` dequantizes Q4_K, Q6_K, Q8_0 and BitNet's I2_S, runs the forward pass for qwen3,
  plain llama blocks (ChatML templates) and bitnet-b1.58, and keeps a KV cache that reuses the
  longest shared prompt prefix. The BitNet path has an int8 ternary kernel compiled with numba.
- `scaffold.py` does schema-constrained decoding by writing the JSON itself and letting the model
  fill only the blanks.

    pip install numpy numba
    python check.py path/to/qwen3-0.6b.gguf

It is slow next to llama.cpp. The point is that the whole path from file bytes to tokens is
readable in about 1,400 lines. The BitNet decode was compared against Microsoft's release: all 210
ternary tensors gave identical codes.

`native.engine()` looks for models in a `models` folder beside the package; `native.Engine(path)` takes any path.
