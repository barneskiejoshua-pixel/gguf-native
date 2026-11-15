"""drock's own inference engine: a GGUF model file in, text out. Pure Python + numpy.

Reverse-engineered from the published model architectures and ggml's GGUF/k-quant layouts, not a
wrapper around any runtime: the k-quant dequantization, the transformer forward pass, the KV cache
with prompt-prefix reuse, the tokenizer (bpe.py) and schema-constrained decoding (scaffold.py) are
all here. drock then needs nothing but Python, numpy and a model file in drock/models, so it is
portable: copy the folder and it runs.

Supported architectures: qwen3, llama (a plain llama block: MiniCPM5, Mistral, Llama 3 -- ChatML
templates only), bitnet-b1.58 (Microsoft's BitNet b1.58 2B4T, ternary weights).

BitNet exists here because bitnet.cpp, its official runtime, is a native build that Smart App Control
blocks on this machine, and Ollama cannot load its I2_S weights. The format and forward pass were
read from bitnet.cpp (src/ggml-bitnet-mad.cpp: quantize_i2_s; its llama.cpp fork: build_bitnet_158)
and Hugging Face's BitNet modeling code. research/model-bakeoff/bitnet_reference.py checks it against
Microsoft's deployment release (microsoft/bitnet-b1.58-2B-4T, the source bitnet.cpp's GGUF was
converted from): all 210 ternary tensors decode to identical codes, and an independent float
forward pass agrees on 32/32 greedy tokens for each test prompt (logit cosine >= 0.9997). The bf16
master weights are a different quantization (up to ~8% of codes differ) and are not the reference.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from . import gguf
from .bpe import BPE

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_CTX = 4096


# ---------- dequantization: ggml block layouts (ggml-quants.c), vectorised ----------

def _f16(col: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(col).view(np.float16).astype(np.float32).reshape(-1)


def dequant_q4_k(raw: np.ndarray) -> np.ndarray:
    """Q4_K: 256 weights per 144-byte block = d, dmin (f16) + 12 bytes of packed 6-bit
    scales/mins for 8 sub-blocks of 32 + 128 bytes of 4-bit quants. w = d*sc*q - dmin*m."""
    b = raw.reshape(-1, 144)
    d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
    s = b[:, 4:16]
    sc = np.empty((len(b), 8), np.float32)
    mn = np.empty((len(b), 8), np.float32)
    sc[:, :4] = s[:, 0:4] & 63
    mn[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    q = b[:, 16:144].reshape(-1, 4, 32)
    q = np.stack([q & 0xF, q >> 4], axis=2).reshape(-1, 8, 32).astype(np.float32)  # low nibbles, then high
    return ((d[:, None] * sc)[:, :, None] * q - (dmin[:, None] * mn)[:, :, None]).reshape(-1)


def dequant_q6_k(raw: np.ndarray) -> np.ndarray:
    """Q6_K: 256 weights per 210-byte block = 128 bytes low 4 bits + 64 bytes high 2 bits +
    16 int8 scales (one per 16 weights) + d (f16). w = d * scale * (q - 32)."""
    b = raw.reshape(-1, 210)
    ql = b[:, 0:128].reshape(-1, 2, 64).astype(np.int16)
    qh = b[:, 128:192].reshape(-1, 2, 32).astype(np.int16)
    sc = np.ascontiguousarray(b[:, 192:208]).view(np.int8).reshape(-1, 2, 8).astype(np.float32)
    d = _f16(b[:, 208:210])
    la, lb = ql[:, :, :32], ql[:, :, 32:]
    q = np.concatenate([(la & 0xF) | ((qh & 3) << 4),
                        (lb & 0xF) | (((qh >> 2) & 3) << 4),
                        (la >> 4) | (((qh >> 4) & 3) << 4),
                        (lb >> 4) | (((qh >> 6) & 3) << 4)], axis=2).astype(np.float32) - 32
    return (q.reshape(-1, 2, 8, 16) * sc[:, :, :, None] * d[:, None, None, None]).reshape(-1)


def ternary_i2_s(raw: np.ndarray, n: int) -> tuple[np.ndarray, np.float32]:
    """I2_S (bitnet.cpp, x86 layout): the whole tensor is one ternary matrix times one scale.
    n/4 bytes of 2-bit codes, then a float32 scale (then padding to 32 bytes). Codes 0/1/2 mean
    -1/0/+1. Codes are packed per block of 128 weights into 32 bytes: byte p holds weight p in bits
    7-6, weight 32+p in bits 5-4, 64+p in 3-2 and 96+p in 1-0 (so a SIMD lane shifts, masks and
    lines up with 32 consecutive activations). The scale is the mean |w| the model was trained with.
    Returns the -1/0/+1 values as int8 (flat, n of them) and the scale."""
    b = raw[:n // 4].reshape(-1, 32)
    q = np.stack([b >> 6, (b >> 4) & 3, (b >> 2) & 3, b & 3], axis=1).reshape(-1)
    return q.view(np.int8) - np.int8(1), raw[n // 4:n // 4 + 4].view(np.float32)[0]


def dequant_i2_s(raw: np.ndarray, n: int) -> np.ndarray:
    t, scale = ternary_i2_s(raw, n)
    return t.astype(np.float32) * scale


def dequant_q8_0(raw: np.ndarray) -> np.ndarray:
    """Q8_0: 32 weights per 34-byte block = d (f16) + 32 int8; w = d * q."""
    b = raw.reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    return (d * b[:, 2:].view(np.int8).astype(np.float32)).reshape(-1)


_DEQUANT = {"Q4_K": dequant_q4_k, "Q6_K": dequant_q6_k, "Q8_0": dequant_q8_0}


def load_tensor(g: gguf.GGUF, name: str) -> np.ndarray:
    """A tensor as float32, in numpy's row-major order (GGUF lists dims innermost-first)."""
    t = g.tensors[name]
    raw = np.frombuffer(g.mm, dtype=np.uint8, count=t.nbytes, offset=g.data_start + t.offset)
    n = math.prod(t.shape)
    kind = t.dtype.name
    if kind == "F32":
        a = raw.view(np.float32)[:n].copy()
    elif kind == "F16":
        a = raw.view(np.float16)[:n].astype(np.float32)
    elif kind == "I2_S":
        a = dequant_i2_s(raw, n)
    elif kind in _DEQUANT:
        a = _DEQUANT[kind](raw)[:n]
    else:
        raise NotImplementedError(f"{name}: tensor type {kind} is not supported yet")
    return a.reshape(tuple(reversed(t.shape)))


# ---------- the model ----------

def _rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * w


def chatml(messages: list[dict]) -> str:
    """Qwen's ChatML prompt, with an empty think block: Qwen3's documented way to switch thinking off
    (drock's reasoning lives in the action's "reason" field)."""
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return s + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Qwen3:
    """Qwen3 decoder: RMSNorm, GQA attention with per-head q/k RMSNorm and NEOX RoPE, SwiGLU MLP."""

    ARCH = "qwen3"
    template = staticmethod(chatml)

    # Research hook (research/flow): a callable trace(stage, layer, value) or None. When set, forward()
    # hands out the residual stream as it flows through the layers, so its dynamics can be measured
    # without copying the forward pass: "embed" (-1, the token embeddings (T, d)), then per layer li
    # "probs" (the attention distribution (kv, g, T, S) after softmax), "attn" (x after the attention
    # residual add) and "layer" (x after the MLP residual add, the layer's output), then "final"
    # (n_layer, the normed last-token state (1, d) the head reads). Every value is a fresh array the
    # forward never touches again, so a trace may keep it. None costs nothing: one attribute read per
    # stage, and the arithmetic is unchanged (tests/test_native_trace.py holds the logits bit-identical).
    trace = None

    def __init__(self, path: Path, ctx: int = DEFAULT_CTX, g: gguf.GGUF | None = None):
        t0 = time.monotonic()
        g = g or gguf.load(path)
        arch = g.get("general.architecture")
        if arch != self.ARCH:
            g.close()
            raise NotImplementedError(f"architecture {arch!r} is not {self.ARCH}")
        p = lambda k, default=None: g.get(f"{arch}.{k}", default)
        self.name = g.get("general.name") or path.stem
        self.n_layer, self.d = p("block_count"), p("embedding_length")
        self.n_head, self.n_kv = p("attention.head_count"), p("attention.head_count_kv")
        self.hd = p("attention.key_length") or self.d // self.n_head
        self.eps = p("attention.layer_norm_rms_epsilon", 1e-6)
        self.ctx = min(ctx, p("context_length", ctx))
        self.tok = BPE(g.get("tokenizer.ggml.tokens"), g.get("tokenizer.ggml.merges"),
                       g.get("tokenizer.ggml.token_type"), g.get("tokenizer.ggml.pre") or "qwen2")
        self.eos = {g.get("tokenizer.ggml.eos_token_id"), self.tok.special.get("<|im_end|>"),
                    self.tok.special.get("<|endoftext|>")} - {None}
        W = lambda n: load_tensor(g, n)
        self.embd = W("token_embd.weight")                                   # (vocab, d)
        self.out = W("output.weight") if "output.weight" in g.tensors else self.embd
        self.out_norm = W("output_norm.weight")
        self.layers = []
        for i in range(self.n_layer):
            b = f"blk.{i}."
            # Per-head q/k RMSNorm is qwen3's; a plain llama block (MiniCPM5, Mistral, Llama 3) has no
            # such tensor, so the weight is read when the file carries it and left None when it does not.
            # forward() skips the norm for a None, which is what makes one decoder serve both shapes.
            self.layers.append({k: (W(b + v) if b + v in g.tensors else None) for k, v in (
                ("attn_norm", "attn_norm.weight"), ("q", "attn_q.weight"), ("k", "attn_k.weight"),
                ("v", "attn_v.weight"), ("o", "attn_output.weight"), ("q_norm", "attn_q_norm.weight"),
                ("k_norm", "attn_k_norm.weight"), ("ffn_norm", "ffn_norm.weight"), ("gate", "ffn_gate.weight"),
                ("up", "ffn_up.weight"), ("down", "ffn_down.weight"))})
            for k in ("attn_norm", "q", "k", "v", "o", "ffn_norm", "gate", "up", "down"):
                if self.layers[-1][k] is None:
                    g.close()
                    raise NotImplementedError(f"{b}{k}: this {arch} file has no such tensor")
        g.close()  # every tensor above is a copy, so the file can be released
        inv = p("rope.freq_base", 10000.0) ** (-np.arange(0, self.hd, 2, dtype=np.float64) / self.hd)
        ang = np.outer(np.arange(self.ctx), inv)
        self.cos, self.sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        self._alloc_cache()
        self.load_seconds = round(time.monotonic() - t0, 1)

    def _alloc_cache(self) -> None:
        """The KV cache: float32, laid out (layer, kv head, position, hd).

        It used to be float16 in (layer, position, kv head, hd) order, which halved its memory but made
        attention convert and transpose the WHOLE cache, every layer, every token. Measured at BitNet's
        shape (30 layers, 5 kv heads, hd 128) with 2500 tokens cached: 420 ms/token of attention, against
        a whole-token budget of ~140 ms. In this layout each kv head's first S positions are one
        contiguous block that BLAS reads in place: 57 ms/token, same numbers to 1e-5. np.zeros commits
        pages only as positions are written, so the doubled size costs RAM only for context in use."""
        self.kc = np.zeros((self.n_layer, self.n_kv, self.ctx, self.hd), np.float32)
        self.vc = np.zeros_like(self.kc)
        self.cached: list[int] = []  # the token ids whose K/V are in the cache, in order

    def _rope(self, x: np.ndarray, pos: slice) -> np.ndarray:
        c, s = self.cos[pos][:, None, :], self.sin[pos][:, None, :]
        h = self.hd // 2
        x1, x2 = x[..., :h], x[..., h:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)

    def forward(self, ids: list[int]) -> np.ndarray:
        """Append `ids` to the cache and return the next-token logits after the last one."""
        P, T = len(self.cached), len(ids)
        if P + T > self.ctx:
            raise ValueError(f"context full ({P + T} > {self.ctx} tokens)")
        x = self.embd[ids]                                                     # (T, d)
        S, g = P + T, self.n_head // self.n_kv
        mask = np.where(np.arange(S)[None, :] <= (P + np.arange(T))[:, None], 0.0, -np.inf).astype(np.float32)
        scale = 1.0 / math.sqrt(self.hd)
        pos = slice(P, P + T)
        trace = self.trace
        if trace is not None:
            trace("embed", -1, x)
        for li, L in enumerate(self.layers):
            h = _rms(x, L["attn_norm"], self.eps)
            q = (h @ L["q"].T).reshape(T, self.n_head, self.hd)
            k = (h @ L["k"].T).reshape(T, self.n_kv, self.hd)
            if L["q_norm"] is not None:      # qwen3 normalises each head; a plain llama block does not
                q = _rms(q, L["q_norm"], self.eps)
                k = _rms(k, L["k_norm"], self.eps)
            v = (h @ L["v"].T).reshape(T, self.n_kv, self.hd)
            o = self._attend(li, q, k, v, pos, S, g, mask, scale)
            x = x + o @ L["o"].T
            if trace is not None:
                trace("attn", li, x)
            h = _rms(x, L["ffn_norm"], self.eps)
            a, u = h @ L["gate"].T, h @ L["up"].T
            # SwiGLU. exp is clipped because exp(-a) overflows float32 below about -88, and a
            # clipped value is already 0 to within float32 anyway.
            x = x + ((a / (1.0 + np.exp(np.clip(-a, -60.0, 60.0)))) * u) @ L["down"].T
            if trace is not None:
                trace("layer", li, x)
        self.cached += list(ids)
        xn = _rms(x[-1:], self.out_norm, self.eps)
        if trace is not None:
            trace("final", self.n_layer, xn)
        return (xn @ self.out.T)[0]

    def _attend(self, li: int, q, k, v, pos: slice, S: int, g: int, mask, scale: float) -> np.ndarray:
        """RoPE q and k, append k/v to layer li's cache, and return causal GQA attention (T, heads*hd)."""
        T = q.shape[0]
        self.kc[li, :, pos] = self._rope(k, pos).transpose(1, 0, 2)
        self.vc[li, :, pos] = v.transpose(1, 0, 2)
        q = self._rope(q, pos).reshape(T, self.n_kv, g, self.hd).transpose(1, 2, 0, 3)  # (kv, g, T, hd)
        K = self.kc[li, :, :S].transpose(0, 2, 1)[:, None]                               # (kv, 1, hd, S), a view
        V = self.vc[li, :, :S][:, None]                                                  # (kv, 1, S, hd), a view
        att = q @ K * scale + mask
        att = np.exp(att - att.max(axis=-1, keepdims=True))
        att /= att.sum(axis=-1, keepdims=True)
        if self.trace is not None:
            self.trace("probs", li, att)
        return (att @ V).transpose(2, 0, 1, 3).reshape(T, self.n_head * self.hd)

    def prime(self, ids: list[int], chunk: int = 256, check=None) -> np.ndarray:
        """Make the cache hold exactly `ids`, reusing the longest prefix already cached (drock's
        prompts repeat the same system text every step), and return the next-token logits."""
        common = 0
        for a, b in zip(self.cached, ids):
            if a != b:
                break
            common += 1
        common = min(common, len(ids) - 1)  # always recompute the last token to get its logits
        self.cached = self.cached[:common]
        logits = None
        for s in range(common, len(ids), chunk):
            if check is not None:
                check()
            logits = self.forward(ids[s:s + chunk])
        return logits


class Llama(Qwen3):
    """A plain llama-architecture decoder: the same block as qwen3 without the per-head q/k norm.

    `general.architecture = "llama"` is what MiniCPM5 (openbmb/MiniCPM5-1B, Apache-2.0,
    `LlamaForCausalLM`), Mistral and Llama 3 GGUFs all declare, so one class covers them: the shapes,
    the RoPE base, the head dimension and the vocabulary all come from the file, and the forward pass
    is Qwen3's with the norms absent.

    The prompt format comes from the file too (`tokenizer.chat_template`), not from a table keyed by
    model name. Only ChatML is read here, because that is the one drock's scaffolded decoder has been
    checked against; another family's template raises rather than silently prompting the model wrong.
    """

    ARCH = "llama"

    def _rope(self, x: np.ndarray, pos: slice) -> np.ndarray:
        """Pair-wise RoPE (llama.cpp's LLAMA_ROPE_TYPE_NORM): dimensions 2i and 2i+1 rotate together.

        qwen3 rotates halves instead (NEOX). The difference is not cosmetic and it is not a choice:
        convert_hf_to_gguf.py PERMUTES a llama model's q and k weights so that pair-wise rotation
        reproduces what HuggingFace's rotate_half computes, and leaves qwen's unpermuted. Rotating
        halves here produced fluent but different text -- the top-5 next tokens overlapped the
        reference runtime's and the order was wrong (2026-10-21)."""
        c, s = self.cos[pos][:, None, :], self.sin[pos][:, None, :]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x1 * c - x2 * s
        out[..., 1::2] = x2 * c + x1 * s
        return out

    def __init__(self, path: Path, ctx: int = DEFAULT_CTX, g: gguf.GGUF | None = None):
        g = g or gguf.load(path)
        tpl = g.get("tokenizer.chat_template") or ""
        toks = g.get("tokenizer.ggml.tokens") or []
        if "<|im_start|>" not in tpl:
            g.close()
            raise NotImplementedError(f"{Path(path).name}: only a ChatML chat template is supported "
                                      f"(this file's template is a different family's)")
        # The template emits the BOS token itself when the tokenizer does not add it (MiniCPM5: "<s>",
        # add_bos_token false). Taking it from the file keeps the prompt byte-identical to the one the
        # model was trained and served with.
        bos_id = g.get("tokenizer.ggml.bos_token_id")
        head = "" if g.get("tokenizer.ggml.add_bos_token") or bos_id is None or "bos_token" not in tpl \
            else str(toks[bos_id])
        # An empty think block switches hybrid thinking off, as the template's enable_thinking=false
        # branch writes it; drock's reasoning lives in the action's "reason" field.
        think = "<think>\n\n</think>\n\n" if "<think>" in tpl else ""
        super().__init__(path, ctx, g)
        self.template = lambda ms: (head + "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                                                   for m in ms) + f"<|im_start|>assistant\n{think}")


def bitnet_chat(messages: list[dict]) -> str:
    """BitNet b1.58 2B4T's own template (its tokenizer_config.json, not the one embedded in the GGUF):
    "Role: content<|eot_id|>" per message, after a BOS, then "Assistant: "."""
    s = "".join(f"{m['role'].capitalize()}: {m['content'].strip()}<|eot_id|>" for m in messages)
    return "<|begin_of_text|>" + s + "Assistant: "


def _act_int8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BitNet's activation quantization, which the model was trained through: every ternary matmul
    sees its input rounded to int8 per token (row), scaled by that row's absolute maximum.
    Returns the int8 values and the per-row factor s (the real value is q / s)."""
    s = (127.0 / np.maximum(np.abs(x).max(axis=-1, keepdims=True), 1e-5)).astype(np.float32)
    return np.clip(np.round(x * s), -128, 127).astype(np.int8), s


def _act_float(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """No activation quantization: the input as float32 with factor 1, so q / s is x itself. The
    research arm DROCK_BITNET_ACT=float (research/flow): what the ternary weights compute when the
    int8 rounding the model was trained through is taken away. Numpy path only (the kernel is int8)."""
    return x.astype(np.float32, copy=False), np.float32(1.0)


def _act_quant(x: np.ndarray) -> np.ndarray:
    q, s = _act_int8(x)
    return q / s


_KERNEL = None


def _ternary_kernel():
    """The int8 x ternary matmul, JIT-compiled by numba when it is installed, else None.

    Why a kernel at all: decode is memory-bound, and as float32 BitNet's 2B ternary weights are
    8.3 GB read per token (~3 tok/s here). numpy has no int8 matmul, and converting int8 to float32
    per call costs more than it saves (measured 5x slower). Every signed runtime on this machine was
    tried: torch's CPU _int_mm, onnxruntime's MatMulInteger (s8s8 and u8s8) and MatMulNBits (4- and
    2-bit); none decoded faster than float32 BLAS. numba compiles in memory, so there is no
    unsigned binary on disk for Smart App Control to block, and the arithmetic is BitNet's own:
    int8 activations times -1/0/+1, summed exactly in int32.

    Measured (fused per-layer shapes x 30 layers, i5-8400, 6 threads): T=1 0.07 s vs 0.29-0.35 s
    for float32 BLAS; T=256 ~7 s vs ~2-5 s, so prefill is slower and decode ~4-5x faster, with a
    quarter of the RAM. DROCK_NATIVE_KERNEL=numpy forces the float32 path."""
    global _KERNEL
    if _KERNEL is None:
        import os
        _KERNEL = False
        if os.environ.get("DROCK_NATIVE_KERNEL", "").lower() != "numpy":
            try:
                from numba import njit, prange
            except ImportError:
                return None

            @njit(parallel=True, fastmath=True, cache=True)
            def tmm(xq, w, out):
                N, K = w.shape
                for n in prange(N):
                    for t in range(xq.shape[0]):
                        acc = np.int32(0)
                        for k in range(K):
                            acc += np.int32(xq[t, k]) * np.int32(w[n, k])
                        out[t, n] = acc
            _KERNEL = tmm
    return _KERNEL or None


def sylvester(B: int) -> np.ndarray:
    """The B x B Sylvester Hadamard matrix of +-1 (B a power of two): symmetric, S @ S = B * I."""
    S = np.ones((1, 1), np.float32)
    while S.shape[0] < B:
        S = np.block([[S, S], [S, -S]])
    return S


def hadamard_blocks(x: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Rows of x (N, K) times the block-diagonal orthonormal Hadamard H = S / sqrt(B), B = len(S),
    K a multiple of B. H is symmetric and H @ H = I, so (W H)(H x) = W x for any W."""
    N, K = x.shape
    B = S.shape[0]
    return (x.astype(np.float32, copy=False).reshape(N, K // B, B) @ S).reshape(N, K) * np.float32(1.0 / math.sqrt(B))


class _Ternary:
    """Ternary projections that share an input, fused into one matrix (q|k|v, gate|up), each part
    keeping its own scale. Held as int8 for the kernel, or as float32 for BLAS without it.

    rot=B (research/flow Phase 4.5, DROCK_BITNET_ROT=down): rotate the input by the block-diagonal
    Hadamard H = S / sqrt(B) before it is int8-quantized, and fold the same H into the weights, so
    W x = (W H)(H x) exactly and only the rounding changes. The down projection's input has absmax/rms
    ~35 (a few huge channels set every row's int8 step); H spreads each block's mass over all B
    entries. W H is no longer ternary: W @ S is an integer in [-B, B], held as int16 with scale / sqrt(B)."""

    def __init__(self, parts: list[tuple[np.ndarray, np.float32]], kernel, act=None, rot: int = 0):
        self.kernel = kernel
        self.act = act or _act_int8     # the activation quantizer: x -> (q, s) with x ~ q / s
        self.rot, self.S = rot, None
        if rot:
            if any(t.shape[1] % rot for t, _ in parts):
                raise ValueError(f"rot={rot} does not divide the input width {parts[0][0].shape[1]}")
            self.S = sylvester(rot)
            # float32 BLAS is exact here (integers up to B) and far faster than numpy's integer matmul
            parts = [((t.astype(np.float32).reshape(t.shape[0], -1, rot) @ self.S)
                      .reshape(t.shape).astype(np.int16), np.float32(sc / math.sqrt(rot))) for t, sc in parts]
        self.bounds, at = [], 0
        for t, sc in parts:
            self.bounds.append((at, at + t.shape[0], np.float32(sc)))
            at += t.shape[0]
        if kernel is not None:
            self.w = np.ascontiguousarray(np.concatenate([t for t, _ in parts]))
        else:
            self.w = np.concatenate([t.astype(np.float32) * sc for t, sc in parts])

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        """x (T, K) float -> one (T, N_part) float32 output per part, input int8-quantized first."""
        if self.rot:
            x = hadamard_blocks(x, self.S)
        q, s = self.act(x)
        if self.kernel is None:
            y = (q / s) @ self.w.T
            return [y[:, a:b] for a, b, _ in self.bounds]
        acc = np.empty((q.shape[0], self.w.shape[0]), np.int32)
        self.kernel(q, self.w, acc)
        # float32 before scaling: int32 * float32 promotes to float64 in numpy, and a float64 residual
        # stream made the tied head cast its 1.3 GB of weights to float64 on every token (4.3 s/token).
        y = acc.astype(np.float32)
        return [y[:, a:b] * (sc / s) for a, b, sc in self.bounds]


class BitNet(Qwen3):
    """BitNet b1.58 decoder (arch "bitnet-b1.58"): LLaMA-shaped GQA attention with NEOX RoPE and no
    q/k norms, a squared-ReLU gated MLP, and two extra RMSNorms ("sub norms") just before the
    attention output and down projections. Every projection is ternary x one scale per tensor and
    takes int8-quantized input (_act_int8); the tied output head is plain float16, unquantized.

    With numba (_ternary_kernel) the projections stay int8: ~2.1 GB, plus the float32 head/embedding
    (1.3 GB). Without it they are dequantized to float32 for BLAS: ~9.6 GB in all. The attention,
    cache, RoPE and prefix reuse are Qwen3's, unchanged."""

    template = staticmethod(bitnet_chat)

    def __init__(self, path: Path, ctx: int = DEFAULT_CTX, g: gguf.GGUF | None = None):
        t0 = time.monotonic()
        g = g or gguf.load(path)
        arch = g.get("general.architecture")
        if arch != "bitnet-b1.58":
            g.close()
            raise NotImplementedError(f"architecture {arch!r} is not bitnet-b1.58")
        p = lambda k, default=None: g.get(f"{arch}.{k}", default)
        self.name = g.get("general.name") or path.stem
        self.n_layer, self.d = p("block_count"), p("embedding_length")
        self.n_head, self.n_kv = p("attention.head_count"), p("attention.head_count_kv")
        self.hd = p("rope.dimension_count") or self.d // self.n_head
        self.eps = p("attention.layer_norm_rms_epsilon", 1e-5)
        self.ctx = min(ctx, p("context_length", ctx))
        # The 2B4T GGUF has no tokenizer.ggml.pre, so a loader that defaults it splits digits and
        # words differently from training. Its tokenizer.json says LLaMA-3's split.
        self.tok = BPE(g.get("tokenizer.ggml.tokens"), g.get("tokenizer.ggml.merges"),
                       g.get("tokenizer.ggml.token_type"), g.get("tokenizer.ggml.pre") or "llama-bpe")
        self.eos = {g.get("tokenizer.ggml.eos_token_id"), self.tok.special.get("<|eot_id|>"),
                    self.tok.special.get("<|end_of_text|>")} - {None}
        W = lambda n: load_tensor(g, n)
        self.embd = W("token_embd.weight")
        self.out = W("output.weight") if "output.weight" in g.tensors else self.embd
        self.out_norm = W("output_norm.weight")
        import os
        # DROCK_BITNET_ACT=float (research/flow): no int8 activation rounding, float32 BLAS path.
        self.act = os.environ.get("DROCK_BITNET_ACT", "int8").lower()
        self.kernel = None if self.act == "float" else _ternary_kernel()
        act = _act_float if self.act == "float" else _act_int8
        self._head8, self._head_S = None, None
        self._head_k = int(os.environ.get("DROCK_BITNET_HEAD_K", "16") or 0)
        head = os.environ.get("DROCK_BITNET_HEAD", "").lower()
        if self.kernel is not None and head in ("int8", "int8rot"):
            w = self.out
            if head == "int8rot" and self.d % 256 == 0:
                # research/flow Phase 4: fold the block Hadamard into the float head before rounding it,
                # so both roundings (weight rows, final state) see rotated, outlier-free vectors. The
                # head is float, so unlike W_down it stays int8 after the fold: same RAM, same kernel.
                self._head_S = sylvester(256)
                w = hadamard_blocks(w, self._head_S)
            hs = (np.maximum(np.abs(w).max(axis=1), 1e-12) / 127).astype(np.float32)
            self._head8 = (np.clip(np.round(w / hs[:, None]), -127, 127).astype(np.int8), hs)
            del w

        # DROCK_BITNET_ROT=down (research/flow Phase 4.5): Hadamard-rotate the down projection's input
        rot_down = 256 if os.environ.get("DROCK_BITNET_ROT", "").lower() == "down" else 0
        self.rot = "down" if rot_down else None

        def T(*names, rot=0):
            parts = []
            for n in names:
                t = g.tensors[n]
                raw = np.frombuffer(g.mm, dtype=np.uint8, count=t.nbytes, offset=g.data_start + t.offset)
                if t.dtype.name != "I2_S":
                    raise NotImplementedError(f"{n}: expected I2_S ternary weights, found {t.dtype.name}")
                q, sc = ternary_i2_s(raw, math.prod(t.shape))
                parts.append((q.reshape(tuple(reversed(t.shape))), sc))
            return _Ternary(parts, self.kernel, act, rot)

        self.layers = []
        for i in range(self.n_layer):
            b = f"blk.{i}."
            L = {k: W(b + v) for k, v in (("attn_norm", "attn_norm.weight"), ("attn_sub_norm", "attn_sub_norm.weight"),
                                          ("ffn_norm", "ffn_norm.weight"), ("ffn_sub_norm", "ffn_sub_norm.weight"))}
            L["qkv"] = T(b + "attn_q.weight", b + "attn_k.weight", b + "attn_v.weight")
            L["o"] = T(b + "attn_output.weight")
            L["gate_up"] = T(b + "ffn_gate.weight", b + "ffn_up.weight")
            L["down"] = T(b + "ffn_down.weight", rot=rot_down)
            self.layers.append(L)
        g.close()
        inv = p("rope.freq_base", 500000.0) ** (-np.arange(0, self.hd, 2, dtype=np.float64) / self.hd)
        ang = np.outer(np.arange(self.ctx), inv)
        self.cos, self.sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        self._alloc_cache()
        self.load_seconds = round(time.monotonic() - t0, 1)

    def forward(self, ids: list[int]) -> np.ndarray:
        P, T = len(self.cached), len(ids)
        if P + T > self.ctx:
            raise ValueError(f"context full ({P + T} > {self.ctx} tokens)")
        x = self.embd[ids]
        S, g = P + T, self.n_head // self.n_kv
        mask = np.where(np.arange(S)[None, :] <= (P + np.arange(T))[:, None], 0.0, -np.inf).astype(np.float32)
        scale = 1.0 / math.sqrt(self.hd)
        pos = slice(P, P + T)
        trace = self.trace
        if trace is not None:
            trace("embed", -1, x)
        for li, L in enumerate(self.layers):
            q, k, v = L["qkv"](_rms(x, L["attn_norm"], self.eps))
            q = q.reshape(T, self.n_head, self.hd)
            k = k.reshape(T, self.n_kv, self.hd)
            v = v.reshape(T, self.n_kv, self.hd)
            o = self._attend(li, q, k, v, pos, S, g, mask, scale)
            x = x + L["o"](_rms(o, L["attn_sub_norm"], self.eps))[0]
            if trace is not None:
                trace("attn", li, x)
            a, u = L["gate_up"](_rms(x, L["ffn_norm"], self.eps))
            a = np.maximum(a, 0.0)
            x = x + L["down"](_rms(a * a * u, L["ffn_sub_norm"], self.eps))[0]
            if trace is not None:
                trace("layer", li, x)
        self.cached += list(ids)
        xn = _rms(x[-1:], self.out_norm, self.eps)
        if trace is not None:
            trace("final", self.n_layer, xn)
        return self._head(xn)

    def _head(self, xn: np.ndarray) -> np.ndarray:
        """Next-token logits from the final normed hidden state (1, d): the tied float head, exact.

        Once the projections use the kernel, this 1.3 GB float32 product is a third of every token
        (~40 of ~140 ms). DROCK_BITNET_HEAD=int8 quantizes it per row (+0.33 GB) and runs it through
        the kernel in ~13 ms, but it is an approximation the model was not trained with: measured
        over 135 greedy steps it picked a different token 3 times (logit cosine >= 0.9998). So it is
        opt-in, and the exact head is the default.

        Those flips are near-ties (top-1 minus top-2 logit 0.03-0.06), so the int8 head now re-scores
        its own top-k (DROCK_BITNET_HEAD_K, default 16; 0 turns it off) exactly from the float rows,
        k x d multiplies: greedy then matches the exact head unless the winner falls outside the top-k.
        research/flow/headbench.py, 576 teacher-forced steps: int8 2 flips -> 0, KL 3.8e-4 -> 8e-6,
        ~12 ms either way."""
        if self._head8 is None:
            return (xn @ self.out.T)[0]
        h8, hs = self._head8
        xr = hadamard_blocks(xn, self._head_S) if self._head_S is not None else xn
        q, s = _act_int8(xr)
        acc = np.empty((1, h8.shape[0]), np.int32)
        self.kernel(q, h8, acc)
        logits = (acc.astype(np.float32) * hs / s)[0]
        if self._head_k:
            idx = np.argpartition(logits, -self._head_k)[-self._head_k:]
            logits[idx] = (xn @ self.out[idx].T)[0]
        return logits


_ARCHS = {"qwen3": Qwen3, "llama": Llama, "bitnet-b1.58": BitNet}


def load_model(path: Path, ctx: int = DEFAULT_CTX) -> Qwen3:
    """The model class for the file's general.architecture, loaded (the header is parsed once)."""
    g = gguf.load(path)
    arch = g.get("general.architecture")
    if arch not in _ARCHS:
        g.close()
        raise NotImplementedError(f"architecture {arch!r} is not supported yet ({', '.join(_ARCHS)})")
    return _ARCHS[arch](path, ctx, g)


class Superseded(ValueError):
    """A newer call on the same engine took over; this one's answer would be ignored anyway."""


class Engine:
    """One loaded model. Keeps its KV cache between calls, so the shared prompt prefix is free.

    One call at a time. Esc in the REPL (ui.interruptible) abandons a call on its worker thread
    without stopping it, and the next call used to run forward() on the same cache concurrently:
    two threads appending to `cached`/kc/vc corrupt both answers or raise from a shape mismatch.
    So every call takes the lock, and a new call first bumps `_turn`, which the running one checks
    per token and per prefill chunk: an abandoned generation stops at its next token instead of
    making the user wait out its whole budget."""

    def __init__(self, path: Path, ctx: int = DEFAULT_CTX):
        import threading
        self.m = load_model(path, ctx)
        self.tok = self.m.tok
        self._lock = threading.Lock()
        self._turn = 0
        self._turn_lock = threading.Lock()

    def _claim(self):
        """Take the next turn (superseding any running call) and return its check()."""
        with self._turn_lock:  # two claims must never share a turn, or neither is superseded
            self._turn += 1
            mine = self._turn

        def check():
            if self._turn != mine:
                raise Superseded("superseded by a newer call")
        return check

    def _prompt(self, messages: list[dict]) -> list[int]:
        ids = self.tok.encode(self.m.template(messages))
        if len(ids) >= self.m.ctx - 16:
            raise ValueError(f"prompt is {len(ids)} tokens; the context holds {self.m.ctx}")
        return ids

    def chat(self, messages: list[dict], schema: dict | None = None, max_tokens: int = 400,
             temperature: float = 0.1) -> tuple[str, dict]:
        from .scaffold import Decoder
        t0 = time.monotonic()
        ids = self._prompt(messages)
        check = self._claim()
        with self._lock:
            check()
            reused = min(sum(1 for _ in _common(self.m.cached, ids)), len(ids) - 1)
            lm = _LM(self.m, self.m.prime(ids, check=check), check)
            t1 = time.monotonic()
            if schema is not None:
                dec = Decoder(lm, self.tok, temperature, budget=max_tokens)
                text = dec.value_text(schema)
                gen = dec.tokens  # the last few may never be pushed (see Decoder), so not len(cached)
            else:
                text = "".join(self._free(lm, max_tokens, temperature))
                gen = max(len(self.m.cached) - len(ids), 0)
        dt = time.monotonic() - t0
        return text, {"seconds": round(dt, 2), "tok_s": round(gen / max(time.monotonic() - t1, 1e-6), 1),
                      "prompt_tokens": len(ids), "completion_tokens": gen, "cached": reused}

    def stream(self, messages: list[dict], max_tokens: int = 600, temperature: float = 0.7):
        ids = self._prompt(messages)
        check = self._claim()
        with self._lock:
            check()
            lm = _LM(self.m, self.m.prime(ids, check=check), check)
        # The lock is held per token, never across a yield: a consumer that stops iterating without
        # closing the generator must not leave the engine locked for every later call.
        yield from self._free(lm, max_tokens, temperature, self._lock)

    def _free(self, lm: "_LM", max_tokens: int, temperature: float, lock=None):
        """Unconstrained generation until end-of-turn; yields text as whole characters complete.
        With `lock`, each token is computed under it (stream); chat() already holds it throughout."""
        buf, sent = b"", 0
        for _ in range(max_tokens):
            if lock is not None:
                with lock:
                    tid = lm.sample(temperature)
                    if tid not in self.m.eos:
                        lm.push([tid])
            else:
                tid = lm.sample(temperature)
                if tid not in self.m.eos:
                    lm.push([tid])
            if tid in self.m.eos:
                break
            buf += self.tok.token_bytes(tid)
            text = buf.decode("utf-8", "ignore")
            if len(text) > sent:
                yield text[sent:]
                sent = len(text)


def _common(a, b):
    for x, y in zip(a, b):
        if x != y:
            return
        yield x


class _LM:
    """The decoder's view of the model: current logits, push tokens, pick a token."""

    def __init__(self, model: Qwen3, logits: np.ndarray, check=None):
        self.m, self.logits = model, logits
        self.check = check
        self.rng = np.random.default_rng()

    def push(self, ids: list[int]) -> None:
        if ids:
            if self.check is not None:
                self.check()
            self.logits = self.m.forward(list(ids))

    def top(self, k: int = 256) -> np.ndarray:
        idx = np.argpartition(-self.logits, k)[:k]
        return idx[np.argsort(-self.logits[idx])]

    def sample(self, temperature: float, allowed: np.ndarray | None = None) -> int:
        cand = self.top(64) if allowed is None else allowed
        if temperature <= 0.15 or len(cand) == 1:
            return int(cand[np.argmax(self.logits[cand])])
        z = self.logits[cand] / temperature
        p = np.exp(z - z.max())
        return int(self.rng.choice(cand, p=p / p.sum()))


_ENGINES: dict[tuple[str, int], Engine] = {}


def engine(model: str, ctx: int | None = None) -> Engine:
    """The loaded engine for a model file name in drock/models (or a path), loaded once per process.

    `ctx` is the tier's context size from models.json (DEFAULT_CTX when not given). It used to be
    ignored, so a native tier set to 8192 got a 4096-token engine while the agent budgeted prompts
    for 8192, and long prompts were refused by _prompt()."""
    path = Path(model) if Path(model).is_absolute() else MODELS_DIR / model
    key = (str(path), int(ctx or DEFAULT_CTX))
    if key not in _ENGINES:
        if not path.exists():
            raise FileNotFoundError(f"model file missing: {path}")
        _ENGINES[key] = Engine(path, key[1])
    return _ENGINES[key]
