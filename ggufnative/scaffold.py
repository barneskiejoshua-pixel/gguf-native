"""Schema-constrained decoding by scaffolding: drock writes the JSON, the model fills the blanks.

A grammar engine would mask the whole vocabulary at every step. drock's schemas are simple (one
object per action: fixed property order, strings, integers, booleans, consts, enums, and a oneOf
keyed by the "action" const), so the decoder instead emits every structural character itself and
consults the model only where the schema leaves a choice:
  * which action name (and which optional parameter, or none)   -> choose() among candidates
  * free text in a string                                       -> string()
  * digits of an integer                                        -> integer()
The model's tokens are checked against the candidates in logit order, so each choice is the most
likely token the schema allows. The output is valid JSON for the schema by construction.
"""
from __future__ import annotations

import json

import numpy as np


class Decoder:
    """`batch`: hold the tokens of forced text and of taken tokens, and push them in one forward pass
    just before the model is next consulted. Every piece is encoded exactly as before, so the token
    sequence the model sees is identical; only the number of passes changes. One action forces
    `{`, `"reason": `, `"`, the closing `"`, `, "action": ` and each key separately, and each of those
    was a full pass over every layer — 8-12 extra passes per action, when a pass costs about as much
    for 5 tokens as for 1 (decode is bound by reading the weights, not by arithmetic). Tokens still
    pending at the end are never pushed: nothing reads logits after the closing brace."""

    def __init__(self, lm, tok, temperature: float = 0.1, budget: int = 400, batch: bool = True):
        self.lm, self.tok, self.t, self.budget = lm, tok, temperature, budget
        self.out: list[str] = []
        self.batch = batch
        self._pending: list[int] = []
        self.tokens = 0  # tokens emitted (pushed or pending): the completion length

    # ---------- primitives ----------
    def _feed(self, ids: list[int]) -> None:
        self.tokens += len(ids)
        if self.batch:
            self._pending += ids
        else:
            self.lm.push(ids)

    def _ready(self):
        """The model, with every emitted token pushed: call before reading its logits."""
        if self._pending:
            ids, self._pending = self._pending, []
            self.lm.push(ids)
        return self.lm

    def force(self, text: str) -> None:
        """Structural text the schema dictates: fed through the model so its cache stays in step."""
        if text:
            self._feed(self.tok.encode(text, special=False))
            self.out.append(text)

    def _take(self, tid: int, piece: str) -> None:
        self._feed([tid])
        self.out.append(piece)
        self.budget -= 1

    def _pick(self, valid: list[int]) -> int:
        return self._ready().sample(self.t, np.array(valid)) if len(valid) > 1 else valid[0]

    def choose(self, options: list[str]) -> str:
        """Emit exactly one of `options` (none a prefix of another), token by token, following the
        model's preference among tokens that keep at least one option possible."""
        done, live = "", list(options)
        while True:
            if len(live) == 1:
                self.force(live[0][len(done):])
                return live[0]
            rest = [o[len(done):] for o in live]
            valid = []
            for tid in self._ready().top(256):
                p = self.tok.piece(int(tid))
                if p and "�" not in p and any(r.startswith(p) for r in rest):
                    valid.append(int(tid))
                    if len(valid) >= 8:
                        break
            if not valid:  # the model wants none of them: take the option whose first token it likes best
                best = max(live, key=lambda o: self._ready().logits[self.tok.encode(o[len(done):], special=False)[0]])
                self.force(best[len(done):])
                return best
            tid = self._pick(valid)
            p = self.tok.piece(tid)
            self._take(tid, p)
            done += p
            live = [o for o in live if o.startswith(done)]
            if done in live:
                return done

    def string(self, max_len: int | None = None, max_tokens: int | None = None) -> str:
        """Free text up to the closing quote (the opening one is already out). Quotes, backslashes
        and control characters are never emitted inside, so the result is always a valid JSON string.

        Bounded by the schema's maxLength when it has one, otherwise only by the call's token budget.
        A fixed 200-char / 80-token cap used to apply to every string, so CREATE_FILE/WRITE_FILE
        `content` on a native tier was silently cut at 200 characters and the gate failed the step."""
        text = ""
        max_len = float("inf") if max_len is None else max_len
        n = 0
        while max_tokens is None or n < max_tokens:
            n += 1
            if self.budget <= 0 or len(text) >= max_len:
                break
            valid, end = [], False
            for tid in self._ready().top(64):
                tid = int(tid)
                p = self.tok.piece(tid)
                if p.startswith('"'):          # '"', '",' ...: the model wants to close the string
                    if not valid:
                        end = True
                    break
                if p and "�" not in p and '"' not in p and "\\" not in p and not any(ord(c) < 32 for c in p) \
                        and tid not in self.tok.special_ids:
                    valid.append(tid)
                    if len(valid) >= 8:
                        break
            if end or not valid:
                break
            tid = self._pick(valid)
            p = self.tok.piece(tid)
            if len(text) + len(p) > max_len:
                p = p[:int(max_len - len(text))]
            self._take(tid, p)
            text += p
        self.force('"')
        return text

    def integer(self, max_digits: int = 6) -> str:
        """Digits only, and never a leading zero: JSON rejects 007, so the model can't write it."""
        s = ""
        while len(s) < max_digits and self.budget > 0:
            nxt = None
            for tid in self._ready().top(64):
                p = self.tok.piece(int(tid))
                if s and p.strip()[:1] in (",", "}"):
                    break                                        # the model wants to end the number
                if not (p.isascii() and p.isdigit()) or len(s) + len(p) > max_digits:
                    continue
                if not s and p.startswith("0") and p != "0":
                    continue
                nxt = (int(tid), p)
                break
            if nxt is None:
                break
            self._take(*nxt)
            s += nxt[1]
            if s == "0":
                break                                            # a bare 0 is the only legal zero
        if not s:
            self.force("1")
            s = "1"
        return s

    # ---------- schema walk ----------
    def value(self, sch: dict) -> None:
        if "oneOf" in sch:
            return self._one_of(sch["oneOf"])
        if "const" in sch:
            return self.force(json.dumps(sch["const"]))
        if "enum" in sch:
            self.choose([json.dumps(e) for e in sch["enum"]])
            return
        t = sch.get("type")
        if t == "object":
            self.force("{")
            self._tail(sch, [], True)
        elif t == "string":
            self.force('"')
            self.string(sch.get("maxLength"))
        elif t == "integer":
            self.integer()
        elif t == "boolean":
            self.choose(["true", "false"])
        else:
            raise NotImplementedError(f"schema type {t!r}")

    def value_text(self, sch: dict) -> str:
        self.value(sch)
        return "".join(self.out)

    def _tail(self, sch: dict, done: list[str], first: bool) -> None:
        """The rest of an object after the keys in `done`: required keys in order, then any of the
        optional keys (in order) the model chooses to include, then '}'."""
        props, req = sch["properties"], set(sch.get("required", []))
        rest = [k for k in props if k not in done]
        for k in [k for k in rest if k in req]:
            self.force(("" if first else ", ") + json.dumps(k) + ": ")
            self.value(props[k])
            first = False
        opt = [k for k in rest if k not in req]
        while opt:
            sep = "" if first else ", "
            pick = self.choose(["}"] + [sep + json.dumps(k) for k in opt])
            if pick == "}":
                return
            k = json.loads(pick[len(sep):])
            self.force(": ")
            self.value(props[k])
            first = False
            opt = opt[opt.index(k) + 1:]
        self.force("}")

    def _one_of(self, branches: list[dict]) -> None:
        """oneOf over objects that share leading properties and are told apart by a const property
        (drock: "reason" first, then "action": <NAME>). Shared part, then the discriminator, then the branch."""
        objs = [b for b in branches if b.get("type") == "object"]
        keys = list(objs[0]["properties"])
        di = next(i for i, k in enumerate(keys) if all("const" in b["properties"].get(k, {}) for b in objs))
        self.force("{")
        for i, k in enumerate(keys[:di]):
            self.force(("" if i == 0 else ", ") + json.dumps(k) + ": ")
            self.value(objs[0]["properties"][k])
        dk = keys[di]
        self.force((", " if di else "") + json.dumps(dk) + ": ")
        by_name = {json.dumps(b["properties"][dk]["const"]): b for b in objs}
        branch = by_name[self.choose(list(by_name))]
        self._tail(branch, keys[:di + 1], False)
