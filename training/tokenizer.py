from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer


@dataclass(frozen=True)
class ResolvedTokenizer:
    tokenizer: Any
    bos_token_id: int
    eos_token_id: int
    pad_wait_token_id: int
    word_start_token_id: int


def load_tokenizer(tokenizer_cfg: dict[str, Any]) -> ResolvedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_cfg["name"],
        revision=tokenizer_cfg.get("revision"),
        use_fast=bool(tokenizer_cfg.get("use_fast", True)),
    )

    bos_token = tokenizer_cfg.get("bos_token", tokenizer.bos_token or "[BOS]")
    eos_token = tokenizer_cfg.get("eos_token", tokenizer.eos_token or "[EOS]")
    if tokenizer.bos_token is None or tokenizer.bos_token != bos_token:
        tokenizer.add_special_tokens({"bos_token": bos_token})
    if tokenizer.eos_token is None or tokenizer.eos_token != eos_token:
        tokenizer.add_special_tokens({"eos_token": eos_token})

    pad_wait = tokenizer_cfg.get("pad_wait_token", "[P]")
    word_start = tokenizer_cfg.get("word_start_token", "[W]")
    extra_specials = list(tokenizer_cfg.get("additional_special_tokens", ["[P]", "[W]"]))
    for token in (pad_wait, word_start):
        if token not in extra_specials:
            extra_specials.append(token)
    tokenizer.add_special_tokens({"additional_special_tokens": extra_specials})

    resolved = ResolvedTokenizer(
        tokenizer=tokenizer,
        bos_token_id=int(tokenizer.convert_tokens_to_ids(bos_token)),
        eos_token_id=int(tokenizer.convert_tokens_to_ids(eos_token)),
        pad_wait_token_id=int(tokenizer.convert_tokens_to_ids(pad_wait)),
        word_start_token_id=int(tokenizer.convert_tokens_to_ids(word_start)),
    )
    print(
        "[TOKENIZER] "
        f"vocab_size={len(tokenizer)} "
        f"bos={bos_token!r}:{resolved.bos_token_id} "
        f"eos={eos_token!r}:{resolved.eos_token_id} "
        f"pad_wait={pad_wait!r}:{resolved.pad_wait_token_id} "
        f"word_start={word_start!r}:{resolved.word_start_token_id}"
    )
    return resolved
