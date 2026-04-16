from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from training.tokenizer import load_tokenizer


def _resolve_text_field(example: dict[str, Any], text_field: str) -> str:
    value: Any = example
    for part in text_field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Text field {text_field!r} not found in dataset example.")
        value = value[part]

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return " ".join(parts)
    raise TypeError(f"Unsupported text field type for {text_field!r}: {type(value)!r}")


def iter_dataset_texts(dataset_cfg: dict[str, Any]) -> Iterable[str]:
    dataset = load_dataset(
        path=str(dataset_cfg["name"]),
        name=dataset_cfg.get("config_name"),
        split=str(dataset_cfg.get("split", "train")),
        streaming=bool(dataset_cfg.get("streaming", True)),
        token=dataset_cfg.get("token"),
    )

    text_field = str(dataset_cfg.get("text_field", "text"))
    max_examples = dataset_cfg.get("max_examples")
    min_text_length = int(dataset_cfg.get("min_text_length", 1))

    count = 0
    for example in dataset:
        text = _resolve_text_field(example, text_field).strip()
        if len(text) < min_text_length:
            continue
        yield text
        count += 1
        if max_examples not in (None, "", "null", "None") and count >= int(max_examples):
            break


def train_bpe_tokenizer(tokenizer_cfg: dict[str, Any], dataset_cfg: dict[str, Any]) -> PreTrainedTokenizerFast:
    unk_token = str(tokenizer_cfg.get("unk_token", "[UNK]"))
    bos_token = str(tokenizer_cfg.get("bos_token", "[BOS]"))
    eos_token = str(tokenizer_cfg.get("eos_token", "[EOS]"))
    pad_wait_token = str(tokenizer_cfg.get("pad_wait_token", "[P]"))
    word_start_token = str(tokenizer_cfg.get("word_start_token", "[W]"))
    extra_specials = list(tokenizer_cfg.get("additional_special_tokens", []))

    # Keep order stable so token ids for the training control symbols are deterministic.
    special_tokens: list[str] = []
    for token in [unk_token, bos_token, eos_token, pad_wait_token, word_start_token, *extra_specials]:
        if token not in special_tokens:
            special_tokens.append(token)

    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    normalizer_steps = [normalizers.NFKC()]
    if bool(tokenizer_cfg.get("lowercase", False)):
        normalizer_steps.append(normalizers.Lowercase())
    tokenizer.normalizer = normalizers.Sequence(normalizer_steps)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=bool(tokenizer_cfg.get("add_prefix_space", False)))
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=int(tokenizer_cfg.get("vocab_size", 32000)),
        min_frequency=int(tokenizer_cfg.get("min_frequency", 2)),
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_dataset_texts(dataset_cfg), trainer=trainer)

    additional_special_tokens = [token for token in special_tokens if token not in {unk_token, bos_token, eos_token}]
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        model_max_length=int(tokenizer_cfg.get("model_max_length", 16384)),
        clean_up_tokenization_spaces=False,
    )
    hf_tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    explicit_pad_token = tokenizer_cfg.get("pad_token")
    if explicit_pad_token not in (None, "", "null", "None"):
        hf_tokenizer.pad_token = str(explicit_pad_token)

    return hf_tokenizer


def build_training_tokenizer_cfg(cfg: dict[str, Any], tokenizer_path: str) -> dict[str, Any]:
    tokenizer_cfg = cfg["tokenizer"]
    additional_special_tokens = list(tokenizer_cfg.get("additional_special_tokens", []))
    for token in (tokenizer_cfg.get("pad_wait_token", "[P]"), tokenizer_cfg.get("word_start_token", "[W]")):
        if token not in additional_special_tokens:
            additional_special_tokens.append(token)

    return {
        "name": tokenizer_path,
        "revision": None,
        "use_fast": True,
        "bos_token": str(tokenizer_cfg.get("bos_token", "[BOS]")),
        "eos_token": str(tokenizer_cfg.get("eos_token", "[EOS]")),
        "pad_wait_token": str(tokenizer_cfg.get("pad_wait_token", "[P]")),
        "word_start_token": str(tokenizer_cfg.get("word_start_token", "[W]")),
        "additional_special_tokens": additional_special_tokens,
    }


def validate_training_compatibility(cfg: dict[str, Any], tokenizer_path: str) -> dict[str, int]:
    training_tokenizer_cfg = build_training_tokenizer_cfg(cfg, tokenizer_path)
    vocab_before = len(PreTrainedTokenizerFast.from_pretrained(tokenizer_path))
    resolved = load_tokenizer(training_tokenizer_cfg)
    vocab_after = len(resolved.tokenizer)
    if vocab_after != vocab_before:
        raise RuntimeError(
            "Saved tokenizer is not fully compatible with the training stack: loading it through "
            "`training.tokenizer.load_tokenizer` changed the vocabulary size."
        )
    return {
        "vocab_size": vocab_after,
        "bos_token_id": resolved.bos_token_id,
        "eos_token_id": resolved.eos_token_id,
        "pad_wait_token_id": resolved.pad_wait_token_id,
        "word_start_token_id": resolved.word_start_token_id,
    }


def save_tokenizer_artifacts(tokenizer: PreTrainedTokenizerFast, output_dir: str | Path) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    return output_path


def maybe_push_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    *,
    output_dir: str | Path,
    push_cfg: dict[str, Any],
) -> str | None:
    if not bool(push_cfg.get("enabled", False)):
        return None

    repo_id = push_cfg.get("repo_id")
    if repo_id in (None, "", "null", "None"):
        raise ValueError("push_to_hub is enabled but no output.repo_id was provided.")

    return tokenizer.push_to_hub(
        repo_id=str(repo_id),
        private=bool(push_cfg.get("private", True)),
        token=push_cfg.get("token"),
        commit_message=str(push_cfg.get("commit_message", "Upload tokenizer")),
    )
