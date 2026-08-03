"""Serialization contract for Gemma's corrected Mistral-family tokenizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def mark_regex_fix_consumed(model_dir: Path) -> None:
    """Prevent Transformers from applying the regex fix to tokenizer.json twice."""
    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["fix_mistral_regex"] = False
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def converter_vocab(vocab: Mapping[str, int], model_vocab_size: int) -> dict[str, int]:
    """Return Gemma's embeddable text vocabulary, rejecting unknown overflow."""
    overflow = {token: index for token, index in vocab.items() if index >= model_vocab_size}
    expected = {"<image_soft_token>": model_vocab_size}
    if overflow and overflow != expected:
        raise RuntimeError(
            f"unexpected tokenizer IDs outside model vocabulary: {overflow}; "
            f"expected only {expected}"
        )
    return {token: index for token, index in vocab.items() if index < model_vocab_size}
