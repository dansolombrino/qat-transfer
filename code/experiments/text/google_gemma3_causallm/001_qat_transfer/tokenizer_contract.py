"""Serialization contract for Gemma's corrected Mistral-family tokenizer."""

from __future__ import annotations

import json
from pathlib import Path


def mark_regex_fix_consumed(model_dir: Path) -> None:
    """Prevent Transformers from applying the regex fix to tokenizer.json twice."""
    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["fix_mistral_regex"] = False
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
