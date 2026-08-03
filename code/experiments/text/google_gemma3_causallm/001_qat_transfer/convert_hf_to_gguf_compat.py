"""Run pinned llama.cpp conversion with Gemma 3's text-vocabulary contract."""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoTokenizer

from tokenizer_contract import converter_vocab


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: convert_hf_to_gguf_compat.py CONVERTER MODEL_DIR [ARGS...]")
    converter, converter_args = Path(sys.argv[1]).resolve(), sys.argv[2:]
    original_from_pretrained = AutoTokenizer.from_pretrained

    def compatible_from_pretrained(model_dir: Any, *args: Any, **kwargs: Any) -> Any:
        tokenizer = original_from_pretrained(model_dir, *args, **kwargs)
        model_vocab_size = int(AutoConfig.from_pretrained(model_dir).vocab_size)
        bounded = converter_vocab(tokenizer.get_vocab(), model_vocab_size)
        if len(bounded) != len(tokenizer.get_vocab()):
            # Gemma 3 1B's text checkpoint has no embedding row for the
            # multimodal-only <image_soft_token>. b9637 already serializes only
            # range(vocab_size); its pre-loop assertion simply predates this
            # tokenizer layout, so expose that same bounded view to it.
            tokenizer.get_vocab = types.MethodType(lambda self: dict(bounded), tokenizer)
            print(
                "[compat] excluded multimodal-only <image_soft_token> from "
                f"the {model_vocab_size}-row text vocabulary",
                file=sys.stderr,
            )
        return tokenizer

    AutoTokenizer.from_pretrained = staticmethod(compatible_from_pretrained)
    sys.argv = [str(converter), *converter_args]
    runpy.run_path(str(converter), run_name="__main__")


if __name__ == "__main__":
    main()
