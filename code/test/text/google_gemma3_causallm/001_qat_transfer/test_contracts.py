from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[5]
DATA_PATH = ROOT / "code/experiments/text/google_gemma3_causallm/001_qat_transfer/data.py"
SPEC = importlib.util.spec_from_file_location("gemma_qv_data", DATA_PATH)
data = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(data)
TOKENIZER_CONTRACT_PATH = ROOT / "code/experiments/text/google_gemma3_causallm/001_qat_transfer/tokenizer_contract.py"
TOKENIZER_SPEC = importlib.util.spec_from_file_location("gemma_tokenizer_contract", TOKENIZER_CONTRACT_PATH)
tokenizer_contract = importlib.util.module_from_spec(TOKENIZER_SPEC)
assert TOKENIZER_SPEC.loader is not None
TOKENIZER_SPEC.loader.exec_module(tokenizer_contract)
TRAINING_CONTRACT_PATH = ROOT / "code/experiments/text/google_gemma3_causallm/001_qat_transfer/training_contract.py"
TRAINING_SPEC = importlib.util.spec_from_file_location("gemma_training_contract", TRAINING_CONTRACT_PATH)
training_contract = importlib.util.module_from_spec(TRAINING_SPEC)
assert TRAINING_SPEC.loader is not None
TRAINING_SPEC.loader.exec_module(training_contract)
sys.path.insert(0, str(ROOT / "code"))
from common.run_id import guard_run_config, run_id_path

RUN_ID_PARAMS = ("model", "task", "mode", "seed", "data_spec", "train_spec", "qv_source", "alpha", "quantizer", "eval_spec")


def test_prompts_are_frozen() -> None:
    assert data.SYSTEMS["gsm8k"] == "You are a careful mathematical reasoner."
    assert data.user_prompt("gsm8k", "2+2?") == (
        "Solve the problem. Show your reasoning, then end with #### <answer>.\n\nProblem:\n2+2?"
    )
    assert data.user_prompt("samsum", "A: hi") == (
        "Summarize the following dialogue in one concise paragraph.\n\nDialogue:\nA: hi"
    )
    assert data.user_prompt("e2e_nlg", "name[X]") == (
        "Express the following meaning representation as one natural sentence.\n\n"
        "Meaning representation:\nname[X]"
    )


def test_gsm8k_normalization_and_missing_marker() -> None:
    rows = [{"target": "work\n#### 1,024"}, {"target": "#### -3.0"}]
    predictions = ["reasoning\n#### 1024.", "the answer is -3.0"]
    assert data.score_predictions("gsm8k", rows, predictions)["gsm8k_em"] == 0.5


def test_e2e_uses_multiple_references() -> None:
    rows = [{"references": ["A small cafe.", "There is a small cafe."], "target": "A small cafe."}]
    metrics = data.score_predictions("e2e_nlg", rows, ["There is a small cafe."])
    assert metrics["rougeL"] == 1.0
    assert metrics["bleu"] > 99.0
    assert metrics["cider"] is None
    assert metrics["nist"] is None


def test_config_identity_and_collision_guard(tmp_path: Path) -> None:
    config = OmegaConf.to_container(OmegaConf.load(
        ROOT / "config/experiments/text/google_gemma3_causallm/001_qat_transfer/run_task.yaml"
    ), resolve=True)
    assert list(run_id_path(config, RUN_ID_PARAMS).parts) == [
        "model=gemma-3-1b-it", "task=gsm8k", "mode=full", "seed=2038",
        "data_spec=equal6449_v1", "train_spec=emnlp2025_fullft_v1",
        "qv_source=gemma-3-1b-it-qat-q4_0", "alpha=1.0",
        "quantizer=llamacpp-b9637-q4_0", "eval_spec=gemma_gen_v1",
    ]
    guard_run_config(config, RUN_ID_PARAMS, tmp_path)
    changed = dict(config)
    changed["model_revision"] = "different"
    try:
        guard_run_config(changed, RUN_ID_PARAMS, tmp_path)
    except RuntimeError as error:
        assert "model_revision" in str(error)
    else:
        raise AssertionError("full-config collision was not rejected")


def test_bf16_qv_exactly_reconstructs_donor() -> None:
    fp = torch.tensor([1.0001234, -0.1250678], dtype=torch.float32)
    qat = torch.tensor([1.0004321, -0.1249321], dtype=torch.float32)
    delta = qat.to(torch.bfloat16).float() - fp.to(torch.bfloat16).float()
    reconstructed = (fp.to(torch.bfloat16).float() + delta).to(torch.bfloat16)
    assert torch.equal(reconstructed, qat.to(torch.bfloat16))

    # Catastrophic cancellation can make an FP32 delta insufficient. The
    # materializer detects this and stores only the affected tensor in FP64.
    fp_tiny = torch.tensor([0.0002574920654296875], dtype=torch.bfloat16)
    qat_tiny = torch.tensor([-1.8553691916167736e-09], dtype=torch.bfloat16)
    delta32 = qat_tiny.float() - fp_tiny.float()
    assert not torch.equal((fp_tiny.float() + delta32).to(torch.bfloat16), qat_tiny)
    delta64 = qat_tiny.double() - fp_tiny.double()
    assert torch.equal((fp_tiny.double() + delta64).to(torch.bfloat16), qat_tiny)


def test_serialized_tokenizer_does_not_reapply_regex_fix(tmp_path: Path) -> None:
    config_path = tmp_path / "tokenizer_config.json"
    config_path.write_text(json.dumps({"fix_mistral_regex": True, "model_max_length": 32768}))
    tokenizer_contract.mark_regex_fix_consumed(tmp_path)
    config = json.loads(config_path.read_text())
    assert config == {"fix_mistral_regex": False, "model_max_length": 32768}


def test_converter_vocab_excludes_only_unembeddable_image_token() -> None:
    vocab = {"hello": 0, "world": 1, "<image_soft_token>": 2}
    assert tokenizer_contract.converter_vocab(vocab, 2) == {"hello": 0, "world": 1}
    with pytest.raises(RuntimeError, match="unexpected tokenizer IDs"):
        tokenizer_contract.converter_vocab({"hello": 0, "surprise": 2}, 2)


def test_converter_wrapper_imports_its_sibling_contract(tmp_path: Path) -> None:
    dummy_converter = tmp_path / "dummy_converter.py"
    dummy_converter.write_text("print('dummy converter reached')\n")
    wrapper = ROOT / "code/experiments/text/google_gemma3_causallm/001_qat_transfer/convert_hf_to_gguf_compat.py"
    completed = subprocess.run(
        [sys.executable, "-P", str(wrapper), str(dummy_converter), str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.stdout.strip() == "dummy converter reached"


def test_accumulation_schedule_preserves_final_partial_window() -> None:
    assert [training_contract.optimizer_step_due(i, 807, 4) for i in range(803, 807)] == [True, False, False, True]
    assert [training_contract.accumulation_divisor(i, 807, 4) for i in range(803, 807)] == [4, 3, 3, 3]
