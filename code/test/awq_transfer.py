import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/vision/ilharco_timm_supervised/010_awq_transfer/qv_transfer_awqv.py"
)
SPEC = importlib.util.spec_from_file_location("qv_transfer_awqv", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_awq_vector_excludes_head_and_integer_buffers():
    fp = {
        "model.block.weight": torch.tensor([1.0, 2.0]),
        "model.head.weight": torch.tensor([3.0]),
        "model.counter": torch.tensor([1], dtype=torch.int64),
    }
    awq = {
        "model.block.weight": torch.tensor([0.5, 2.5]),
        "model.head.weight": torch.tensor([9.0]),
        "model.counter": torch.tensor([2], dtype=torch.int64),
    }
    vector, heads, integers = MODULE.build_awq_vector(fp, awq)
    assert set(vector) == {"model.block.weight"}
    assert torch.equal(vector["model.block.weight"], torch.tensor([-0.5, 0.5]))
    assert heads == 1
    assert integers == 1


def test_apply_awq_vector_preserves_receiver_head():
    target = {
        "model.block.weight": torch.tensor([10.0, 20.0]),
        "model.head.weight": torch.tensor([30.0]),
    }
    vector = {"model.block.weight": torch.tensor([-2.0, 4.0])}
    patched = MODULE.apply_vector(target, vector, alpha=0.5)
    assert torch.equal(patched["model.block.weight"], torch.tensor([9.0, 22.0]))
    assert torch.equal(patched["model.head.weight"], target["model.head.weight"])
