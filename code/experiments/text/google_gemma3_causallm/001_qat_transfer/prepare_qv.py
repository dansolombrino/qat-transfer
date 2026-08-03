"""Materialize and verify the FP32 Gemma 3 QAT quantization vector once."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import hydra
import torch
from dotenv import load_dotenv
from filelock import FileLock
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_map(snapshot: Path) -> dict[str, Path]:
    index_files = sorted(snapshot.glob("*.safetensors.index.json"))
    if index_files:
        index = json.loads(index_files[0].read_text())
        return {key: snapshot / filename for key, filename in index["weight_map"].items()}
    files = sorted(snapshot.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights in {snapshot}")
    result: dict[str, Path] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in result:
                    raise RuntimeError(f"duplicate tensor key {key}")
                result[key] = path
    return result


def _read(mapping: dict[str, Path], key: str) -> torch.Tensor:
    with safe_open(mapping[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def prepare(config: dict[str, Any]) -> Path:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    qv_path = output_dir / "qv.safetensors"
    manifest_path = output_dir / "manifest.json"
    with FileLock(str(output_dir / ".prepare.lock")):
        if qv_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["qv_sha256"] != _sha256(qv_path):
                raise RuntimeError("existing QV checksum does not match its manifest")
            expected = {"fp_model_id": config["fp_model_id"], "fp_revision": config["fp_revision"], "qat_model_id": config["qat_model_id"], "qat_revision": config["qat_revision"]}
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise RuntimeError("existing QV was built from different source revisions")
            print(f"verified existing QV: {qv_path}", flush=True)
            return qv_path

        patterns = ["*.json", "*.model", "*.safetensors", "*.safetensors.index.json", "tokenizer*"]
        fp_snapshot = Path(snapshot_download(repo_id=config["fp_model_id"], revision=config["fp_revision"], allow_patterns=patterns))
        qat_snapshot = Path(snapshot_download(repo_id=config["qat_model_id"], revision=config["qat_revision"], allow_patterns=patterns))
        fp_map, qat_map = _tensor_map(fp_snapshot), _tensor_map(qat_snapshot)
        if set(fp_map) != set(qat_map):
            raise RuntimeError(f"checkpoint key mismatch: missing_fp={sorted(set(qat_map)-set(fp_map))[:5]} missing_qat={sorted(set(fp_map)-set(qat_map))[:5]}")
        qv: dict[str, torch.Tensor] = {}
        floating = nonfloating = numel = 0
        float64_fallbacks: list[str] = []
        for index, key in enumerate(sorted(fp_map)):
            fp, qat = _read(fp_map, key), _read(qat_map, key)
            if fp.shape != qat.shape or fp.is_floating_point() != qat.is_floating_point():
                raise RuntimeError(f"source tensor incompatibility for {key}")
            if fp.is_floating_point():
                fp_bf16 = fp.to(torch.bfloat16)
                qat_bf16 = qat.to(torch.bfloat16)
                delta = qat_bf16.float() - fp_bf16.float()
                if not torch.equal((fp_bf16.float() + delta).to(torch.bfloat16), qat_bf16):
                    delta = qat_bf16.double() - fp_bf16.double()
                    if not torch.equal((fp_bf16.double() + delta).to(torch.bfloat16), qat_bf16):
                        raise RuntimeError(f"BF16 reconstruction failed for {key}")
                    float64_fallbacks.append(key)
                qv[key] = delta.contiguous()
                floating += 1
                numel += delta.numel()
            else:
                if not torch.equal(fp, qat):
                    raise RuntimeError(f"non-floating source state differs for {key}")
                nonfloating += 1
            if index % 50 == 0:
                print(f"QV tensors {index + 1}/{len(fp_map)}", flush=True)
        temporary = qv_path.with_suffix(".safetensors.tmp")
        save_file(qv, temporary, metadata={"format": "pt", "alpha_reference": "1.0"})
        temporary.replace(qv_path)
        manifest = {"format_version": 2, "definition": "QAT_it[bfloat16] - FP_it[bfloat16]; float32 unless exact reconstruction requires float64", "fp_model_id": config["fp_model_id"], "fp_revision": config["fp_revision"], "fp_snapshot": str(fp_snapshot), "qat_model_id": config["qat_model_id"], "qat_revision": config["qat_revision"], "qat_snapshot": str(qat_snapshot), "floating_tensors": floating, "float32_tensors": floating - len(float64_fallbacks), "float64_fallback_tensors": float64_fallbacks, "nonfloating_tensors": nonfloating, "floating_numel": numel, "bf16_reconstruction_exact": True, "qv_sha256": _sha256(qv_path)}
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary_manifest.replace(manifest_path)
        print(f"wrote verified QV: {qv_path}", flush=True)
        return qv_path


@hydra.main(version_base=None, config_path="../../../../../config/experiments/text/google_gemma3_causallm/001_qat_transfer", config_name="prepare_qv")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    prepare(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
