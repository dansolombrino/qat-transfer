"""Full-parameter generative fine-tuning and QV/PTQ evaluation for Gemma 3."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import requests
import torch
import torch.distributed as dist
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(HERE))
from common.run_id import guard_run_config, run_id_path  # noqa: E402
from common.status import StatusWriter  # noqa: E402
from data import SYSTEMS, load_task_data, messages, score_predictions, user_prompt  # noqa: E402
from tokenizer_contract import mark_regex_fix_consumed  # noqa: E402
from training_contract import accumulation_divisor, epoch_requires_training, optimizer_step_due  # noqa: E402

RUN_ID_PARAMS = ("model", "task", "mode", "seed", "data_spec", "train_spec", "qv_source", "alpha", "quantizer", "eval_spec")
_CPU_GROUP: Any | None = None


class TokenDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _distributed() -> bool:
    return _world() > 1


def _barrier() -> None:
    if _distributed():
        dist.barrier()


def _cpu_group() -> Any:
    if _CPU_GROUP is None:
        raise RuntimeError("CPU process group is not initialized")
    return _CPU_GROUP


def _init_distributed(timeout_s: int) -> None:
    global _CPU_GROUP
    timeout = timedelta(seconds=timeout_s)
    dist.init_process_group("nccl", timeout=timeout)
    _CPU_GROUP = dist.new_group(backend="gloo", timeout=timeout)


def _destroy_distributed() -> None:
    global _CPU_GROUP
    if _CPU_GROUP is not None:
        dist.destroy_process_group(_CPU_GROUP)
        _CPU_GROUP = None
    dist.destroy_process_group()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _render(tokenizer: Any, example: dict[str, Any], include_answer: bool, tokenize: bool) -> Any:
    kwargs = {"tokenize": tokenize, "add_generation_prompt": not include_answer}
    try:
        return tokenizer.apply_chat_template(messages(example, include_answer), **kwargs)
    except (ValueError, RuntimeError, TypeError):
        fallback = [{"role": "user", "content": f"{SYSTEMS[example['task']]}\n\n{user_prompt(example['task'], example['source'])}"}]
        if include_answer:
            fallback.append({"role": "assistant", "content": example["target"]})
        return tokenizer.apply_chat_template(fallback, **kwargs)


def _tokenize_training(tokenizer: Any, rows: list[dict[str, Any]], context_cap: int) -> tuple[list[dict[str, Any]], int]:
    encoded, maximum = [], 0
    for row in rows:
        prompt = _render(tokenizer, row, False, True)
        full = _render(tokenizer, row, True, True)
        prompt = prompt["input_ids"] if isinstance(prompt, dict) else prompt
        full = full["input_ids"] if isinstance(full, dict) else full
        if full[: len(prompt)] != prompt:
            raise RuntimeError(f"assistant boundary is not a prompt prefix for {row['id']}")
        maximum = max(maximum, len(full))
        if len(full) > context_cap:
            raise RuntimeError(f"zero-truncation invariant failed for {row['id']}: {len(full)} > {context_cap}")
        encoded.append({"input_ids": full, "attention_mask": [1] * len(full), "labels": [-100] * len(prompt) + full[len(prompt):]})
    return encoded, maximum


def _collator(pad_id: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in batch)
        result: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in batch:
            padding = width - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [pad_id] * padding)
            result["attention_mask"].append(row["attention_mask"] + [0] * padding)
            result["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}
    return collate


def _native_generate(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_new_tokens: int, device: torch.device) -> list[str] | None:
    model.eval()
    local: list[tuple[int, str]] = []
    for index in range(_rank(), len(rows), _world()):
        prompt = _render(tokenizer, rows[index], False, True)
        prompt = prompt["input_ids"] if isinstance(prompt, dict) else prompt
        input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
        local.append((index, tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)))
    if _distributed():
        gathered: list[Any] = [None for _ in range(_world())]
        # Validation shards can finish far apart on variable-length generation.
        # Keep Python-object exchange off CUDA and outside NCCL's short watchdog.
        dist.all_gather_object(gathered, local, group=_cpu_group())
        if _rank() != 0:
            return None
        merged = [item for part in gathered for item in part]
    else:
        merged = local
    return [text for _, text in sorted(merged)]


def _save_final(model: Any, tokenizer: Any, output: Path) -> None:
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    state = {key: value.detach().to("cpu", torch.bfloat16).contiguous() for key, value in model.state_dict().items()}
    model.save_pretrained(temporary, state_dict=state, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    # tokenizer.json already contains the corrected regex. Persisting the
    # original True flag makes downstream AutoTokenizer users (including the
    # pinned llama.cpp converter) try to patch that serialization a second time.
    mark_regex_fix_consumed(temporary)
    if output.exists():
        shutil.rmtree(output)
    temporary.replace(output)


def _save_checkpoint(model: Any, optimizer: Any, scheduler: Any, output: Path, state: dict[str, Any]) -> None:
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    # Clone so tied embedding/lm-head storage is representable in a plain full-state file.
    save_file({key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}, temporary / "model_state.safetensors")
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
    torch.save({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}, temporary / "rng.pt")
    (temporary / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    previous = output.with_name(output.name + ".previous")
    if previous.exists():
        shutil.rmtree(previous)
    if output.exists():
        output.replace(previous)
    temporary.replace(output)
    if previous.exists():
        shutil.rmtree(previous)


def _load_checkpoint(model: Any, optimizer: Any, scheduler: Any, output: Path, device: torch.device) -> dict[str, Any] | None:
    if not (output / "state.json").exists():
        return None
    model.load_state_dict(load_file(output / "model_state.safetensors"), strict=True)
    optimizer.load_state_dict(torch.load(output / "optimizer.pt", map_location=device, weights_only=False))
    scheduler.load_state_dict(torch.load(output / "scheduler.pt", map_location=device, weights_only=False))
    rng = torch.load(output / "rng.pt", map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state_all(rng["cuda"])
    return json.loads((output / "state.json").read_text())


def _train(cfg: dict[str, Any], splits: dict[str, Any], run_checkpoint: Path, status: StatusWriter | None) -> None:
    device = torch.device("cuda", _local_rank())
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"],
        fix_mistral_regex=bool(cfg["tokenizer_fix_mistral_regex"]),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    task_cfg = cfg["datasets"][cfg["task"]]
    encoded, maximum = _tokenize_training(tokenizer, splits["train"], int(task_cfg["context_cap"]))
    if _rank() == 0:
        (run_checkpoint / "token_lengths.json").write_text(json.dumps({"maximum_train_tokens": maximum, "context_cap": task_cfg["context_cap"], "truncated": 0}, indent=2, sort_keys=True) + "\n")
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.float32, attn_implementation=cfg["train"]["attention"]).to(device)
    model.config.use_cache = False
    dataset = TokenDataset(encoded)
    sampler = DistributedSampler(dataset, num_replicas=_world(), rank=_rank(), shuffle=True, seed=cfg["seed"]) if _distributed() else None
    microbatch = int(task_cfg.get("per_device_batch", cfg["train"]["per_device_batch"]))
    accumulation_steps = int(task_cfg.get("gradient_accumulation_steps", cfg["train"]["gradient_accumulation_steps"]))
    loader = DataLoader(dataset, batch_size=microbatch, sampler=sampler, shuffle=sampler is None, collate_fn=_collator(tokenizer.pad_token_id), num_workers=2, pin_memory=True)
    epochs = 1 if cfg["mode"].startswith("smoke") else int(cfg["train"]["epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]), betas=(float(cfg["train"]["beta1"]), float(cfg["train"]["beta2"])), eps=float(cfg["train"]["eps"]), weight_decay=float(cfg["train"]["weight_decay"]), fused=bool(cfg["train"]["fused_adamw"]))
    optimizer_steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(cfg["train"]["warmup_steps"]), num_training_steps=epochs * optimizer_steps_per_epoch)
    latest = run_checkpoint / "checkpoint_latest"
    state = _load_checkpoint(model, optimizer, scheduler, latest, device)
    start_epoch = 0 if state is None else int(state["completed_epochs"])
    best = -math.inf if state is None else float(state["best_metric"])
    history = [] if state is None else state["history"]
    pending_validation_epoch = None if state is None else state.get("pending_validation_epoch")
    wrapped = DistributedDataParallel(model, device_ids=[_local_rank()]) if _distributed() else model
    selection_metric = task_cfg["selection_metric"]
    for epoch in range(start_epoch, epochs):
        bare = wrapped.module if isinstance(wrapped, DistributedDataParallel) else wrapped
        train_loss = None
        if epoch_requires_training(epoch, pending_validation_epoch):
            if sampler is not None:
                sampler.set_epoch(epoch)
            wrapped.train()
            model.config.use_cache = False
            running = 0.0
            optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(loader):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                step_due = optimizer_step_due(batch_index, len(loader), accumulation_steps)
                sync_context = nullcontext() if step_due or not isinstance(wrapped, DistributedDataParallel) else wrapped.no_sync()
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = wrapped(**batch).loss
                    (loss / accumulation_divisor(batch_index, len(loader), accumulation_steps)).backward()
                if step_due:
                    torch.nn.utils.clip_grad_norm_(wrapped.parameters(), float(cfg["train"]["clip_grad_norm"]))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                running += float(loss.detach())
            train_loss = running / len(loader)
            if _rank() == 0:
                _save_checkpoint(bare, optimizer, scheduler, latest, {"completed_epochs": epoch, "pending_validation_epoch": epoch + 1, "pending_train_loss": train_loss, "total_epochs": epochs, "best_metric": best, "selection_metric": selection_metric, "history": history})
                if status is not None:
                    status.heartbeat(f"epoch {epoch + 1}/{epochs} trained; validating")
            pending_validation_epoch = epoch + 1
            _barrier()
        else:
            train_loss = float(state["pending_train_loss"])
        model.config.use_cache = True
        predictions = _native_generate(bare, tokenizer, splits["validation"], int(task_cfg["max_new_tokens"]), device)
        metric_value, metrics = 0.0, None
        if _rank() == 0:
            metrics = score_predictions(cfg["task"], splits["validation"], predictions)
            metric_value = float(metrics[selection_metric])
        metric_tensor = torch.tensor(metric_value, device=device)
        if _distributed():
            dist.broadcast(metric_tensor, src=0)
        metric_value = float(metric_tensor)
        if _rank() == 0:
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation": metrics, "selection_metric": selection_metric, "selection_value": metric_value})
            if metric_value > best:
                best = metric_value
                _save_final(bare, tokenizer, run_checkpoint / "model_final")
            _save_checkpoint(bare, optimizer, scheduler, latest, {"completed_epochs": epoch + 1, "pending_validation_epoch": None, "total_epochs": epochs, "best_metric": best, "selection_metric": selection_metric, "history": history})
            (run_checkpoint / "training_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
            if status is not None:
                status.heartbeat(f"epoch {epoch + 1}/{epochs}; {selection_metric}={metric_value:.6g}")
        pending_validation_epoch = None
        _barrier()


def _apply_qv(receiver_dir: Path, qv_path: Path, alpha: float, output: Path) -> None:
    model = AutoModelForCausalLM.from_pretrained(receiver_dir, torch_dtype=torch.bfloat16, device_map="cpu")
    state, seen, covered_storage = model.state_dict(), set(), set()
    with safe_open(qv_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key not in state:
                raise RuntimeError(f"QV tensor missing from receiver: {key}")
            tensor, delta = state[key], handle.get_tensor(key)
            if tensor.shape != delta.shape or not tensor.is_floating_point():
                raise RuntimeError(f"QV tensor incompatible with receiver: {key}")
            storage_pointer = tensor.untyped_storage().data_ptr()
            if storage_pointer in covered_storage:
                seen.add(key)
                continue
            compute_dtype = torch.float64 if delta.dtype == torch.float64 else torch.float32
            tensor.copy_((tensor.to(compute_dtype) + alpha * delta).to(tensor.dtype))
            seen.add(key)
            covered_storage.add(storage_pointer)
    missing = [key for key, value in state.items() if value.is_floating_point() and value.untyped_storage().data_ptr() not in covered_storage]
    if missing:
        raise RuntimeError(f"receiver floating-state coverage mismatch: missing={sorted(missing)[:5]}")
    if output.exists():
        shutil.rmtree(output)
    model.save_pretrained(output, safe_serialization=True)
    for source in receiver_dir.iterdir():
        if source.is_file() and (
            source.name.startswith("tokenizer")
            or source.name.startswith("special_tokens")
            or source.name.startswith("added_tokens")
            or source.name.startswith("chat_template")
        ):
            shutil.copy2(source, output / source.name)


def _run_checked(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)


def _convert_and_quantize(cfg: dict[str, Any], run_checkpoint: Path, eval_dir: Path) -> dict[str, Path]:
    llama_dir = Path(cfg["paths"]["llama_cpp_dir"]).resolve()
    converter, quantizer = llama_dir / "convert_hf_to_gguf.py", llama_dir / "build/bin/llama-quantize"
    converter_compat = HERE / "convert_hf_to_gguf_compat.py"
    if not converter.exists() or not quantizer.exists():
        raise FileNotFoundError(f"llama.cpp b9637 is not prepared at {llama_dir}")
    receiver, patched = run_checkpoint / "model_final", run_checkpoint / "receiver_plus_qv_hf.tmp"
    _apply_qv(receiver, Path(cfg["paths"]["qv_dir"]) / "qv.safetensors", float(cfg["alpha"]), patched)
    outputs = {"receiver_bf16": run_checkpoint / "receiver_bf16.tmp.gguf", "receiver_q4_0": run_checkpoint / "receiver_q4_0.gguf", "patched_bf16": run_checkpoint / "receiver_plus_qv_bf16.tmp.gguf", "patched_q4_0": run_checkpoint / "receiver_plus_qv_q4_0.gguf"}
    converter_env = os.environ.copy()
    converter_env["PYTHONPATH"] = str(llama_dir) + os.pathsep + converter_env.get("PYTHONPATH", "")
    for source, target, name in ((receiver, outputs["receiver_bf16"], "receiver"), (patched, outputs["patched_bf16"], "patched")):
        if not target.exists():
            _run_checked([sys.executable, str(converter_compat), str(converter), str(source), "--outfile", str(target), "--outtype", "bf16"], eval_dir / f"convert_{name}.log", env=converter_env)
    for source, target, name in ((outputs["receiver_bf16"], outputs["receiver_q4_0"], "receiver"), (outputs["patched_bf16"], outputs["patched_q4_0"], "patched")):
        if not target.exists():
            _run_checked([str(quantizer), str(source), str(target), "Q4_0"], eval_dir / f"quantize_{name}.log")
    return outputs


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _completion(url: str, prompt: str, cfg: dict[str, Any], max_tokens: int) -> str:
    response = requests.post(url + "/completion", json={"prompt": prompt, "n_predict": max_tokens, "temperature": float(cfg["evaluation"]["temperature"]), "seed": int(cfg["seed"]), "cache_prompt": True}, timeout=900)
    response.raise_for_status()
    return response.json()["content"]


def _llama_predictions(cfg: dict[str, Any], model_path: Path, rows: list[dict[str, Any]], tokenizer: Any, log_path: Path) -> list[str]:
    server = Path(cfg["paths"]["llama_cpp_dir"]).resolve() / "build/bin/llama-server"
    port, env = _free_port(), os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    command = [str(server), "-m", str(model_path), "-ngl", "99", "--host", "127.0.0.1", "--port", str(port), "-c", str(cfg["evaluation"]["llama_context"]), "-np", str(cfg["evaluation"]["parallel_requests"])]
    prompts = [_render(tokenizer, row, False, False) for row in rows]
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + float(cfg["evaluation"]["server_start_timeout_s"])
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited early; see {log_path}")
                try:
                    if requests.get(url + "/health", timeout=2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"llama-server did not become healthy; see {log_path}")
            results: list[str | None] = [None] * len(rows)
            with ThreadPoolExecutor(max_workers=int(cfg["evaluation"]["parallel_requests"])) as pool:
                futures = {pool.submit(_completion, url, prompt, cfg, int(cfg["datasets"][cfg["task"]]["max_new_tokens"])): index for index, prompt in enumerate(prompts)}
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            return [str(value) for value in results]
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _evaluate(cfg: dict[str, Any], splits: dict[str, Any], run_checkpoint: Path, eval_dir: Path, status: StatusWriter) -> None:
    golden = eval_dir / "eval_results.json"
    if golden.exists():
        status.heartbeat("golden evaluation already present")
        return
    model_paths = _convert_and_quantize(cfg, run_checkpoint, eval_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        run_checkpoint / "model_final",
        # The source load was already fixed before serialization; reapplying
        # the patch to the serialized single Split pre-tokenizer is invalid.
        fix_mistral_regex=False,
    )
    rows, conditions = splits["test"], {}
    for index, (name, model_path) in enumerate(model_paths.items(), start=1):
        prediction_path = eval_dir / f"{name}.predictions.jsonl"
        if prediction_path.exists():
            predictions = [json.loads(line)["prediction"] for line in prediction_path.read_text().splitlines()]
        else:
            predictions = _llama_predictions(cfg, model_path, rows, tokenizer, eval_dir / f"llama_server_{name}.log")
            temporary = prediction_path.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps({"id": row["id"], "prediction": prediction, "references": row["references"]}, ensure_ascii=False) + "\n" for row, prediction in zip(rows, predictions)))
            temporary.replace(prediction_path)
        conditions[name] = {"model_path": str(model_path), "metrics": score_predictions(cfg["task"], rows, predictions), "predictions": str(prediction_path)}
        status.heartbeat(f"evaluation condition {index}/{len(model_paths)}: {name}")
    metric = cfg["datasets"][cfg["task"]]["selection_metric"]
    values = {name: float(record["metrics"][metric]) for name, record in conditions.items()}
    result = {"task": cfg["task"], "mode": cfg["mode"], "selection_metric": metric, "test_examples": len(rows), "conditions": conditions, "contrasts": {"ptq_degradation": values["receiver_q4_0"] - values["receiver_bf16"], "bf16_qv_effect": values["patched_bf16"] - values["receiver_bf16"], "quantized_qv_gain": values["patched_q4_0"] - values["receiver_q4_0"]}, "optional_metrics": {"gem_metrics_revision": cfg["evaluation"]["gem_metrics_revision"], "cider_nist": "null when unavailable; non-blocking by design"}}
    temporary = golden.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(golden)
    shutil.rmtree(run_checkpoint / "receiver_plus_qv_hf.tmp", ignore_errors=True)
    for name in ("receiver_bf16.tmp.gguf", "receiver_plus_qv_bf16.tmp.gguf"):
        path = run_checkpoint / name
        if path.exists():
            path.unlink()


def _main(cfg: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if _distributed():
        _init_distributed(int(cfg["train"]["distributed_timeout_s"]))
    _seed_all(int(cfg["seed"]) + _rank())
    run_path = run_id_path(cfg, RUN_ID_PARAMS)
    run_checkpoint = Path(cfg["paths"]["checkpoint_root"]) / run_path
    eval_dir = Path(cfg["paths"]["evaluation_root"]) / run_path
    if _rank() == 0:
        run_checkpoint.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        guard_run_config(cfg, RUN_ID_PARAMS, eval_dir)
        source = {"source_revision": os.environ.get("SOURCE_REVISION"), "source_tag": os.environ.get("SOURCE_TAG"), "wave_id": os.environ.get("WAVE_ID"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "hostname": socket.gethostname()}
        (eval_dir / "environment.json").write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    _barrier()
    splits = load_task_data(cfg["task"], cfg["datasets"][cfg["task"]], int(cfg["seed"]), cfg["mode"], eval_dir / "data_manifest.json" if _rank() == 0 else None)
    _barrier()
    with (StatusWriter(eval_dir) if _rank() == 0 else nullcontext(None)) as status:
        _train(cfg, splits, run_checkpoint, status)
        torch.cuda.empty_cache()
        _barrier()
        if _rank() != 0:
            if _distributed():
                _destroy_distributed()
            return
        if _distributed():
            _destroy_distributed()
        status.heartbeat("training complete; converting and evaluating")
        _evaluate(cfg, splits, run_checkpoint, eval_dir, status)


@hydra.main(version_base=None, config_path="../../../../../config/experiments/text/google_gemma3_causallm/001_qat_transfer", config_name="run_task")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    _main(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
