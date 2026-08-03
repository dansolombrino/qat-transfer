"""Pinned generative tasks, deterministic selection, prompts, and metrics."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import sacrebleu
from datasets import load_dataset
from rouge_score import rouge_scorer

SYSTEMS = {
    "gsm8k": "You are a careful mathematical reasoner.",
    "samsum": "You summarize dialogues faithfully and concisely.",
    "e2e_nlg": "You verbalize structured data faithfully and naturally.",
}


def user_prompt(task: str, source: str) -> str:
    if task == "gsm8k":
        return "Solve the problem. Show your reasoning, then end with #### <answer>.\n\nProblem:\n" + source
    if task == "samsum":
        return "Summarize the following dialogue in one concise paragraph.\n\nDialogue:\n" + source
    if task == "e2e_nlg":
        return "Express the following meaning representation as one natural sentence.\n\nMeaning representation:\n" + source
    raise KeyError(task)


def messages(example: dict[str, Any], include_answer: bool) -> list[dict[str, str]]:
    result = [
        {"role": "system", "content": SYSTEMS[example["task"]]},
        {"role": "user", "content": user_prompt(example["task"], example["source"])},
    ]
    if include_answer:
        result.append({"role": "assistant", "content": example["target"]})
    return result


def _fingerprint(records: list[dict[str, Any]]) -> str:
    encoded = "\n".join(json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for record in records).encode()
    return hashlib.sha256(encoded).hexdigest()


def _choose(items: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)
    return [items[index] for index in indices[:count]]


def _gsm8k(dataset: Any, seed: int) -> dict[str, list[dict[str, Any]]]:
    train = [{"id": f"train:{i}", "task": "gsm8k", "source": row["question"], "target": row["answer"], "references": [row["answer"]]} for i, row in enumerate(dataset["train"])]
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    if len(order) != 7473:
        raise RuntimeError(f"pinned GSM8K train size changed: {len(order)}")
    test = [{"id": f"test:{i}", "task": "gsm8k", "source": row["question"], "target": row["answer"], "references": [row["answer"]]} for i, row in enumerate(dataset["test"])]
    return {"train": [train[i] for i in order[1024:7473]], "validation": [train[i] for i in order[:1024]], "test": test}


def _samsum(dataset: Any, seed: int) -> dict[str, list[dict[str, Any]]]:
    def convert(split: str) -> list[dict[str, Any]]:
        return [{"id": str(row.get("id", f"{split}:{i}")), "task": "samsum", "source": row["dialogue"], "target": row["summary"], "references": [row["summary"]]} for i, row in enumerate(dataset[split])]
    return {"train": _choose(convert("train"), 6449, seed), "validation": convert("validation"), "test": convert("test")}


def _e2e(dataset: Any, seed: int) -> dict[str, list[dict[str, Any]]]:
    def group(split: str) -> list[dict[str, Any]]:
        grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for row in dataset[split]:
            source = row.get("meaning_representation", row.get("mr"))
            target = row.get("target", row.get("ref"))
            if source is None or target is None:
                raise KeyError(f"unexpected E2E columns: {sorted(row)}")
            record = grouped.setdefault(source, {"target": target, "references": []})
            record["references"].extend(row.get("references") or [target])
        return [{"id": hashlib.sha256(source.encode()).hexdigest()[:20], "task": "e2e_nlg", "source": source, "target": record["target"], "references": record["references"]} for source, record in grouped.items()]
    train = group("train")
    if len(train) < 6449:
        raise RuntimeError(f"pinned E2E train has only {len(train)} distinct MRs; 6449 required")
    return {"train": _choose(train, 6449, seed), "validation": group("validation"), "test": group("test")}


def load_task_data(task: str, dataset_spec: dict[str, Any], seed: int, mode: str, manifest_path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    kwargs: dict[str, Any] = {"path": dataset_spec["id"], "revision": dataset_spec["revision"]}
    if dataset_spec.get("config"):
        kwargs["name"] = dataset_spec["config"]
    if task == "e2e_nlg":
        kwargs["trust_remote_code"] = True
    dataset = load_dataset(**kwargs)
    builders = {"gsm8k": _gsm8k, "samsum": _samsum, "e2e_nlg": _e2e}
    splits = builders[task](dataset, seed)
    if mode.startswith("smoke"):
        splits = {name: rows[: 16 if name == "train" else 8] for name, rows in splits.items()}
    manifest = {"task": task, "dataset_id": dataset_spec["id"], "dataset_revision": dataset_spec["revision"], "seed": seed, "mode": mode, "splits": {name: {"count": len(rows), "ids": [row["id"] for row in rows], "fingerprint": _fingerprint(rows)} for name, rows in splits.items()}}
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if manifest_path.exists() and manifest_path.read_text() != payload:
            raise RuntimeError(f"data manifest changed for existing run: {manifest_path}")
        manifest_path.write_text(payload)
    return splits


def _gsm_answer(text: str) -> str | None:
    matches = re.findall(r"####\s*([^\n]+)", text)
    if not matches:
        return None
    return re.sub(r"\s+", "", matches[-1].strip().replace(",", "")).rstrip(".")


def score_predictions(task: str, rows: list[dict[str, Any]], predictions: list[str]) -> dict[str, float | None]:
    if len(rows) != len(predictions):
        raise ValueError("row/prediction count mismatch")
    if task == "gsm8k":
        return {"gsm8k_em": sum(_gsm_answer(pred) == _gsm_answer(row["target"]) for row, pred in zip(rows, predictions)) / len(rows)}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge = {key: 0.0 for key in ("rouge1", "rouge2", "rougeL")}
    for row, prediction in zip(rows, predictions):
        candidates = [scorer.score(reference, prediction) for reference in row["references"]]
        for key in rouge:
            rouge[key] += max(candidate[key].fmeasure for candidate in candidates)
    result: dict[str, float | None] = {key: value / len(rows) for key, value in rouge.items()}
    if task == "e2e_nlg":
        max_refs = max(len(row["references"]) for row in rows)
        references = [[row["references"][min(index, len(row["references"]) - 1)] for row in rows] for index in range(max_refs)]
        result.update(bleu=sacrebleu.corpus_bleu(predictions, references).score, cider=None, nist=None)
    return result
