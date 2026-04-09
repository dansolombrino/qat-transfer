import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# MUST be the first thing that runs: HF libs (transformers, huggingface_hub,
# datasets) and our own src.vision.data.common all snapshot env vars at import
# time. Loading .env after those imports has no effect.
from dotenv import load_dotenv
load_dotenv()

import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm

log = logging.getLogger(__name__)

IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)
LOG_EVERY = 50

from src.vision.data.common import (
    DATASET_NAME_TO_NUM_CLASSES,
    DATASET_NAME_TO_EPOCHS,
    maybe_dictionarize
)
from src.vision.data.registry import get_dataset
from src.vision.ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head.modeling import ImageClassifier
from src.vision.utils import (
    LabelSmoothing,
    cosine_lr,
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)

OmegaConf.register_new_resolver(
    "sanitize_timm", sanitize_timm_model_name, replace=True
)


@hydra.main(
    config_path="../../../../config/src/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head",
    config_name="finetune_fp",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    checkpoint_base_path = os.environ["CHECKPOINT_BASE_PATH"]
    num_workers = int(os.environ["TORCH_NUM_WORKERS"])

    is_dryrun = (
        cfg.limit_num_batches is not None or cfg.limit_num_epochs is not None
    )
    save_dir_parts = [
        checkpoint_base_path,
        "vision",
        "ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head",
        "fp_dryrun" if is_dryrun else "fp",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"seed={cfg.seed}",
    ]
    if is_dryrun:
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
        save_dir_parts.append(f"lnb={lnb}_lne={lne}")
    save_dir = os.path.join(*save_dir_parts)
    os.makedirs(save_dir, exist_ok=True)

    classifier = ImageClassifier(
        model_name=cfg.model_name,
        num_classes=DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name]
    )

    # Freeze biases, cls_token, pos_embed, patch_embed, norms, head
    frozen_names = []
    trainable_names = []
    for name, param in classifier.named_parameters():
        if (
            name.endswith(".bias")
            or "cls_token" in name
            or "pos_embed" in name
            or "patch_embed" in name
            or "norm" in name
            or "head" in name
        ):
            param.requires_grad = False
            frozen_names.append(name)
        else:
            trainable_names.append(name)

    if IS_SLURM:
        log.info("Frozen parameters (%d): %s", len(frozen_names), frozen_names)
        log.info("Trainable parameters (%d): %s", len(trainable_names), trainable_names)
    else:
        print(f"\nFrozen parameters ({len(frozen_names)}):")
        pprint(frozen_names, expand_all=True)
        print(f"\nTrainable parameters ({len(trainable_names)}):")
        pprint(trainable_names, expand_all=True)

    # Dataset (seeded with the run seed — not SPLIT_SEED)
    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=classifier.train_preprocess,
        preprocess_inference=classifier.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        seed=cfg.seed,
    )

    classifier.to(device=device)

    if IS_SLURM:
        log.info("state_dict keys: %s", list(classifier.state_dict().keys()))
        log.info("classifier: %s", classifier)
    else:
        pprint(list(classifier.state_dict().keys()), expand_all=True)
        pprint(classifier, expand_all=True)

    epochs = DATASET_NAME_TO_EPOCHS[cfg.dataset_name]
    epochs = (
        min(epochs, cfg.limit_num_epochs)
        if cfg.limit_num_epochs is not None
        else epochs
    )
    num_batches = len(dataset.train_loader)

    # Optimizer / scheduler / loss
    params = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = cosine_lr(optimizer, cfg.lr, cfg.wl, epochs * num_batches)
    loss_fn = (
        LabelSmoothing(cfg.ls) if cfg.ls > 0 else torch.nn.CrossEntropyLoss()
    )

    # Training loop
    epoch_bar = tqdm(
        range(epochs), desc="epochs", colour=random_tqdm_color(), **TQDM_KW
    )
    for epoch in epoch_bar:
        classifier.train()
        train_bar = tqdm(
            dataset.train_loader,
            desc=f"train e{epoch}",
            colour=random_tqdm_color(),
            leave=False,
            **TQDM_KW,
        )
        for i, batch in enumerate(train_bar):
            if (
                cfg.limit_num_batches is not None
                and i >= cfg.limit_num_batches
            ):
                break
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device=device, dtype=torch.long)

            logits = classifier(images)
            loss: torch.Tensor = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()

            if IS_SLURM:
                if i % LOG_EVERY == 0 or i == num_batches - 1:
                    log.info(
                        "epoch %d step %d/%d loss=%.4f",
                        epoch, i, num_batches, loss.item(),
                    )
            else:
                train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Per-epoch validation
        classifier.eval()
        val_correct, val_total = 0, 0
        val_bar = tqdm(
            dataset.val_loader,
            desc=f"val e{epoch}",
            colour=random_tqdm_color(),
            leave=False,
            **TQDM_KW,
        )
        with torch.no_grad():
            for i, batch in enumerate(val_bar):
                if (
                    cfg.limit_num_batches is not None
                    and i >= cfg.limit_num_batches
                ):
                    break
                batch = maybe_dictionarize(batch)
                images = batch["images"].to(device)
                labels = batch["labels"].to(device=device, dtype=torch.long)
                logits = classifier(images)
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                val_total += labels.size(0)
                if not IS_SLURM:
                    val_bar.set_postfix(
                        acc=f"{val_correct / max(val_total, 1):.4f}"
                    )

        val_acc = val_correct / max(val_total, 1)
        if IS_SLURM:
            log.info("epoch %d val_acc=%.4f", epoch, val_acc)
        else:
            epoch_bar.set_postfix(val_acc=f"{val_acc:.4f}")

        # Save classifier and head checkpoints separately, if limit nu
        if cfg.limit_num_epochs:
            classifier_path = os.path.join(save_dir, f"classifier_epoch_{epoch + 1}.pt")
            classifier.save(classifier_path)
            head_path = os.path.join(save_dir, f"head_epoch_{epoch + 1}.pt")
            torch.save(classifier.model.head, head_path)

    # Final test evaluation
    classifier.eval()
    test_correct, test_total = 0, 0
    test_bar = tqdm(
        dataset.test_loader, desc="test", colour=random_tqdm_color(), **TQDM_KW
    )
    with torch.no_grad():
        for i, batch in enumerate(test_bar):
            if (
                cfg.limit_num_batches is not None
                and i >= cfg.limit_num_batches
            ):
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device=device, dtype=torch.long)
            logits = classifier(images)
            test_correct += (logits.argmax(dim=-1) == labels).sum().item()
            test_total += labels.size(0)
            if not IS_SLURM:
                test_bar.set_postfix(
                    acc=f"{test_correct / max(test_total, 1):.4f}"
                )

    test_acc = test_correct / max(test_total, 1)
    if IS_SLURM:
        log.info("final test accuracy: %.4f", test_acc)
    else:
        print(f"Final test accuracy: {test_acc:.4f}")

    # Save classifier and head checkpoints separately
    classifier_path = os.path.join(save_dir, f"classifier_epoch_{epochs}.pt")
    classifier.save(classifier_path)
    head_path = os.path.join(save_dir, f"head_epoch_{epochs}.pt")
    torch.save(classifier.model.head, head_path)


if __name__ == "__main__":
    main()
