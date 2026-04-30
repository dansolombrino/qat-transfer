import json
import logging
import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

from src.vision.ilharco_open_clip.modeling import ImageClassifier, ImageEncoder
from src.vision.ilharco_open_clip.heads import get_classification_head
from src.vision.data.registry import get_dataset
from src.vision.data.common import (
    DATASET_NAME_TO_EPOCHS,
    maybe_dictionarize,
)
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_open_clip_model_name,
    set_seed,
)
from src.repqvit import apply_repqvit_

import hydra
from omegaconf import DictConfig
from rich.pretty import pprint
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)


def evaluate(
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    limit_num_batches: int = None,
):
    loader = dataset.test_loader

    num_batches = len(loader)
    effective_num_batches = (
        min(limit_num_batches, num_batches) if limit_num_batches is not None else num_batches
    )

    model.to(device=device)
    model.eval()

    correct = 0
    total = 0

    batch_color = random_tqdm_color()

    with torch.no_grad():
        batch_bar = tqdm(
            enumerate(loader),
            total=effective_num_batches,
            desc="Evaluating (test)",
            colour=batch_color,
            leave=False,
            **TQDM_KW,
        )

        for i, batch in batch_bar:
            if i >= effective_num_batches:
                break

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to(device=device)
            labels = batch['labels'].to(device=device, dtype=torch.long)

            logits = model(inputs)

            top1, = accuracy(logits, labels, topk=(1,))

            correct += top1
            total += labels.size(0)

            batch_bar.set_postfix(
                batch=f"{i}/{effective_num_batches}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

    return correct / total


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_open_clip/000_baselines",
    config_name="evaluate_fp_repqvit",
    version_base=None,
)
def main(cfg: DictConfig):

    set_seed(cfg.seed)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    epochs = (
        DATASET_NAME_TO_EPOCHS[cfg.dataset_name]
        if cfg.limit_num_epochs is None
        else cfg.limit_num_epochs
    )

    sanitized_model = sanitize_open_clip_model_name(cfg.model_name, cfg.pretrained)

    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)

    pipeline_steps = [
        "Loading checkpoint",
        "Creating dataset",
        "Building classifier",
        "Preparing calibration data",
        "RepQ-ViT quantization",
        "Evaluating (test)",
        "Saving results",
    ]
    pipeline_color = random_tqdm_color()
    pipeline_bar = tqdm(
        pipeline_steps,
        desc="Pipeline",
        colour=pipeline_color,
        leave=True,
        **TQDM_KW,
    )

    ############################################################################
    # BEGIN checkpoint loading
    ############################################################################

    pipeline_bar.set_postfix_str(pipeline_steps[0])

    checkpoint_base_path = os.environ['CHECKPOINT_BASE_PATH']

    is_dryrun = cfg.limit_num_batches is not None or cfg.limit_num_epochs is not None
    checkpoint_dir_parts = [
        checkpoint_base_path,
        "vision",
        "ilharco_open_clip",
        "fp_dryrun" if is_dryrun else "fp",
        sanitized_model,
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"seed={cfg.seed}",
    ]
    if is_dryrun:
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
        checkpoint_dir_parts.append(f"lnb={lnb}_lne={lne}")
    checkpoint_dir = os.path.join(*checkpoint_dir_parts)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epochs}.pt")

    print(f"\nLoading encoder from: {checkpoint_path}")
    image_encoder = ImageEncoder.load(
        model_name=cfg.model_name,
        pretrained=cfg.pretrained,
        filename=checkpoint_path,
    )
    image_encoder.to(device)

    print(f"\n\nimage_encoder:")
    pprint(image_encoder, expand_all=True)
    print(f"\n\n")
    if cfg.log_to_file:
        log.info(f"image_encoder:\n{image_encoder}")

    pipeline_bar.update(1)

    ############################################################################
    # END checkpoint loading
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN dataset creation
    ############################################################################

    pipeline_bar.set_postfix_str(pipeline_steps[1])

    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=image_encoder.train_preprocess,
        preprocess_inference=image_encoder.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ['TORCH_NUM_WORKERS']),
        seed=cfg.seed,
    )

    pipeline_bar.update(1)

    ############################################################################
    # END dataset creation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN classifier creation
    ############################################################################

    pipeline_bar.set_postfix_str(pipeline_steps[2])

    head_base_path = os.environ['HEAD_BASE_PATH']

    classification_head = get_classification_head(
        model_name=cfg.model_name,
        pretrained=cfg.pretrained,
        dataset_name=cfg.dataset_name,
        save_dir=head_base_path,
        device=device,
    )

    image_classifier = ImageClassifier(
        image_encoder=image_encoder,
        classification_head=classification_head,
    )
    image_classifier.to(device)

    print(f"\n\nimage_classifier:")
    pprint(image_classifier, expand_all=True)
    print(f"\n\n")
    if cfg.log_to_file:
        log.info(f"image_classifier:\n{image_classifier}")

    pipeline_bar.update(1)

    ############################################################################
    # END classifier creation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN calibration data
    ############################################################################

    pipeline_bar.set_postfix_str(pipeline_steps[3])

    calib_loader = DataLoader(
        dataset.val_dataset,
        batch_size=cfg.repqvit.calib_batch_size,
        shuffle=False,
        num_workers=int(os.environ['TORCH_NUM_WORKERS']),
    )
    for batch in calib_loader:
        batch = maybe_dictionarize(batch)
        calib_data = batch['images']
        break

    print(f"\nCalibration batch (val split): {calib_data.shape} (calib_batch_size={cfg.repqvit.calib_batch_size})")

    pipeline_bar.update(1)

    ############################################################################
    # END calibration data
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN RepQ-ViT quantization
    ############################################################################

    pipeline_bar.set_postfix_str(pipeline_steps[4])

    skip_modules = frozenset(cfg.repqvit.skip_modules)

    print(
        f"\nRepQ-ViT config: w_bits={cfg.repqvit.w_bits}, a_bits={cfg.repqvit.a_bits}, "
        f"skip_modules={list(cfg.repqvit.skip_modules)}"
    )

    quantized_names = apply_repqvit_(
        model=image_classifier,
        calib_data=calib_data,
        w_bits=cfg.repqvit.w_bits,
        a_bits=cfg.repqvit.a_bits,
        skip_modules=skip_modules,
        family="open_clip",
        channel_wise_names=frozenset(("c_fc",)),
        tqdm_kw=TQDM_KW,
    )

    print(f"\nQuantized modules ({len(quantized_names)}):")
    for name in quantized_names:
        print(f"  - {name}")
    print()

    if cfg.log_to_file:
        log.info(f"Quantized modules ({len(quantized_names)}): {quantized_names}")

    pipeline_bar.update(1)

    ############################################################################
    # END RepQ-ViT quantization
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN evaluation
    ############################################################################

    pipeline_bar.set_postfix_str(pipeline_steps[5])

    test_accuracy = evaluate(
        dataset=dataset,
        model=image_classifier,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
    )

    print(f"\n    eval test_accuracy (RepQ-ViT): {test_accuracy}\n")

    num_classes = len(dataset.class_names)
    random_chance = 1.0 / num_classes
    print(f"    random chance baseline : {random_chance}  (1 / {num_classes} classes)\n")

    pipeline_bar.update(1)

    ############################################################################
    # END evaluation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN save results
    ############################################################################

    pipeline_bar.set_postfix_str(pipeline_steps[6])

    evaluation_base_path = os.environ['EVALUATION_BASE_PATH']

    skip_modules_tag = (
        "-".join(sorted(cfg.repqvit.skip_modules))
        if len(cfg.repqvit.skip_modules) > 0
        else "none"
    )

    eval_dir = os.path.join(
        evaluation_base_path,
        "vision",
        "ilharco_open_clip",
        "000_baselines",
        "vision",
        "fp_repqvit",
        sanitized_model,
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"repqvit=wbits={cfg.repqvit.w_bits}_abits={cfg.repqvit.a_bits}_skip={skip_modules_tag}_cbs={cfg.repqvit.calib_batch_size}",
        f"seed={cfg.seed}",
    )

    results = {
        "model_name": cfg.model_name,
        "pretrained": cfg.pretrained,
        "dataset_name": cfg.dataset_name,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "ls": cfg.ls,
        "wl": cfg.wl,
        "max_grad_norm": cfg.max_grad_norm,
        "seed": cfg.seed,
        "limit_num_epochs": cfg.limit_num_epochs,
        "limit_num_batches": cfg.limit_num_batches,
        "epochs": epochs,
        "device": str(device),
        "test_accuracy": test_accuracy,
        "random_chance": random_chance,
        "num_classes": num_classes,
        "checkpoint_path": checkpoint_path,
        "repqvit_w_bits": cfg.repqvit.w_bits,
        "repqvit_a_bits": cfg.repqvit.a_bits,
        "repqvit_skip_modules": list(cfg.repqvit.skip_modules),
        "repqvit_calib_batch_size": cfg.repqvit.calib_batch_size,
        "quantized_modules": quantized_names,
    }

    os.makedirs(eval_dir, exist_ok=True)
    eval_results_path = os.path.join(eval_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {eval_results_path}")

    pipeline_bar.update(1)
    pipeline_bar.close()

    ############################################################################
    # END save results
    ############################################################################


if __name__ == "__main__":
    main()
