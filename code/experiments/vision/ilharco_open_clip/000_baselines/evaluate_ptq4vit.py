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
from src.ptq4vit import apply_ptq4vit_
from src.ptq4vit.net_wrap import _TupleLoaderWrapper

import hydra
from omegaconf import DictConfig
from rich.pretty import pprint
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset


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
    config_name="evaluate_ptq4vit",
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
        "PTQ4ViT quantization",
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

    calib_num_samples = int(cfg.ptq4vit.calib_num_samples)
    calib_batch_size = int(cfg.ptq4vit.calib_batch_size)

    calib_subset = Subset(dataset.val_dataset, list(range(min(calib_num_samples, len(dataset.val_dataset)))))
    raw_calib_loader = DataLoader(
        calib_subset,
        batch_size=calib_batch_size,
        shuffle=False,
        num_workers=int(os.environ['TORCH_NUM_WORKERS']),
    )
    calib_loader = _TupleLoaderWrapper(raw_calib_loader, maybe_dictionarize)

    print(f"\nCalibration: {len(calib_subset)} samples, batch_size={calib_batch_size}")

    pipeline_bar.update(1)

    ############################################################################
    # END calibration data
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN PTQ4ViT quantization
    ############################################################################

    pipeline_bar.set_postfix_str(pipeline_steps[4])

    skip_modules = frozenset(cfg.ptq4vit.skip_modules)

    print(
        f"\nPTQ4ViT config: bit={cfg.ptq4vit.bit}, metric={cfg.ptq4vit.metric}, "
        f"calibrator={cfg.ptq4vit.calibrator}, "
        f"skip_modules={list(cfg.ptq4vit.skip_modules)}"
    )

    quantized_names = apply_ptq4vit_(
        model=image_classifier,
        calib_loader=calib_loader,
        cfg=cfg.ptq4vit,
        skip_modules=skip_modules,
        device=device,
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
    # END PTQ4ViT quantization
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

    print(f"\n    eval test_accuracy (PTQ4ViT): {test_accuracy}\n")

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
        "-".join(sorted(cfg.ptq4vit.skip_modules))
        if len(cfg.ptq4vit.skip_modules) > 0
        else "none"
    )

    eval_dir = os.path.join(
        evaluation_base_path,
        "vision",
        "ilharco_open_clip",
        "000_baselines",
        "vision",
        "fp_ptq4vit",
        sanitized_model,
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"ptq4vit=bit={cfg.ptq4vit.bit}_metric={cfg.ptq4vit.metric}_cal={cfg.ptq4vit.calibrator}_sr={cfg.ptq4vit.search_round}_eqa={cfg.ptq4vit.eq_alpha}_eqb={cfg.ptq4vit.eq_beta}_eqn={cfg.ptq4vit.eq_n}_skip={skip_modules_tag}_cns={cfg.ptq4vit.calib_num_samples}_cbs={cfg.ptq4vit.calib_batch_size}_hbs={cfg.ptq4vit.hessian_batch_size}",
        f"nV={cfg.ptq4vit.n_V}_nH={cfg.ptq4vit.n_H}_na={cfg.ptq4vit.n_a}_nGA={cfg.ptq4vit.n_G_A}_nVA={cfg.ptq4vit.n_V_A}_nHA={cfg.ptq4vit.n_H_A}_nGB={cfg.ptq4vit.n_G_B}_nVB={cfg.ptq4vit.n_V_B}_nHB={cfg.ptq4vit.n_H_B}_nosm={cfg.ptq4vit.no_softmax}_nopg={cfg.ptq4vit.no_postgelu}_bc={cfg.ptq4vit.bias_correction}",
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
        "ptq4vit_bit": int(cfg.ptq4vit.bit),
        "ptq4vit_metric": str(cfg.ptq4vit.metric),
        "ptq4vit_calibrator": str(cfg.ptq4vit.calibrator),
        "ptq4vit_search_round": int(cfg.ptq4vit.search_round),
        "ptq4vit_eq_alpha": float(cfg.ptq4vit.eq_alpha),
        "ptq4vit_eq_beta": float(cfg.ptq4vit.eq_beta),
        "ptq4vit_eq_n": int(cfg.ptq4vit.eq_n),
        "ptq4vit_skip_modules": list(cfg.ptq4vit.skip_modules),
        "ptq4vit_calib_num_samples": int(cfg.ptq4vit.calib_num_samples),
        "ptq4vit_calib_batch_size": int(cfg.ptq4vit.calib_batch_size),
        "ptq4vit_hessian_batch_size": int(cfg.ptq4vit.hessian_batch_size),
        "ptq4vit_no_softmax": bool(cfg.ptq4vit.no_softmax),
        "ptq4vit_no_postgelu": bool(cfg.ptq4vit.no_postgelu),
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
