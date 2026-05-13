import json
import logging
import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.data.registry import get_dataset
from src.vision.data.common import (
    DATASET_NAME_TO_NUM_CLASSES,
    DATASET_NAME_TO_EPOCHS,
    maybe_dictionarize,
)
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)
from src.aphq_vit import apply_aphq_vit_, TupleLoaderWrapper

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)


_PATH_KEYS = {
    "w_bit", "a_bit", "qhead_a_bit", "qconv_a_bit",
    "reconstruct_mlp", "optimize",
    "calib_size", "calib_batch_size", "calib_metric",
    "optim_size", "optim_batch_size",
    "recon_metric", "pct",
    "optim_metric", "optim_mode", "drop_prob",
    "use_mean_hessian", "temp",
    "eq_n", "search_round",
    "matmul_head_channel_wise", "token_channel_wise",
    "skip_modules",
}
_IGNORED_FROM_PATH = {"keep_gpu"}


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
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/000_baselines",
    config_name="evaluate_fp_aphq_vit",
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

    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(OmegaConf.to_container(cfg, resolve=True), expand_all=True)

    # Drift guard: every cfg.aphq_vit key must either appear in the eval_dir
    # path or be explicitly excluded as runtime-only. Catches future
    # contributors silently overwriting results when adding new hparams.
    aphq_keys = set(OmegaConf.to_container(cfg.aphq_vit, resolve=True).keys())
    assert aphq_keys == _PATH_KEYS | _IGNORED_FROM_PATH, (
        "cfg.aphq_vit keys drifted from eval_dir path encoding. "
        f"unexpected={aphq_keys - (_PATH_KEYS | _IGNORED_FROM_PATH)}, "
        f"missing={(_PATH_KEYS | _IGNORED_FROM_PATH) - aphq_keys}. "
        "Update _PATH_KEYS/_IGNORED_FROM_PATH and the eval_dir path encoding "
        "to avoid silent result overwrites."
    )

    pipeline_steps = [
        "Loading checkpoint",
        "Creating dataset",
        "Preparing calibration data",
        "APHQ-ViT quantization",
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
        "ilharco_timm_supervised",
        "fp",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"seed={cfg.seed}",
    ]
    checkpoint_dir = os.path.join(*checkpoint_dir_parts)
    classifier_path = os.path.join(checkpoint_dir, f"classifier_epoch_{epochs}.pt")
    head_path = os.path.join(checkpoint_dir, f"head_epoch_{epochs}.pt")

    print(f"\nLoading classifier from: {classifier_path}")
    image_classifier = ImageClassifier.load(
        model_name=cfg.model_name,
        num_classes=DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name],
        filename=classifier_path,
    )
    image_classifier.to(device)

    print(f"\n\nimage_classifier:")
    pprint(image_classifier, expand_all=True)
    print(f"\n\n")
    if cfg.log_to_file:
        log.info(f"image_classifier:\n{image_classifier}")

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
        preprocess_train=image_classifier.train_preprocess,
        preprocess_inference=image_classifier.val_preprocess,
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
    # BEGIN calibration data
    ############################################################################

    pipeline_bar.set_postfix_str(pipeline_steps[2])

    if is_dryrun:
        aphq_cfg_effective = OmegaConf.merge(
            cfg.aphq_vit,
            OmegaConf.create({
                "calib_size": 2 * int(cfg.aphq_vit.calib_batch_size),
                "reconstruct_mlp": False,
                "optimize": False,
            }),
        )
        msg = (
            f"[DRY-RUN] APHQ-ViT overrides: calib_size={aphq_cfg_effective.calib_size}, "
            "reconstruct_mlp=False, optimize=False"
        )
        print(f"\n{msg}")
        if IS_SLURM:
            log.info(msg)
    else:
        aphq_cfg_effective = cfg.aphq_vit

    calib_size = int(aphq_cfg_effective.calib_size)
    calib_batch_size = int(aphq_cfg_effective.calib_batch_size)
    optim_size = int(aphq_cfg_effective.optim_size)
    optim_batch_size = int(aphq_cfg_effective.optim_batch_size)

    val_len = len(dataset.val_dataset)
    calib_subset = Subset(dataset.val_dataset, list(range(min(calib_size, val_len))))
    optim_subset = Subset(dataset.val_dataset, list(range(min(optim_size, val_len))))

    raw_calib_loader = DataLoader(
        calib_subset,
        batch_size=calib_batch_size,
        shuffle=False,
        num_workers=int(os.environ['TORCH_NUM_WORKERS']),
    )
    raw_optim_loader = DataLoader(
        optim_subset,
        batch_size=optim_batch_size,
        shuffle=False,
        num_workers=int(os.environ['TORCH_NUM_WORKERS']),
    )
    calib_loader = TupleLoaderWrapper(raw_calib_loader, maybe_dictionarize)
    optim_loader = TupleLoaderWrapper(raw_optim_loader, maybe_dictionarize)

    print(
        f"\nCalibration (Stage 2): {len(calib_subset)} samples, batch_size={calib_batch_size}"
    )
    print(
        f"Optimization (Stages 1/3): {len(optim_subset)} samples, batch_size={optim_batch_size}"
    )

    pipeline_bar.update(1)

    ############################################################################
    # END calibration data
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN APHQ-ViT quantization
    ############################################################################

    pipeline_bar.set_postfix_str(pipeline_steps[3])

    skip_modules = frozenset(aphq_cfg_effective.skip_modules)

    print(
        f"\nAPHQ-ViT config: w_bit={aphq_cfg_effective.w_bit}, a_bit={aphq_cfg_effective.a_bit}, "
        f"qhead_a_bit={aphq_cfg_effective.qhead_a_bit}, qconv_a_bit={aphq_cfg_effective.qconv_a_bit}, "
        f"reconstruct_mlp={aphq_cfg_effective.reconstruct_mlp}, optimize={aphq_cfg_effective.optimize}, "
        f"skip_modules={list(aphq_cfg_effective.skip_modules)}"
    )

    quantized_names = apply_aphq_vit_(
        image_classifier=image_classifier,
        model_name=cfg.model_name,
        calib_loader=calib_loader,
        optim_loader=optim_loader,
        aphq_cfg=aphq_cfg_effective,
        skip_modules=skip_modules,
        device=device,
    )

    all_quantizable_names = [
        name for name, m in image_classifier.named_modules()
        if isinstance(m, (nn.Linear, nn.Conv2d))
    ]
    quantized_set = set(quantized_names)
    skipped_names = sorted(
        n for n in all_quantizable_names if not any(n.endswith(q) or q.endswith(n) for q in quantized_set)
    )

    print(f"\nQuantized modules ({len(quantized_names)}):")
    for name in quantized_names:
        print(f"  - {name}")
    print(f"\nSkipped (unquantized) candidates ({len(skipped_names)}):")
    for name in skipped_names:
        print(f"  - {name}")
    print()

    if cfg.log_to_file:
        log.info(f"Quantized modules ({len(quantized_names)}): {quantized_names}")
        log.info(f"Skipped modules ({len(skipped_names)}): {skipped_names}")

    pipeline_bar.update(1)

    ############################################################################
    # END APHQ-ViT quantization
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN evaluation
    ############################################################################

    pipeline_bar.set_postfix_str(pipeline_steps[4])

    test_accuracy = evaluate(
        dataset=dataset,
        model=image_classifier,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
    )

    print(f"\n    eval test_accuracy (APHQ-ViT): {test_accuracy}\n")

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

    pipeline_bar.set_postfix_str(pipeline_steps[5])

    evaluation_base_path = os.environ['EVALUATION_BASE_PATH']

    skip_modules_tag = (
        "-".join(sorted(aphq_cfg_effective.skip_modules))
        if len(aphq_cfg_effective.skip_modules) > 0
        else "none"
    )

    aphq_dir_a = (
        f"aphq_vit=wbit={aphq_cfg_effective.w_bit}_abit={aphq_cfg_effective.a_bit}"
        f"_qha={aphq_cfg_effective.qhead_a_bit}_qca={aphq_cfg_effective.qconv_a_bit}"
        f"_recon={aphq_cfg_effective.reconstruct_mlp}_opt={aphq_cfg_effective.optimize}"
        f"_cs={aphq_cfg_effective.calib_size}_cbs={aphq_cfg_effective.calib_batch_size}"
        f"_cm={aphq_cfg_effective.calib_metric}"
        f"_os={aphq_cfg_effective.optim_size}_obs={aphq_cfg_effective.optim_batch_size}"
    )
    aphq_dir_b = (
        f"rm={aphq_cfg_effective.recon_metric}_pct={aphq_cfg_effective.pct}"
        f"_oM={aphq_cfg_effective.optim_metric}_om={aphq_cfg_effective.optim_mode}"
        f"_dp={aphq_cfg_effective.drop_prob}_umh={aphq_cfg_effective.use_mean_hessian}"
        f"_temp={aphq_cfg_effective.temp}"
        f"_eqn={aphq_cfg_effective.eq_n}_sr={aphq_cfg_effective.search_round}"
        f"_mhcw={aphq_cfg_effective.matmul_head_channel_wise}"
        f"_tcw={aphq_cfg_effective.token_channel_wise}"
        f"_skip={skip_modules_tag}"
    )

    eval_dir = os.path.join(
        evaluation_base_path,
        "vision",
        "ilharco_timm_supervised",
        "000_baselines",
        "vision",
        "fp_aphq_vit_dryrun" if is_dryrun else "fp_aphq_vit",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        aphq_dir_a,
        aphq_dir_b,
        f"seed={cfg.seed}",
    )

    results = {
        "model_name": cfg.model_name,
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
        "is_dryrun": is_dryrun,
        "epochs": epochs,
        "device": str(device),
        "test_accuracy": test_accuracy,
        "random_chance": random_chance,
        "num_classes": num_classes,
        "encoder_path": classifier_path,
        "head_path": head_path,
        "aphq_vit_w_bit": int(aphq_cfg_effective.w_bit),
        "aphq_vit_a_bit": int(aphq_cfg_effective.a_bit),
        "aphq_vit_qhead_a_bit": int(aphq_cfg_effective.qhead_a_bit),
        "aphq_vit_qconv_a_bit": int(aphq_cfg_effective.qconv_a_bit),
        "aphq_vit_reconstruct_mlp": bool(aphq_cfg_effective.reconstruct_mlp),
        "aphq_vit_optimize": bool(aphq_cfg_effective.optimize),
        "aphq_vit_calib_size": int(aphq_cfg_effective.calib_size),
        "aphq_vit_calib_batch_size": int(aphq_cfg_effective.calib_batch_size),
        "aphq_vit_calib_metric": str(aphq_cfg_effective.calib_metric),
        "aphq_vit_optim_size": int(aphq_cfg_effective.optim_size),
        "aphq_vit_optim_batch_size": int(aphq_cfg_effective.optim_batch_size),
        "aphq_vit_recon_metric": str(aphq_cfg_effective.recon_metric),
        "aphq_vit_pct": float(aphq_cfg_effective.pct),
        "aphq_vit_optim_metric": str(aphq_cfg_effective.optim_metric),
        "aphq_vit_optim_mode": str(aphq_cfg_effective.optim_mode),
        "aphq_vit_drop_prob": float(aphq_cfg_effective.drop_prob),
        "aphq_vit_use_mean_hessian": bool(aphq_cfg_effective.use_mean_hessian),
        "aphq_vit_temp": float(aphq_cfg_effective.temp),
        "aphq_vit_keep_gpu": bool(aphq_cfg_effective.keep_gpu),
        "aphq_vit_eq_n": int(aphq_cfg_effective.eq_n),
        "aphq_vit_search_round": int(aphq_cfg_effective.search_round),
        "aphq_vit_matmul_head_channel_wise": bool(aphq_cfg_effective.matmul_head_channel_wise),
        "aphq_vit_token_channel_wise": bool(aphq_cfg_effective.token_channel_wise),
        "aphq_vit_skip_modules": list(aphq_cfg_effective.skip_modules),
        "quantized_modules": quantized_names,
        "skipped_modules": skipped_names,
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
