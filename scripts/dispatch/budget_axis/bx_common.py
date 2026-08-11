"""Shared vocabulary for the training-budget-axis wave.

Why this module exists
----------------------
The wave has to answer one question about thousands of work items: *is this one
already done?*  Every wrong answer is expensive in a different direction -- a
false "done" silently leaves a hole in a heatmap, and a false "not done" burns a
GPU-hour recomputing what is already on disk.

Both failure modes come from the same root cause: a guard path spelled by hand
in a shell script drifting from the path the Python writer actually emits.  That
drift is silent by construction, because a guard that looks in the wrong place
simply never matches.  So the guards are not spelled by hand.  This module
imports the *same* helpers the writers use -- ``mult_path_frag``,
``role_path_frag``, ``checkpoint_epochs``, ``sanitize_timm_model_name`` -- and
reproduces each writer's ``eval_dir`` exactly once, here.  Workers ask this
module; they never build a path themselves.

The budget levels
-----------------
The axis is really two *step budgets*, not two multipliers.  ImageNet's 1x
schedule is 9,971 steps -- about 4.8x the median dataset's 2,030 -- so putting
every dataset on "the same multiplier" would compare a long run against a short
one.  Two levels, each meaning roughly the same number of optimizer steps for
every dataset:

    SHORT   ordinary datasets at mult=1     ImageNet at mult=0.25   (~2.5k steps)
    LONG    ordinary datasets at mult=4     ImageNet at mult=1      (~10k steps)

``4 * 0.25 == 1``, so the two legs meet exactly in the middle: the ordinary
datasets' LONG budget is ImageNet's native one, and ImageNet's SHORT budget is
theirs.  That is the whole point of the axis, and it is why the high multiplier
is 4 rather than a round 5.
"""

import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
os.chdir(_PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

from src.duration import checkpoint_epochs, mult_path_frag, role_path_frag

# ---------------------------------------------------------------------------
# Two values are needed from modules that import torch, and importing torch here
# is not affordable.
#
# This module is the guard every worker consults, and a worker rescans its whole
# queue each cycle -- with ~150 deferred items that is ~150 process starts. With
# the torch-importing modules in the chain each start cost 3.0 s (1,120 modules),
# so an evaluation worker spent minutes per cycle in Python startup instead of on
# the GPU; behemoth's utilisation fell to ~41% purely from this. `src.duration`
# alone costs 0.01 s, so the fix is to stop pulling in the other two.
#
# Correctness is preserved differently for each value rather than by trusting a
# copy: the epochs table is *parsed from its source file* with `ast`, so it cannot
# drift, and the sanitiser -- three literal replacements -- is reimplemented and
# checked against the real function by `bx_check.py verify-imports`.
# ---------------------------------------------------------------------------
import ast


def _load_epochs_table():
    """DATASET_NAME_TO_EPOCHS, read from source without importing torch."""
    src = (_CODE_DIR / "src" / "vision" / "data" / "common.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DATASET_NAME_TO_EPOCHS"
            for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("DATASET_NAME_TO_EPOCHS not found in common.py")


DATASET_NAME_TO_EPOCHS = _load_epochs_table()


def sanitize_timm_model_name(model_name):
    """Mirror of src.vision.utils.sanitize_timm_model_name (which imports torch).

    Verified equal to the original by `bx_check.py verify-imports`.
    """
    return model_name.replace("/", "_").replace("-", "_").replace(".", "_")

# ---------------------------------------------------------------------------
# The canonical configuration -- CLAUDE.md "Canonical experimental configuration"
# ---------------------------------------------------------------------------
MODEL = "vit_base_patch16_224.orig_in21k"
MODEL_SAN = sanitize_timm_model_name(MODEL)
SEED = 2038
BATCH_SIZE = 128
LR = "1e-05"          # as it appears in a path: Hydra's 1e-5 stringifies to 1e-05
WD = "0.1"
LS = "0.0"
WL = "500"
MGN = "1.0"
BITS = 3
GRAN = "channel"
SKIP = ("head",)

OPTIM_FRAG = f"optim=adamw_lr={LR}_wd={WD}_ls={LS}_wl={WL}_mgn={MGN}_bs={BATCH_SIZE}"
SKIP_TAG = "-".join(sorted(SKIP)) if SKIP else "none"
QAT_FRAG = f"qat=bits={BITS}_gran={GRAN}_skip={SKIP_TAG}"
PTQ_FRAG = f"ptq=bits={BITS}_gran={GRAN}_skip={SKIP_TAG}"
ALPHA = "1.0"
QV_FRAG = f"qv=alpha={ALPHA}"
SPLIT = "test"

IMAGENET = "ImageNet"
# Insertion order is the dataset registry's; ORDINARY is everything but ImageNet.
DATASETS = tuple(DATASET_NAME_TO_EPOCHS.keys())
ORDINARY = tuple(d for d in DATASETS if d != IMAGENET)

SHORT, LONG = "S", "L"
GRIDS = ((SHORT, SHORT), (LONG, LONG), (SHORT, LONG), (LONG, SHORT))
GRID_NAMES = {
    (SHORT, SHORT): "G_SS",
    (LONG, LONG): "G_LL",
    (SHORT, LONG): "G_SL",
    (LONG, SHORT): "G_LS",
}
# Lower runs first. Baselines lead because they are cheap and every heatmap reads
# them; then the two grids the user asked for; then the two cross-budget grids,
# so that stopping early costs only the questions that were not asked.
GRID_PRIORITY = {(SHORT, SHORT): 1, (LONG, LONG): 2, (SHORT, LONG): 3, (LONG, SHORT): 4}
BASELINE_PRIORITY = 0

BASELINE_VARIANTS = (
    "fp",
    "fp_ptq",
    "qat",
    "qat_ptq",
    "pretrained",
    "pretrained_ptq",
)


def level_mult(dataset, level):
    """The multiplier that puts `dataset` on the given step budget.

    ImageNet is the only dataset whose native schedule is already the LONG one,
    so it is the only one that moves in the opposite direction.
    """
    if dataset == IMAGENET:
        return "0.25" if level == SHORT else "1"
    return "1" if level == SHORT else "4"


# ---------------------------------------------------------------------------
# Bases.  These differ per rig: rig-3090-ti keeps checkpoints on /mnt/WD_4TB and
# evaluations on /mnt/KS_960GB -- different mounts.  Reading both from the
# environment is what makes one script correct on all three rigs.
# ---------------------------------------------------------------------------
def ckpt_base():
    return os.environ["CHECKPOINT_BASE_PATH"]


def eval_base():
    return os.environ["EVALUATION_BASE_PATH"]


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def ckpt_dir(kind, dataset, mult):
    """`kind` is "fp" or "qat".  Mirrors CLAUDE.md "Checkpoint paths"."""
    parts = [
        ckpt_base(), "vision", "ilharco_timm_supervised", kind,
        MODEL_SAN, dataset, OPTIM_FRAG, mult_path_frag(mult),
    ]
    if kind == "qat":
        parts.append(QAT_FRAG)
    parts.append(f"seed={SEED}")
    return os.path.join(*parts)


def ckpt_file(kind, dataset, mult):
    """The classifier checkpoint.

    The epoch in the filename names the *1x reference* schedule, not the
    realized run -- `checkpoint_epochs` is the reader-side helper that encodes
    that, and it is loader-independent so a code-only clone can name any
    checkpoint.  Spelling it any other way would break exactly that property.
    """
    n = checkpoint_epochs(dataset, DATASET_NAME_TO_EPOCHS)
    return os.path.join(ckpt_dir(kind, dataset, mult), f"classifier_epoch_{n}.pt")


def head_file(kind, dataset, mult):
    n = checkpoint_epochs(dataset, DATASET_NAME_TO_EPOCHS)
    return os.path.join(ckpt_dir(kind, dataset, mult), f"head_epoch_{n}.pt")


# A 343 MB `torch.save` is not atomic: the final path exists, and grows, for
# seconds before it is complete. A consumer that only tests existence can load a
# truncated checkpoint, and the replicator can copy one -- which is worse, since
# the truncated copy then looks complete on the peer. Requiring the file to have
# been untouched for a while converts that race into a short wait.
STABLE_AGE_S = 90


def _settled(path):
    try:
        return (os.path.isfile(path)
                and (time.time() - os.path.getmtime(path)) >= STABLE_AGE_S)
    except OSError:
        return False


def ckpt_ready(dataset, mult):
    """Both halves of the QV must exist at the same budget, and be settled.

    QV = QAT - FP, so a dataset with only one of the two is useless as a donor
    and only half-usable as a receiver.  Gating on the pair is what keeps
    `qv_transfer.py`'s exit-0-on-missing-checkpoint behaviour from writing a
    silently incomplete grid.
    """
    return (_settled(ckpt_file("fp", dataset, mult))
            and _settled(ckpt_file("qat", dataset, mult)))


def ckpt_exists(dataset, mult):
    """Existence without the settle wait -- for progress counting, not for reads."""
    return (os.path.isfile(ckpt_file("fp", dataset, mult))
            and os.path.isfile(ckpt_file("qat", dataset, mult)))


# ---------------------------------------------------------------------------
# Baseline evaluations.  One branch per writer under
# code/experiments/vision/ilharco_timm_supervised/000_baselines/.
#
# Note the asymmetry: `pretrained` and `pretrained_ptq` carry no `optim=`
# fragment, because a never-finetuned model has no training hyperparameters to
# name.  Their multiplier therefore goes directly after the dataset.  This is
# rule 3 of the migration mapping, and getting it wrong here would make every
# pretrained guard miss.
# ---------------------------------------------------------------------------
def baseline_dir(variant, dataset, mult):
    head = [eval_base(), "vision", "ilharco_timm_supervised", "000_baselines",
            "vision", variant, MODEL_SAN, dataset]
    if variant in ("pretrained", "pretrained_ptq"):
        parts = head + [mult_path_frag(mult)]
    else:
        parts = head + [OPTIM_FRAG, mult_path_frag(mult)]

    if variant in ("qat", "qat_ptq"):
        parts.append(QAT_FRAG)
    if variant in ("fp_ptq", "qat_ptq", "pretrained_ptq"):
        parts.append(PTQ_FRAG)
    parts.append(f"seed={SEED}")
    return os.path.join(*parts)


def baseline_json(variant, dataset, mult):
    return os.path.join(baseline_dir(variant, dataset, mult), "eval_results.json")


def baseline_done(variant, dataset, mult):
    return os.path.isfile(baseline_json(variant, dataset, mult))


# ---------------------------------------------------------------------------
# Transfer cells
# ---------------------------------------------------------------------------
def transfer_dir(donor, dmult, receiver, rmult):
    return os.path.join(
        eval_base(), "vision", "ilharco_timm_supervised", "001_qat_transfer",
        "vision", "qv_transfer", MODEL_SAN,
        role_path_frag("src", donor, SEED, dmult),
        role_path_frag("tgt", receiver, SEED, rmult),
        OPTIM_FRAG, QAT_FRAG, PTQ_FRAG, QV_FRAG, f"split={SPLIT}",
    )


def transfer_json(donor, dmult, receiver, rmult):
    return os.path.join(transfer_dir(donor, dmult, receiver, rmult), "eval_results.json")


def transfer_done(donor, dmult, receiver, rmult):
    return os.path.isfile(transfer_json(donor, dmult, receiver, rmult))


# ---------------------------------------------------------------------------
# The work.  Enumerated in one place so the queue generator, the workers and the
# status dashboard cannot disagree about what "all of it" means.
# ---------------------------------------------------------------------------
def finetune_items():
    """(kind, dataset, mult) for every checkpoint the axis needs and lacks.

    The 21 ordinary datasets already have SHORT (that is every run this repo has
    ever done) and ImageNet already has LONG, so only the other corner of each
    is new: 21 x 2 at mult=4, plus ImageNet x 2 at mult=0.25.
    """
    out = []
    for ds in ORDINARY:
        for kind in ("fp", "qat"):
            out.append((kind, ds, level_mult(ds, LONG)))
    for kind in ("fp", "qat"):
        out.append((kind, IMAGENET, level_mult(IMAGENET, SHORT)))
    return out


def baseline_items():
    """Baselines at the budgets that are new. The pre-existing tree covers the rest."""
    out = []
    for ds in ORDINARY:
        for v in BASELINE_VARIANTS:
            out.append((v, ds, level_mult(ds, LONG)))
    for v in BASELINE_VARIANTS:
        out.append((v, IMAGENET, level_mult(IMAGENET, SHORT)))
    return out


def transfer_batches():
    """One batch per (grid, receiver, donor-multiplier).

    Batching by receiver is not cosmetic: `qv_transfer.py` takes a CSV of donors
    for a single target and loads that target's FP checkpoint once, so a whole
    receiver column costs one process start instead of 22.

    The split by donor multiplier is forced.  Within a single grid the donor
    multiplier is not constant -- at the SHORT level the 21 ordinary donors sit
    at mult=1 while ImageNet sits at mult=0.25 -- and `v_qv.sh` takes one
    `source.epoch_mult` for the whole CSV.  So each (grid, receiver) yields two
    batches, one per donor group, rather than one mixed batch that could not be
    expressed.
    """
    out = []
    for dl, rl in GRIDS:
        prio = GRID_PRIORITY[(dl, rl)]
        for receiver in DATASETS:
            rmult = level_mult(receiver, rl)
            for group in (ORDINARY, (IMAGENET,)):
                dmult = level_mult(group[0], dl)
                out.append({
                    "priority": prio,
                    "grid": GRID_NAMES[(dl, rl)],
                    "receiver": receiver,
                    "rmult": rmult,
                    "dmult": dmult,
                    "donors": list(group),
                })
    return out


def transfer_cells():
    """Every (donor, dmult, receiver, rmult) the four grids cover -- 4 x 484."""
    out = []
    for b in transfer_batches():
        for d in b["donors"]:
            out.append((d, b["dmult"], b["receiver"], b["rmult"], b["grid"]))
    return out
