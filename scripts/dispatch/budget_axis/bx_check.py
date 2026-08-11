"""Artifact queries for the budget-axis workers.

The shell workers ask this for every "is it done?" and "is it runnable?"
decision instead of globbing a path they spelled themselves. That indirection is
the point: `bx_common` reproduces each writer's output path once, validated
against the existing tree, so a guard cannot drift from its writer. A guard that
drifts fails silently -- it just never matches -- which is the single most
expensive bug class in a wave this size.

Subcommands
-----------
    ckpt <ds> <mult>                     exit 0 if FP *and* QAT exist
    baseline <variant> <ds> <mult>       exit 0 if eval_results.json exists
    cell <donor> <dmult> <recv> <rmult>  exit 0 if the transfer cell exists
    runnable-donors <recv> <rmult> <dmult> <csv>
                                         print the donors that can run now:
                                         checkpoints present and cell not done
    counts                               JSON progress summary for this rig
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bx_common as bx


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd = argv[0]

    if cmd == "verify-imports":
        # bx_common reimplements two things to avoid a 3-second torch import on
        # every guard call. This proves the reimplementations agree with the
        # originals. Run it after any change to common.py or utils.py -- a silent
        # divergence here would make every path in the wave subtly wrong.
        from src.vision.data.common import DATASET_NAME_TO_EPOCHS as REAL_EPOCHS
        from src.vision.utils import sanitize_timm_model_name as real_sanitize

        ok = True
        if bx.DATASET_NAME_TO_EPOCHS != REAL_EPOCHS:
            print("MISMATCH epochs table", file=sys.stderr)
            print(f"  parsed: {bx.DATASET_NAME_TO_EPOCHS}", file=sys.stderr)
            print(f"  real  : {REAL_EPOCHS}", file=sys.stderr)
            ok = False
        else:
            print(f"OK epochs table ({len(REAL_EPOCHS)} datasets, identical)")

        names = [bx.MODEL, "vit_base_patch16_clip_224.openai", "ViT-B-16/laion2b",
                 "deit3_base_patch16_224.fb_in1k", "a-b.c/d"]
        for n in names:
            if bx.sanitize_timm_model_name(n) != real_sanitize(n):
                print(f"MISMATCH sanitize({n!r}): "
                      f"{bx.sanitize_timm_model_name(n)!r} != {real_sanitize(n)!r}",
                      file=sys.stderr)
                ok = False
        if ok:
            print(f"OK sanitizer (identical on {len(names)} names)")
            print(f"OK MODEL_SAN = {bx.MODEL_SAN}")
        return 0 if ok else 1

    if cmd == "ckpt":
        ds, mult = argv[1], argv[2]
        return 0 if bx.ckpt_ready(ds, mult) else 1

    if cmd == "ckpt-one":
        # One half, existence only. Used by the finetune worker to judge the run
        # it has just finished, where the settle wait would be a false negative.
        kind, ds, mult = argv[1], argv[2], argv[3]
        return 0 if os.path.isfile(bx.ckpt_file(kind, ds, mult)) else 1

    if cmd == "baseline":
        variant, ds, mult = argv[1], argv[2], argv[3]
        return 0 if bx.baseline_done(variant, ds, mult) else 1

    if cmd == "cell":
        donor, dmult, recv, rmult = argv[1], argv[2], argv[3], argv[4]
        return 0 if bx.transfer_done(donor, dmult, recv, rmult) else 1

    if cmd == "runnable-donors":
        recv, rmult, dmult, csv = argv[1], argv[2], argv[3], argv[4]
        # The receiver's own FP+QAT gate the whole batch: qv_transfer.py patches
        # the receiver's FP checkpoint and reports against its QAT one, so
        # without both there is nothing to run for any donor.
        if not bx.ckpt_ready(recv, rmult):
            return 0
        donors = [d for d in csv.split(",") if d]
        out = [d for d in donors
               if bx.ckpt_ready(d, dmult)
               and not bx.transfer_done(d, dmult, recv, rmult)]
        if out:
            print(",".join(out))
        return 0

    if cmd == "pending-donors":
        # Donors whose cell is not done, regardless of readiness -- what remains
        # of a batch after a run, so the worker knows whether to requeue it.
        recv, rmult, dmult, csv = argv[1], argv[2], argv[3], argv[4]
        donors = [d for d in csv.split(",") if d]
        out = [d for d in donors if not bx.transfer_done(d, dmult, recv, rmult)]
        if out:
            print(",".join(out))
        return 0

    if cmd == "dump-all-done":
        # Everything this rig holds, tagged by kind, for the dashboard to union
        # across rigs.
        #
        # Summing per-rig counts is wrong, and wrong in a way that matters: the
        # replicator copies every checkpoint to all three rigs, so one finished
        # finetune is counted three times. The first one made the global read
        # 3/44 when a single checkpoint existed. Since the global count is also
        # the wave's completion test, over-counting would declare the run
        # finished early. A union of tagged item ids cannot do that, and it stays
        # correct even if evaluations are replicated later.
        for kind, ds, mult in bx.finetune_items():
            if os.path.isfile(bx.ckpt_file(kind, ds, mult)):
                print(f"FT\t{kind}\t{ds}\t{mult}")
        for variant, ds, mult in bx.baseline_items():
            if bx.baseline_done(variant, ds, mult):
                print(f"BL\t{variant}\t{ds}\t{mult}")
        for donor, dmult, recv, rmult, grid in bx.transfer_cells():
            if bx.transfer_done(donor, dmult, recv, rmult):
                print(f"CE\t{grid}\t{donor}\t{dmult}\t{recv}\t{rmult}")
        return 0

    if cmd == "dump-done":
        # Every transfer cell present on *this* rig, as donor,dmult,recv,rmult.
        #
        # This exists because "is this cell done?" is a global question that each
        # rig can only answer locally. The 484 pre-existing cells live solely on
        # rig-4090, so behemoth and rig-3090-ti correctly report them absent and
        # would recompute all 441 ordinary-by-ordinary G_SS cells -- identical
        # numbers, about ten GPU-hours, for nothing. The hub dumps what it has and
        # the queue generator prunes it out of the remotes' queues.
        for donor, dmult, recv, rmult, _grid in bx.transfer_cells():
            if bx.transfer_done(donor, dmult, recv, rmult):
                print(f"{donor},{dmult},{recv},{rmult}")
        return 0

    if cmd == "replicate-list":
        # Paths of every settled new-budget checkpoint, relative to
        # CHECKPOINT_BASE_PATH, for `rsync --files-from`. Only the budgets this
        # wave produces: the mult=1 tree is already identical on all three rigs,
        # so listing it would make every pass re-stat 176 large files for nothing.
        base = bx.ckpt_base().rstrip("/") + "/"
        for kind, ds, mult in bx.finetune_items():
            for f in (bx.ckpt_file(kind, ds, mult), bx.head_file(kind, ds, mult)):
                if bx._settled(f) and f.startswith(base):
                    print(f[len(base):])
        return 0

    if cmd == "remaining":
        # A single integer: baselines plus transfer cells still absent. The
        # evaluation workers use it to tell "still waiting on a checkpoint" apart
        # from "the wave is finished", which is the difference between sleeping
        # and exiting.
        n = sum(1 for v, d, m in bx.baseline_items() if not bx.baseline_done(v, d, m))
        n += sum(1 for c in bx.transfer_cells()
                 if not bx.transfer_done(c[0], c[1], c[2], c[3]))
        print(n)
        return 0

    if cmd == "counts":
        ft = bx.finetune_items()
        bl = bx.baseline_items()
        cells = bx.transfer_cells()
        by_grid = {}
        for donor, dmult, recv, rmult, grid in cells:
            g = by_grid.setdefault(grid, {"total": 0, "done": 0})
            g["total"] += 1
            if bx.transfer_done(donor, dmult, recv, rmult):
                g["done"] += 1
        out = {
            "host": os.uname().nodename,
            "finetunes": {
                "total": len(ft),
                "done": sum(os.path.isfile(bx.ckpt_file(k, d, m)) for k, d, m in ft),
            },
            "baselines": {
                "total": len(bl),
                "done": sum(bx.baseline_done(v, d, m) for v, d, m in bl),
            },
            "grids": by_grid,
            "cells": {
                "total": len(cells),
                "done": sum(v["done"] for v in by_grid.values()),
            },
        }
        print(json.dumps(out))
        return 0

    print(f"unknown subcommand {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
