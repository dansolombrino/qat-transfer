"""Generate the per-rig work queues for the budget-axis wave.

Placement is not round-robin, because the three rigs are not interchangeable and
the work items are not equal-cost. Both cost models below are measured, not
guessed:

* **Finetunes** use the per-dataset 1x wall clock from the PV finetune ledgers
  (`logs/dispatch/pv_finetune_wave*/done.txt`) -- PV at bits=3 has the same step
  budget and the same forward/backward-plus-fake-quant shape as QAT, so it is the
  closest available proxy. A `mult=4` run costs ~4x its 1x time, because
  `loop_epochs` scales with the multiplier and so does the per-epoch eval.
* **Transfer cells** use the per-receiver median seconds from the 735
  `.status.json` markers of the `009_qat_transfer_awq` waves, which have the same
  four-evaluation shape as a `qv_transfer.py` cell.

Two placement constraints are hard rather than cost-driven:

1. **ImageNet goes to behemoth.** An ImageNet epoch is ~55 min there against ~24 h
   on rig-4090 -- a dataloader-core artifact (288 cores vs 32), not a GPU ratio.
   The same applies to ImageNet as a *receiver*: its test split dominates the cell
   cost (308 s on rig-4090, 554 s on rig-3090-ti).
2. **rig-3090-ti only gets small-train-set finetunes.** It has 16 cores and
   `TORCH_NUM_WORKERS=4`, so a large-train-set dataset there is dataloader-bound
   in exactly the way rule 8 of the dispatch skill warns about.

Static partition, deliberately. A cross-rig pull queue is impossible without a
shared filesystem, and one ssh per item would hit behemoth's `MaxSessions 10`.
The imbalance a static split leaves is corrected by the orchestrator moving lines
between queue files, which is cheap because the queues are plain text.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bx_common as bx

# ---------------------------------------------------------------------------
# Measured constants
# ---------------------------------------------------------------------------
# Per-dataset 1x QAT wall clock on behemoth, seconds (minimum observed across
# repeats, so it excludes contention). STL10 was never run on behemoth; its value
# is scaled from the rig-4090 observation by the ratio the co-observed datasets
# show.
BASE_1X_BEH_S = {
    "Flowers102": 541, "EMNIST": 690, "EuroSAT": 725, "PCAM": 740,
    "SVHN": 791, "FER2013": 798, "CIFAR10": 803, "CIFAR100": 804,
    "FashionMNIST": 804, "GTSRB": 823, "MNIST": 828, "KMNIST": 831,
    "Food101": 847, "RESISC45": 861, "SUN397": 902, "RenderedSST2": 958,
    "STL10": 970, "Cars": 1052, "TinyImageNet": 1077, "DTD": 1314,
    "OxfordIIITPet": 1388, "ImageNet": 3473,
}

# Per-receiver median seconds for one four-evaluation transfer cell on rig-4090.
CELL_4090_S = {
    "ImageNet": 308, "SUN397": 212, "PCAM": 212, "SVHN": 179, "Food101": 176,
    "EMNIST": 150, "GTSRB": 109, "KMNIST": 96, "CIFAR100": 96, "TinyImageNet": 96,
    "MNIST": 95, "CIFAR10": 95, "FashionMNIST": 95, "Cars": 92, "STL10": 85,
    "Flowers102": 82, "FER2013": 81, "RESISC45": 76, "OxfordIIITPet": 64,
    "EuroSAT": 57, "DTD": 54, "RenderedSST2": 54,
}

# Train-split sizes, from evaluations/998_rebuttal/002_cost_amortization/dataset_sizes.json.
TRAIN_SIZE = {
    "Flowers102": 918, "OxfordIIITPet": 3312, "DTD": 3384, "STL10": 4500,
    "RenderedSST2": 6228, "Cars": 7330, "SUN397": 17865, "RESISC45": 17010,
    "EuroSAT": 19440, "FER2013": 25839, "GTSRB": 23976, "CIFAR10": 45000,
    "CIFAR100": 45000, "MNIST": 55000, "FashionMNIST": 55000, "KMNIST": 55000,
    "SVHN": 68257, "Food101": 70750, "TinyImageNet": 95000, "EMNIST": 119800,
    "PCAM": 257144, "ImageNet": 1276167,
}

# Wall-clock cost multiplier per rig, relative to behemoth for finetuning and to
# rig-4090 for evaluation. behemoth's evaluation slot count is 2 (96 GB, 288
# cores), which raises its effective evaluation throughput without doubling it.
RIGS = {
    "behemoth": {
        "root": "/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer",
        "gpu": 0, "nw": 96, "ft_factor": 1.00, "ev_speed": 1.96 * 1.35, "ev_slots": 2,
    },
    "rig-4090": {
        "root": "/mnt/KS_2TB/PARA/Projects/quantization/qat-transfer",
        "gpu": 0, "nw": 16, "ft_factor": 1.30, "ev_speed": 1.00, "ev_slots": 1,
    },
    "rig-3090-ti": {
        "root": "/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer",
        "gpu": 0, "nw": 4, "ft_factor": 2.17, "ev_speed": 0.57, "ev_slots": 1,
    },
}

# rig-3090-ti's 4 dataloader workers make anything bigger than this a bad fit.
SMALL_TRAIN_MAX = 10_000
FP_COST_RATIO = 0.8   # an FP step lacks the fake-quant work of a QAT step


def finetune_cost(kind, dataset, mult):
    """Behemoth-seconds for one finetune run at the given budget."""
    base = BASE_1X_BEH_S[dataset] * float(mult)
    return base * (FP_COST_RATIO if kind == "fp" else 1.0)


def place_finetunes():
    """Longest-processing-time-first, with the two hard constraints applied."""
    load = {r: 0.0 for r in RIGS}
    assigned = {r: [] for r in RIGS}

    # ImageNet first and pinned: it is the only dataloader-bound dataset here, and
    # it unlocks the whole all-SHORT grid, so it must not sit behind anything.
    for kind in ("fp", "qat"):
        mult = bx.level_mult(bx.IMAGENET, bx.SHORT)
        assigned["behemoth"].append((kind, bx.IMAGENET, mult))
        load["behemoth"] += finetune_cost(kind, bx.IMAGENET, mult) * RIGS["behemoth"]["ft_factor"]

    # A dataset is placed as a unit -- its FP and its QAT on the same rig -- so it
    # becomes usable as a donor without waiting on a cross-rig replication.
    mult = bx.level_mult("Cars", bx.LONG)
    cost = {d: finetune_cost("fp", d, mult) + finetune_cost("qat", d, mult)
            for d in bx.ORDINARY}
    eligible = {
        d: [r for r in RIGS
            if r != "rig-3090-ti" or TRAIN_SIZE[d] <= SMALL_TRAIN_MAX]
        for d in bx.ORDINARY
    }

    # Longest-processing-time-first for a starting point.
    where = {}
    for ds in sorted(bx.ORDINARY, key=lambda d: -cost[d]):
        best = min(eligible[ds], key=lambda r: load[r] + cost[ds] * RIGS[r]["ft_factor"])
        where[ds] = best
        load[best] += cost[ds] * RIGS[best]["ft_factor"]

    # Then reduce the makespan by local search. Plain greedy systematically starves
    # rig-3090-ti: its 2.17x cost factor makes every individual placement there
    # look bad, so it ends up idle for hours while behemoth is the critical path --
    # even though only rig-3090-ti-eligible datasets *can* relieve it. Moves and
    # swaps recover that; on 21 items it converges in milliseconds, and it is worth
    # roughly three hours of wall clock on this wave.
    def makespan():
        return max(load.values())

    improved = True
    while improved:
        improved = False
        for ds in bx.ORDINARY:
            src = where[ds]
            for dst in eligible[ds]:
                if dst == src:
                    continue
                before = makespan()
                load[src] -= cost[ds] * RIGS[src]["ft_factor"]
                load[dst] += cost[ds] * RIGS[dst]["ft_factor"]
                if makespan() < before - 1e-9:
                    where[ds] = dst
                    improved = True
                    break
                load[dst] -= cost[ds] * RIGS[dst]["ft_factor"]
                load[src] += cost[ds] * RIGS[src]["ft_factor"]
        for a in bx.ORDINARY:
            for b in bx.ORDINARY:
                ra, rb = where[a], where[b]
                if ra == rb or rb not in eligible[a] or ra not in eligible[b]:
                    continue
                before = makespan()
                load[ra] += (cost[b] - cost[a]) * RIGS[ra]["ft_factor"]
                load[rb] += (cost[a] - cost[b]) * RIGS[rb]["ft_factor"]
                if makespan() < before - 1e-9:
                    where[a], where[b] = rb, ra
                    improved = True
                    break
                load[ra] -= (cost[b] - cost[a]) * RIGS[ra]["ft_factor"]
                load[rb] -= (cost[a] - cost[b]) * RIGS[rb]["ft_factor"]
            if improved:
                break

    # Cheapest-first within a rig. The queue is FIFO, so this makes whole datasets
    # become usable as donors as early as possible, which is what lets the
    # heatmaps fill progressively instead of all at the end.
    for ds in sorted(bx.ORDINARY, key=lambda d: cost[d]):
        assigned[where[ds]].append(("fp", ds, mult))
        assigned[where[ds]].append(("qat", ds, mult))

    return assigned, load


def load_done(path):
    """Cells already on the hub, as a set of (donor, dmult, recv, rmult)."""
    if not path:
        return set()
    out = set()
    for line in Path(path).read_text().splitlines():
        p = line.strip().split(",")
        if len(p) == 4:
            out.add(tuple(p))
    return out


def place_evals(done=frozenset()):
    """Baselines and transfer batches, balanced on measured per-receiver cost.

    `done` is what the hub already holds. Pruning against it is not an
    optimisation -- without it the two rigs whose evaluation trees lack the
    pre-existing 484 cells recompute them, because a local existence check is the
    only thing a worker can do and it is correct-but-local.
    """
    load = {r: 0.0 for r in RIGS}
    assigned = {r: [] for r in RIGS}

    def add(rig, prio, line, cost):
        assigned[rig].append((prio, line))
        load[rig] += cost / RIGS[rig]["ev_speed"]

    # Baselines: six per (dataset, budget). Placed with the dataset that produced
    # the checkpoint, so a baseline never waits on replication.
    ft_assigned, _ = place_finetunes()
    owner = {}
    for rig, items in ft_assigned.items():
        for _kind, ds, mult in items:
            owner[(ds, mult)] = rig
    for variant, ds, mult in bx.baseline_items():
        rig = owner.get((ds, mult), "behemoth")
        add(rig, bx.BASELINE_PRIORITY,
            f"BL\t{variant}\t{ds}\t{mult}", CELL_4090_S[ds] * 0.4)

    # Transfer batches, pruned to the donors that are genuinely missing, then most
    # expensive first so the greedy step has room to work.
    pruned = []
    for b in bx.transfer_batches():
        donors = [d for d in b["donors"]
                  if (d, b["dmult"], b["receiver"], b["rmult"]) not in done]
        if donors:
            pruned.append(dict(b, donors=donors))
    batches = sorted(pruned,
                     key=lambda b: -CELL_4090_S[b["receiver"]] * len(b["donors"]))
    for b in batches:
        cost = CELL_4090_S[b["receiver"]] * len(b["donors"])
        if b["receiver"] == bx.IMAGENET:
            rig = "behemoth"          # its test split dominates the cell cost
        else:
            rig = min(RIGS, key=lambda r: load[r] + cost / RIGS[r]["ev_speed"])
        line = "QV\t{}\t{}\t{}\t{}\t{}".format(
            b["receiver"], b["rmult"], b["dmult"], b["grid"], ",".join(b["donors"]))
        add(rig, b["priority"], line, cost)

    return assigned, load, batches


def launch_local(rig, wave_id):
    """The launcher that runs *on* a rig, with every path already resolved.

    Generated rather than invoked with arguments over ssh because overrides and
    paths that cross two levels of shell quoting (local -> ssh -> remote) are a
    reliable source of parse failures. The caller runs `bash launch_local.sh` and
    passes nothing.

    `tmux has-session || tmux new-session` makes it idempotent: after a reboot or
    a killed lane, re-running it with the same wave id restarts only what is
    missing, and the workers skip completed items on their own because every guard
    is an artifact check.
    """
    cfg = RIGS[rig]
    root = cfg["root"]
    qdir = f"{root}/scripts/dispatch/budget_axis/waves/{wave_id}/{rig}"
    gpu, nw, slots = cfg["gpu"], cfg["nw"], cfg["ev_slots"]
    pre = f"bxaxis_{wave_id}_{rig}"
    return f"""#!/bin/bash
# Generated by bx_gen_queues.py for {rig}, wave {wave_id}. Do not edit.
set -u
ROOT={root}
QDIR={qdir}
PY=$ROOT/.venv/bin/python
GPU={gpu}
NW={nw}

# behemoth has eight cards and only GPU 0 is ours. This is exported into every
# worker, which refuses to start on any other card rather than falling back.
export AUTHORIZED_GPUS={gpu}

mkdir -p "$QDIR/logs"
cd "$ROOT" || exit 1

start() {{  # start <session-name> <command...>
  S=$1; shift
  if tmux has-session -t "$S" 2>/dev/null; then
    echo "SKIP $S (already running)"
  else
    tmux new-session -d -s "$S" "$*"
    echo "START $S"
  fi
}}

start {pre}_ft \\
  "cd $ROOT && AUTHORIZED_GPUS=$GPU bash scripts/dispatch/budget_axis/ft_worker.sh $QDIR $GPU $PY $ROOT $NW ft0"

for i in $(seq 0 {slots - 1}); do
  start {pre}_ev$i \\
    "cd $ROOT && AUTHORIZED_GPUS=$GPU bash scripts/dispatch/budget_axis/ev_worker.sh $QDIR $GPU $PY $ROOT $NW ev$i"
done

# Record pane pids at launch. `pkill -f ft_worker` would match the shell doing the
# killing, so anything that needs to stop these must stop them by pid.
tmux list-panes -a -F '#{{session_name}} #{{pane_pid}}' 2>/dev/null \\
  | grep "^bxaxis_{wave_id}_{rig}" > "$QDIR/pids.txt" || true
cat "$QDIR/pids.txt"
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wave-id", required=True)
    ap.add_argument("--out", default="scripts/dispatch/budget_axis/waves")
    ap.add_argument("--done-manifest",
                    help="cells the hub already holds (bx_check.py dump-done), "
                         "pruned out of every rig's queue")
    ap.add_argument("--ev-only", action="store_true",
                    help="rewrite only the evaluation queues. Required when "
                         "regenerating mid-wave: rewriting ft.q would re-add the "
                         "item a finetune worker is running right now, and an "
                         "in-flight run has no artifact yet, so the guard cannot "
                         "catch the duplicate.")
    args = ap.parse_args()

    done = load_done(args.done_manifest)
    ft, ft_load = place_finetunes()
    ev, ev_load, batches = place_evals(done)

    root = Path(args.out) / args.wave_id
    for rig in RIGS:
        d = root / rig
        d.mkdir(parents=True, exist_ok=True)
        if not args.ev_only:
            (d / "ft.q").write_text("".join(f"{k}\t{ds}\t{m}\n" for k, ds, m in ft[rig]))
        for prio in range(5):
            lines = [ln for p, ln in ev[rig] if p == prio]
            (d / f"ev.q.{prio}").write_text("".join(ln + "\n" for ln in lines))
            (d / f"ev.d.{prio}").write_text("")
        (d / "launch_local.sh").write_text(launch_local(rig, args.wave_id))
        (d / "launch_local.sh").chmod(0o755)

    print(f"wave {args.wave_id} -> {root}\n")
    hdr = f"{'rig':<14}{'ft runs':>9}{'ft wall':>10}{'ev items':>10}{'ev wall':>10}"
    print(hdr)
    print("-" * len(hdr))
    tot_ft = tot_ev = 0
    for rig in RIGS:
        nft, nev = len(ft[rig]), len(ev[rig])
        tot_ft += nft
        tot_ev += nev
        print(f"{rig:<14}{nft:>9}{ft_load[rig]/3600:>9.1f}h"
              f"{nev:>10}{ev_load[rig]/3600:>9.1f}h")
    print("-" * len(hdr))
    print(f"{'total':<14}{tot_ft:>9}{'':>10}{tot_ev:>10}")

    cells = sum(len(b["donors"]) for b in batches)
    print(f"\nfinetunes {tot_ft} (expect 44)   baselines {len(bx.baseline_items())} (expect 132)")
    print(f"transfer batches {len(batches)} covering {cells} cells to compute")
    if done:
        print(f"pruned {len(done)} cells already held by the hub "
              f"({cells + len(done)} total, expect 1936)")
    print("\nfinetune placement:")
    for rig in RIGS:
        ds = sorted({d for _k, d, _m in ft[rig]})
        print(f"  {rig:<14} {', '.join(ds)}")


if __name__ == "__main__":
    main()
