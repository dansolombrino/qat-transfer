# HANDOFF — budget-axis wave 20260810-222115

**Read this file first.** It is the entry point for anyone (human or a fresh Claude
session) taking over this wave. Written 2026-08-11 08:55.

Nothing needs to be relaunched to take over. The workers, the replicator and the
status dashboard are tmux sessions **on the rigs**, not in any chat session. Every
guard reads state from disk. A new session only has to start watching.

---

## 1. What this wave is

Closing out the `epoch_mult` (training-budget) axis for
`vit_base_patch16_224.orig_in21k`, seed 2038, 3-bit QAT/PTQ, `alpha=1.0`,
`split=test`.

The axis is two *step budgets*, not two multipliers, because ImageNet's `mult=1`
(9,971 steps) is already ~4x the median dataset's (2,030):

| level | ordinary 21 datasets | ImageNet |
|---|---|---|
| **S**hort (~2.5k steps) | `mult=1` (pre-existing) | `mult=0.25` (new) |
| **L**ong (~10k steps) | `mult=4` (new) | `mult=1` (pre-existing) |

Four donor x receiver grids, 484 cells each:

| grid | donor | receiver | = user's request |
|---|---|---|---|
| G_SS | S | S | **config 2** |
| G_LL | L | L | **config 3** |
| G_SL | S | L | added — isolates "longer *receiver*" |
| G_LS | L | S | added — isolates "longer *donor*" |

Config 1 ("all 1x except ImageNet donor at 0.25x") is a free slice of G_SS ∪ G_SL.
G_SL and G_LS were added because the three requested configs all hold donor and
receiver at the *same* budget and therefore cannot separate those two claims.

Totals: 44 finetunes, 132 baselines, 1,452 new cells (484 pre-existed → 1,936).

---

## 2. Live status, in one command

    cat logs/dispatch/budget_axis/20260810-222115/status.txt

Refreshed every 120 s by a tmux loop on rig-4090, independent of any chat.
Force a refresh:

    .venv/bin/python scripts/dispatch/budget_axis/bx_status.py \
      --wave-id 20260810-222115 \
      --out logs/dispatch/budget_axis/20260810-222115/status.txt \
      --history logs/dispatch/budget_axis/20260810-222115/history.jsonl

Progress at handoff: **finetunes 14/44 · baselines 35/132 · cells 710/1936**
(G_SS 469, G_LS 116, G_SL 90, G_LL 35). Zero failures, zero parked items.

---

## 3. What is running where

| rig | GPU | sessions |
|---|---|---|
| behemoth | **0 only** | `bxaxis_20260810-222115_behemoth_{ft,ft1,ft2,ev0,ev1}` |
| rig-4090 | 0 | `..._rig-4090_{ft,ev0}` + `..._replicator` + `..._status` |
| rig-3090-ti | 0 | `..._rig-3090-ti_{ft,ev0}` |

**behemoth GPUs 1-7 are not ours.** Every worker exports `AUTHORIZED_GPUS=0` and
exits 64 rather than falling back to another card. Do not change this.

Repo roots: behemoth `/home/dansolombrino/data/PARA/.../qat-transfer`,
rig-4090 `/mnt/KS_2TB/PARA/.../qat-transfer`,
rig-3090-ti `/mnt/KS_960GB/PARA/.../qat-transfer` (checkpoints on `/mnt/WD_4TB`).

---

## 4. Operating it

    # relaunch a lane (idempotent; SAME wave id — artifact guards skip what's done)
    bash scripts/dispatch/budget_axis/waves/20260810-222115/<rig>/launch_local.sh

    # stop workers correctly (by pid, escalates to SIGKILL)
    bash scripts/dispatch/budget_axis/bx_stop.sh 20260810-222115 <rig> [ft|ev|all]

    # move a downed rig's work to the others
    bash scripts/dispatch/budget_axis/bx_reassign.sh 20260810-222115 <down-rig>

    # queries (all guards go through this — never spell a path by hand)
    .venv/bin/python scripts/dispatch/budget_axis/bx_check.py counts
    .venv/bin/python scripts/dispatch/budget_axis/bx_check.py verify-imports

Recovery rules (from the approved plan): rig down only after 2-3 consecutive ssh
failures; a frozen "running" after a reboot is **interrupted, not failed**; requeue a
genuine failure once then park it; **never `pkill -f` a worker** (it matches the
calling shell) — kill by pid from `pids.txt`; one 0% `nvidia-smi` sample is not a stall.

---

## 5. Things that will bite you (each cost a real incident here)

1. **`qv_transfer.py` exits 0 when a checkpoint is missing.** Judge completion by
   artifact only. This is why every guard is a filesystem check.
2. **Pop-then-crash loses an item silently.** A worker pops an item *out* of its queue
   before running it, so a machine dying mid-item leaves no artifact *and* no queue
   entry. Cost us `qat FER2013` (rig-4090 reboot) and `fp Cars` (rig-3090-ti outage).
   Recover by reconciling against the full enumeration, not by restarting the rig.
   **If this wave is ever re-run, add an in-flight marker file per worker.**
3. **`tmux kill-session` does not stop a worker.** The pane dies, `tmux ls` shows
   nothing, and the worker shell survives reparenting and keeps pulling. Use
   `bx_stop.sh`.
4. **Non-interactive bash defers SIGTERM** while a foreground child runs — shells need
   SIGKILL.
5. **Never `uv run`/`uv sync` on behemoth** (sm_120 needs its pinned cu128/cu129 torch).
   Use `.venv/bin/python`.
6. **Do not sum per-rig counts.** The replicator copies checkpoints to all three rigs,
   so a sum triple-counts. `bx_status.py` unions tagged item ids instead.
7. **rig-3090-ti's queue under `stale/`** lists items already reassigned elsewhere.
   Never restart it from those.

---

## 6. Verified so far — budgets are real, not assumed

Every `mult=4` checkpoint is checked against `run_meta.json` as it lands:
EMNIST 7488, SVHN 8544, Flowers102 4704, EuroSAT 7296, FER2013 8080, plus ImageNet
2493 at `mult=0.25`. All `= 4x` (or `0.25x`) the 1x step count, `warmup_length <
max_steps` everywhere. **No budget has silently stayed at 1x.** Keep doing this check
for each new checkpoint; the expected step counts are in section 8.

---

## 7. Findings so far (for the write-up)

**QAT is far more budget-sensitive than FP.** Consistent across two datasets and both
directions of the axis:

| | FP | QAT |
|---|---|---|
| ImageNet `mult=1` -> `0.25` | 74.37% -> 61.61% (−12.8 pp) | 53.21% -> 16.82% (**−36.4 pp**) |
| EMNIST `mult=1` -> `4` | 81.15% -> 91.38% (+10.2) | 10.31% -> **89.40%** (+79.1) |

EMNIST at the standard budget has QAT at 10.31% against FP 81.15% — its quantization
vector points at a nearly-broken model, so it is a hopeless donor *because it is
undertrained*, not intrinsically. That is the axis doing its job, and it is why G_LS
(long donor -> short receiver) was worth adding.

**Caveat that must go in the paper:** `wl=500` is deliberately not rescaled by the
multiplier (`clamped_warmup` only clamps). At `mult=1` warmup is 5% of ImageNet's
schedule; at `mult=0.25` it is **20%**. So short-budget runs differ in schedule
*shape*, not only length, and some of that −36.4 pp is warmup fraction rather than
duration. It applies uniformly across the axis but must be stated.

---

## 8. What remains

    finetunes  30 of 44   (behemoth 10 queued + 3 in-flight, rig-4090 11, rig-3090-ti 5)
    baselines  97 of 132
    cells    1226 of 1936

Expected `mult=4` step counts, for verification:
Cars 8120, DTD 8208, EMNIST 7488, CIFAR10 8448, CIFAR100 8448, GTSRB 8272, MNIST 8600,
FashionMNIST 8600, KMNIST 8600, SVHN 8544, Food101 8848, TinyImageNet 11888, PCAM 8036,
STL10 8640, Flowers102 4704, RenderedSST2 7644, EuroSAT 7296, RESISC45 7980,
SUN397 7840, FER2013 8080, OxfordIIITPet 8528; ImageNet `mult=0.25` = 2493.

**Load balance:** measured throughput is behemoth 6,128 steps/h with 3 concurrent
finetunes vs rig-4090 **6,496 steps/h with one** — 3-way sharing of a saturated GPU
adds no aggregate throughput. Balance by *measured* rate, not by hardware tier. As of
08:55 the queues are balanced at ~14 h each.

### Then, still to do (tasks 9 and 10 of the plan)

1. **Reconcile** by artifact count per rig against the full enumeration; requeue gaps.
2. **Pull evaluations to rig-4090** — JSON only, `rsync -a --partial`, **no `--delete`**:
   - `behemoth:/home/dansolombrino/data/PARA/.../qat-transfer/evaluations/`
   - `rig-3090-ti:/mnt/KS_960GB/PARA/.../qat-transfer/evaluations/`
3. **Assert 1936/1936 cells + 132/132 baselines locally before plotting.**
4. **Plots**, one set per grid (`--source-epoch-mult` / `--target-epoch-mult` keep them
   from overwriting each other, since `smult=`/`tmult=` are mandatory path fragments):
   - `code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap.py` x4
   - `code/experiments/998_rebuttal/001_zero_shot_reframing/compute_win_loss_timm_supervised.py`
     -> `aggregate_win_loss.py` -> `.../win_loss_table.py` x4
   - **a new cross-grid figure** (recovery ratio as a function of donor budget x
     receiver budget) — this is the actual deliverable of G_SL/G_LS and no existing
     script draws it.
5. Update `EXPERIMENTS.md` run table + a `journal.md` entry.

---

## 9. Not yet committed

    M scripts/dispatch/runners/v_fp.sh      (fixed: guard used $ROOT/storage, wrong on rig-3090-ti)
    M scripts/dispatch/runners/v_qat.sh     (added idempotency guard + artifact verification)
    ?? scripts/dispatch/runners/v_baseline.sh
    ?? scripts/dispatch/budget_axis/        (bx_common, bx_check, bx_status, bx_gen_queues,
                                             ft_worker, ev_worker, replicator, bx_stop,
                                             bx_reassign, waves/)

Worth committing so the tooling is reviewable and durable. Not done — needs the user's
go-ahead.

---

## 10. Other docs

| file | what |
|---|---|
| `waves/20260810-222115/README.md` | wave layout, how to resume a lane, incident log |
| `logs/dispatch/budget_axis/20260810-222115/OVERNIGHT.md` | chronological event log incl. every budget verification and the 08:39 crash recovery |
| `~/.claude/plans/ok-let-s-pls-then-wobbly-fountain.md` | the approved plan: full enumeration, placement rationale, verification section |
| `.claude/skills/multi-rig-dispatch/SKILL.md` | the project's hard-won dispatch rules |
