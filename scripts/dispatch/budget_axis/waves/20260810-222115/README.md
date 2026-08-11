# Budget-axis wave 20260810-222115

Launched 2026-08-10 22:27 CEST. vit_base_patch16_224.orig_in21k, seed 2038, 3-bit,
alpha=1.0, split=test.

## What is running

| rig | GPU | tmux sessions |
|---|---|---|
| behemoth | 0 only | bxaxis_20260810-222115_behemoth_{ft,ev0,ev1} |
| rig-4090 | 0 | bxaxis_20260810-222115_rig-4090_{ft,ev0} + _replicator + _status |
| rig-3090-ti | 0 | bxaxis_20260810-222115_rig-3090-ti_{ft,ev0} |

behemoth GPUs 1-7 are NOT ours. Every worker exports AUTHORIZED_GPUS=0 and exits 64
rather than falling back to another card.

## Live status

    cat logs/dispatch/budget_axis/20260810-222115/status.txt        # refreshed every 120s
    .venv/bin/python scripts/dispatch/budget_axis/bx_status.py --wave-id 20260810-222115 \
        --out /tmp/s.txt --history logs/dispatch/budget_axis/20260810-222115/history.jsonl

## Work

44 finetunes (21 ordinary x {fp,qat} at mult=4; ImageNet x {fp,qat} at mult=0.25),
132 baselines at the new budgets, 1452 new transfer cells across four grids
(G_SS 43, G_LL 483, G_SL 463, G_LS 463). 484 cells pre-existed.

## Resume after a reboot or a killed lane

Re-run the same launcher with the same wave id; it is idempotent
(tmux has-session guard, and every worker guard is an artifact check):

    bash scripts/dispatch/budget_axis/waves/20260810-222115/<rig>/launch_local.sh

## Stop

Kill by pid from <rig>/pids.txt. Never `pkill -f ft_worker` -- it matches the
calling shell.

## Incident log

**22:29-22:41 — remotes recomputed pre-existing G_SS cells.** The 484 already-done
cells live only on rig-4090; behemoth and rig-3090-ti correctly reported them
absent locally and began recomputing all 441 ordinary-by-ordinary G_SS cells
(~10 GPU-hours of identical numbers). Fixed by pruning: `bx_check.py dump-done` on
the hub, then `bx_gen_queues.py --done-manifest ... --ev-only`. Queues went from
176 batches / 1936 cells to 132 batches / **1452 cells** -- the genuinely new work.
Nothing was written before the stop; zero duplicates on either remote.

**Same window — `tmux kill-session` did not stop the workers.** The pane died and
`tmux ls` showed nothing, but two worker shells on behemoth survived reparenting
and kept pulling from the stale queue for another eight minutes, starting a fresh
batch 46 s after the "kill". Use `bx_stop.sh <wave> <rig> [ft|ev|all]`, which kills
the recorded pane pids and then reaps children by explicit pid. Cost ~11 min of
behemoth eval time.

**23:11 — rig-3090-ti went DOWN (host/network, not sshd).** Three consecutive ssh
failures plus 100% ICMP loss on 192.168.1.186 (`No route to host`). Lost: Cars FP
`mult=4` at ~35% and one in-flight G_SS cell.

Its remaining work was reassigned from the hub's pristine generated copy with
`bx_reassign.sh <wave> rig-3090-ti`: 6 finetunes (Cars+DTD to behemoth,
OxfordIIITPet to rig-4090), 18 baselines following their datasets, 21 transfer
batches split 2:1 by throughput. Queues verified: zero malformed lines.

**This exposed a design gap worth knowing about.** A worker pops an item *out* of
its queue before running it, so a machine that dies mid-item leaves no artifact
*and* no queue entry -- the item would be silently skipped if the rig were simply
restarted. Recovery must therefore reassign from the hub copy (as above), and the
final reconcile pass (which verifies by artifact count against the full
enumeration) is the backstop that catches any such hole. If this wave is ever
re-run, add an in-flight marker file per worker and fold it back on restart.

The replicator now probes reachability with a 5 s ssh before each cycle; without
it, rssh.sh's retry backoff added ~3.5 min per cycle for the dead rig and slowed
replication for the live ones.

**23:47 — eval workers were spending most of their time in Python startup.**
`bx_common` transitively imported torch (via `src.vision.data.common` and
`src.vision.utils`), so every guard call cost **3.0 s** and loaded 1,120 modules.
A worker rescans its whole queue each cycle, so with ~150 deferred items that was
minutes per cycle of pure startup; behemoth's GPU utilisation had fallen to 41%.

Fixed by making `bx_common` torch-free: the epochs table is now parsed from
`common.py` with `ast` (so it cannot drift from its source) and the timm
sanitiser -- three literal replacements -- is reimplemented locally. Equivalence
is *proved*, not assumed, by `bx_check.py verify-imports`, which imports the real
module and asserts both match; run it after any change to those two files.
Guard cost 3.03 s -> **0.067 s (45x)**. `src.duration` was already clean at 0.01 s.

Two further lessons from stopping the workers to deploy this:

* `bx_stop.sh`'s session pattern was anchored to end-of-*line*, but `pids.txt`
  lines are `<session> <pid>`, so it matched nothing -- and then killed the
  children of workers that were still alive, which simply restarted the work.
  Now matched against `$1` in awk, and the script refuses to kill children when
  no worker shell matched.
* A non-interactive bash defers SIGTERM while a foreground child runs, so the
  worker shells survived `kill`. They need SIGKILL (the script escalates).
