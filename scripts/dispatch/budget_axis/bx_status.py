"""Per-rig status and ETA for the budget-axis wave. Runs on rig-4090.

Design constraints that shaped this:

* **One ssh round trip per rig.** behemoth runs the stock `MaxSessions 10` with no
  passwordless sudo, and the wave already holds sessions for its workers. A
  dashboard that opened a connection per question would be the thing that pushes
  it over the limit, so every question a rig is asked is bundled into a single
  remote command whose output is parsed here.
* **Artifact counts are the progress signal**, not exit statuses and not log
  tails. A dropped ssh returns non-zero after the remote work finished, and
  `qv_transfer.py` exits 0 when a checkpoint is missing, so both of the obvious
  signals lie, in opposite directions. `bx_check.py counts` stats the filesystem.
* **ETA comes from observed throughput on this wave**, not from the a-priori cost
  model. The model placed the work; it should not also grade it, or a systematic
  error in it would be invisible. Rates are measured over a trailing window of the
  history file this script appends to.
* **A rig is only declared down after repeated failures.** One refused session or
  one 0% `nvidia-smi` sample is normal -- samples land between epochs constantly.
  The counter lives in the history file so a restart does not reset it.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]

RIGS = {
    "behemoth": {
        "root": "/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer",
    },
    "rig-4090": {
        "root": "/mnt/KS_2TB/PARA/Projects/quantization/qat-transfer",
        "local": True,
    },
    "rig-3090-ti": {
        "root": "/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer",
    },
}

# Everything one rig is asked, in one shell. Sections are delimited so a partial
# answer (a rig that answers some questions then dies) is detectable rather than
# silently misparsed.
PROBE = r"""
cd {root} 2>/dev/null || {{ echo "@@ERR no root"; exit 9; }}
echo "@@COUNTS"
.venv/bin/python scripts/dispatch/budget_axis/bx_check.py counts 2>/dev/null || echo '{{}}'
echo "@@ITEMS"
.venv/bin/python scripts/dispatch/budget_axis/bx_check.py dump-all-done 2>/dev/null
echo "@@GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -20
echo "@@TMUX"
tmux ls 2>/dev/null | grep bxaxis || echo none
echo "@@QUEUES"
for f in {qdir}/ft.q {qdir}/ev.q.0 {qdir}/ev.q.1 {qdir}/ev.q.2 {qdir}/ev.q.3 {qdir}/ev.q.4; do
  printf '%s %s\n' "$(basename $f)" "$(grep -c . $f 2>/dev/null || echo 0)"
done
echo "@@DEFERRED"
for f in {qdir}/ev.d.0 {qdir}/ev.d.1 {qdir}/ev.d.2 {qdir}/ev.d.3 {qdir}/ev.d.4; do
  printf '%s %s\n' "$(basename $f)" "$(grep -c . $f 2>/dev/null || echo 0)"
done
echo "@@RUNNING"
grep -h '^=== ' {qdir}/logs/*.log 2>/dev/null | tail -4
echo "@@FAILS"
grep -hE 'FT_FAIL|BL_FAIL|FT_PARKED' {qdir}/done.txt 2>/dev/null | tail -6
echo "@@LEDGER"
tail -3 {qdir}/done.txt 2>/dev/null
echo "@@BOOT"
uptime -s
echo "@@END"
"""


def probe(rig, qdir, timeout=240):
    cfg = RIGS[rig]
    cmd = PROBE.format(root=cfg["root"], qdir=qdir)
    if cfg.get("local"):
        argv = ["bash", "-c", cmd]
    else:
        argv = ["bash", str(_ROOT / "scripts/dispatch/rssh.sh"), rig, cmd]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "why": "probe timeout"}
    if "@@END" not in out.stdout:
        return {"ok": False, "why": (out.stderr or out.stdout or "no answer").strip()[:200]}

    sec, cur = {}, None
    for line in out.stdout.splitlines():
        if line.startswith("@@"):
            cur = line[2:]
            sec[cur] = []
        elif cur:
            sec[cur].append(line)

    try:
        counts = json.loads("\n".join(sec.get("COUNTS", ["{}"])).strip() or "{}")
    except json.JSONDecodeError:
        counts = {}

    def ints(name):
        out = {}
        for line in sec.get(name, []):
            p = line.split()
            if len(p) == 2 and p[1].isdigit():
                out[p[0]] = int(p[1])
        return out

    return {
        "ok": True,
        "counts": counts,
        "items": [l for l in sec.get("ITEMS", []) if l.strip()],
        "gpu": [l for l in sec.get("GPU", []) if l.strip()],
        "tmux": [l for l in sec.get("TMUX", []) if l.strip() and l.strip() != "none"],
        "queues": ints("QUEUES"),
        "deferred": ints("DEFERRED"),
        "running": [l for l in sec.get("RUNNING", []) if l.strip()],
        "fails": [l for l in sec.get("FAILS", []) if l.strip()],
        "ledger": [l for l in sec.get("LEDGER", []) if l.strip()],
        "boot": "\n".join(sec.get("BOOT", [])).strip(),
    }


def fmt_eta(seconds):
    if seconds is None:
        return "--"
    if seconds <= 0:
        return "done"
    m = int(seconds // 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d:
        return f"{d}d{h:02d}h"
    return f"{h}h{m:02d}m" if h else f"{m}m"


def rate_and_eta(history, rig, key, done, total, window_s=5400):
    """Items/second over the trailing window, and the ETA it implies.

    A trailing window rather than since-launch: the mix of work changes over the
    run (finetunes first, then pure evaluation), so a since-launch average would
    lag reality by hours exactly when the numbers matter most.
    """
    now = time.time()
    pts = [(h["t"], h.get("rigs", {}).get(rig, {}).get(key)) for h in history]
    pts = [(t, v) for t, v in pts if v is not None and now - t <= window_s]
    if len(pts) < 2:
        return None, None
    (t0, v0), (t1, v1) = pts[0], pts[-1]
    if t1 <= t0 or v1 < v0:
        return None, None
    rate = (v1 - v0) / (t1 - t0)
    if rate <= 0:
        return 0.0, None
    return rate, (total - done) / rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--history", required=True)
    args = ap.parse_args()

    history = []
    hp = Path(args.history)
    if hp.is_file():
        for line in hp.read_text().splitlines():
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    snap = {"t": time.time(), "rigs": {}}
    results = {}
    for rig in RIGS:
        qdir = (f"{RIGS[rig]['root']}/scripts/dispatch/budget_axis/waves/"
                f"{args.wave_id}/{rig}")
        r = probe(rig, qdir)
        results[rig] = r
        if r["ok"]:
            c = r["counts"]
            snap["rigs"][rig] = {
                "ft": c.get("finetunes", {}).get("done"),
                "bl": c.get("baselines", {}).get("done"),
                "cells": c.get("cells", {}).get("done"),
            }

    # Union totals go into the snapshot so the ETA can be derived from global
    # progress. A per-rig cell ETA is structurally meaningless here: each rig is
    # assigned roughly a third of the work, so `(1936 - its_local_count) / its_rate`
    # divides a total it will never reach by a rate that only covers its share --
    # which is how the first one came out at 55 days. The honest per-rig number is
    # its own throughput and its own queue depth; the honest ETA is global.
    #
    # Global progress is the union of tagged item ids across rigs, never a sum of
    # per-rig counts: the replicator puts every checkpoint on all three rigs, so a
    # sum triple-counts each finished finetune.
    union = set()
    for rig in RIGS:
        if results[rig]["ok"]:
            union.update(results[rig].get("items", []))
    u_ft = sum(1 for i in union if i.startswith("FT\t"))
    u_bl = sum(1 for i in union if i.startswith("BL\t"))
    u_ce = sum(1 for i in union if i.startswith("CE\t"))
    u_grid = {}
    for i in union:
        if i.startswith("CE\t"):
            g = i.split("\t")[1]
            u_grid[g] = u_grid.get(g, 0) + 1

    prev_fail = (history[-1].get("fail", {}) if history else {})
    snap["fail"] = {}
    for rig in RIGS:
        snap["fail"][rig] = 0 if results[rig]["ok"] else int(prev_fail.get(rig, 0)) + 1

    snap["union"] = {"ft": u_ft, "bl": u_bl, "ce": u_ce}
    history.append(snap)
    with hp.open("a") as f:
        f.write(json.dumps(snap) + "\n")

    def global_rate_eta(key, done, total, window_s=5400):
        now = time.time()
        pts = [(h["t"], h.get("union", {}).get(key)) for h in history]
        pts = [(t, v) for t, v in pts if v is not None and now - t <= window_s]
        if len(pts) < 2 or pts[-1][1] <= pts[0][1]:
            return None, None
        rate = (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])
        if rate <= 0:
            return None, None
        return rate, (total - done) / rate

    L = []
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    L.append(f"BUDGET-AXIS WAVE {args.wave_id}    {ts}")
    L.append("=" * 78)

    tot = {"ft": [0, 0], "bl": [0, 0], "cells": [0, 0]}
    for rig in RIGS:
        r = results[rig]
        if not r["ok"]:
            n = snap["fail"][rig]
            state = "DOWN" if n >= 3 else f"unreachable (attempt {n}/3)"
            L.append(f"\n{rig:<14} {state}: {r['why']}")
            continue

        c = r["counts"]
        g = lambda k, f: c.get(k, {}).get(f, 0)  # noqa: E731
        ft_d, ft_t = g("finetunes", "done"), g("finetunes", "total")
        bl_d, bl_t = g("baselines", "done"), g("baselines", "total")
        ce_d, ce_t = g("cells", "done"), g("cells", "total")

        qd = r["queues"]
        ft_left = qd.get("ft.q", 0)
        ev_left = sum(v for k, v in qd.items() if k.startswith("ev.q"))
        dfr = sum(r["deferred"].values())

        # This rig's own throughput, in cells per hour. Not an ETA: a rig only
        # owns about a third of the grid, so it never reaches the 1936 total.
        rate, _ = rate_and_eta(history, rig, "cells", ce_d, ce_t)
        cph = f"{rate * 3600:.1f} cells/h" if rate else "-- cells/h"
        gpu = r["gpu"][0] if r["gpu"] else "?"
        sessions = len(r["tmux"])

        L.append(f"\n{rig:<14} tmux={sessions}  gpu[{gpu}]  boot={r['boot']}")
        L.append(f"  queues       ft={ft_left} left   ev={ev_left} left   deferred={dfr}")
        L.append(f"  finetunes    {ft_d}/{ft_t}       baselines  {bl_d}/{bl_t}")
        L.append(f"  cells here   {ce_d}    throughput {cph}")
        L.append("    " + "  ".join(
            f"{k} {v['done']}/{v['total']}" for k, v in sorted(c.get("grids", {}).items())))
        for line in r["running"][-2:]:
            L.append(f"  now  {line[4:][:96]}")
        for line in r["ledger"][-2:]:
            L.append(f"  last {line[:112]}")
        for line in r["fails"]:
            L.append(f"  FAIL {line[:112]}")

        tot["ft"][1], tot["bl"][1], tot["cells"][1] = ft_t, bl_t, ce_t

    ce_rate, ce_eta = global_rate_eta("ce", u_ce, tot["cells"][1])
    bl_rate, bl_eta = global_rate_eta("bl", u_bl, tot["bl"][1])
    ft_rate, ft_eta = global_rate_eta("ft", u_ft, tot["ft"][1])

    L.append("\n" + "=" * 78)
    L.append(f"GLOBAL  finetunes {u_ft}/{tot['ft'][1]}"
             f"   baselines {u_bl}/{tot['bl'][1]}"
             f"   cells {u_ce}/{tot['cells'][1]}")
    L.append("        " + "  ".join(
        f"{g} {u_grid.get(g, 0)}/484" for g in ("G_SS", "G_LL", "G_SL", "G_LS")))
    L.append(f"        rates: finetunes {ft_rate * 3600:.1f}/h" if ft_rate else
             "        rates: finetunes --/h")
    L.append(f"        cells {ce_rate * 3600:.0f}/h  eta {fmt_eta(ce_eta)}"
             f"   |   baselines {bl_rate * 3600:.0f}/h  eta {fmt_eta(bl_eta)}"
             if ce_rate and bl_rate else
             f"        cells eta {fmt_eta(ce_eta)}   baselines eta {fmt_eta(bl_eta)}")
    # The cell rate is still climbing while finetunes hold most of each GPU, so an
    # early ETA reads far too long. Say so rather than print a number that will be
    # wrong by a factor of ten.
    if u_ft < tot["ft"][1]:
        L.append("        (cell eta is not yet meaningful: finetunes still hold "
                 "most GPU time, cell rate rises as they finish)")
    if u_ce == tot["cells"][1] and u_bl == tot["bl"][1]:
        L.append("        WAVE COMPLETE -- all cells and baselines present")

    text = "\n".join(L) + "\n"
    Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
