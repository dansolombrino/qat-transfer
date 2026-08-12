"""Unattended health check for a budget-axis wave: silent unless something is wrong.

Why this exists rather than a human reading `status.txt` every couple of minutes:
the previous watch on wave 20260810-222115 was a hand-driven 2-minute loop, and it
died together with rig-4090's 04:42 hang. Nothing then watched for four hours, and
one item (`qat FER2013 mult=4`) was popped from its queue and lost silently -- a
worker removes an item *before* running it, so a machine dying mid-item leaves no
artifact and no queue entry. The gap was not a failure of attention; it was that
the watcher shared a fate with the thing it watched, and that its silence was
indistinguishable from a quiet wave.

So this check is built around three properties:

1.  **Silence is structural.** It prints nothing when all is well, which is what
    makes a 120-second cadence affordable for the days a wave runs. Speaking is the
    exception, and every reason to speak is enumerated below.
2.  **Silence is falsifiable.** A stale `status.txt` is itself an alarm, so the
    on-rig dashboard loop (or the rig hosting it) dying becomes visible instead of
    looking like calm. This is precisely the failure mode that produced the 4-hour
    gap.
3.  **A condition is reported once.** State is deduped against a JSON file, so an
    ongoing outage does not emit the same line 30 times an hour and train its
    reader to ignore it.

It re-implements nothing. `bx_status.py` already does the ssh probing, the
three-strike DOWN rule, tmux counts, boot times, failure-line surfacing and the
union-not-sum global counts; this reads its output. `bx_common` -- which is
deliberately torch-free, at 0.067 s per call against 3.03 s before that fix --
supplies every path.

Reasons to speak
----------------
    rig down / degraded   a DOWN or `unreachable (attempt N/3)` line, a tmux count
                          away from its expected value, or a boot timestamp that
                          moved (a reboot: interrupted, not failed)
    budget mismatch       a new checkpoint whose run_meta.json does not prove the
                          budget it claims -- see `verify_budget`
    failure / parked      a new FT_FAIL, BL_FAIL or FT_PARKED line
    completion           finetunes, baselines and cells all at their totals
    stale dashboard      status.txt older than --stale-after seconds

and, regardless of quiet, one compact status line every --report-every ticks.

Exit code is always 0: this runs inside a supervising loop that treats an exit as
the end of the watch, so a check that failed to check must not also stop it.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bx_common as bx

# ---------------------------------------------------------------------------
# Expected shape of the wave
# ---------------------------------------------------------------------------

# How many tmux sessions each rig should be running. behemoth has three finetune
# lanes and two eval lanes; rig-4090 additionally hosts the replicator and the
# status loop; rig-3090-ti runs one of each. A count below these means a lane died
# -- the wave keeps making progress, so nothing else would report it.
EXPECTED_TMUX = {"behemoth": 5, "rig-4090": 4, "rig-3090-ti": 2}

# ...but only while there is still finetune work to do. A finetune worker exits
# deliberately when its queue drains (`FT_WORKER_EXIT ... queue empty`), so as the
# wave winds down the launch-time count stops being the right expectation: on
# 2026-08-11 behemoth dropped 5 -> 4 -> 3 within ten minutes purely from lanes
# retiring, with `ft.q` empty and no work lost.
#
# Reporting those as DEGRADED is worse than not checking at all. An alarm that
# fires on normal shutdown teaches its reader to skip the line, and this is the
# same line that would report a genuinely dead lane. So a shortfall is only an
# alarm while the rig's `ft.q` still holds work; once it is empty, the finetune
# lanes are allowed to be gone and only the eval/support lanes are still expected.
FT_LANES = {"behemoth": 3, "rig-4090": 1, "rig-3090-ti": 1}

# The step count each new-budget finetune must realize, from the approved
# enumeration (HANDOFF section 8). This is deliberately a *hardcoded second
# opinion*: run_meta.json also permits a self-consistency check
# (max_steps == epoch_mult * base_epochs * num_batches), but that check cannot
# catch a wrong num_batches, since a wrong num_batches would agree with itself.
# Two independent derivations agreeing is the actual evidence that a budget is
# real; either alone is only evidence that a number was copied consistently.
EXPECTED_MAX_STEPS = {
    ("Cars", "4"): 8120,
    ("DTD", "4"): 8208,
    ("EMNIST", "4"): 7488,
    ("CIFAR10", "4"): 8448,
    ("CIFAR100", "4"): 8448,
    ("GTSRB", "4"): 8272,
    ("MNIST", "4"): 8600,
    ("FashionMNIST", "4"): 8600,
    ("KMNIST", "4"): 8600,
    ("SVHN", "4"): 8544,
    ("Food101", "4"): 8848,
    ("TinyImageNet", "4"): 11888,
    ("PCAM", "4"): 8036,
    ("STL10", "4"): 8640,
    ("Flowers102", "4"): 4704,
    ("RenderedSST2", "4"): 7644,
    ("EuroSAT", "4"): 7296,
    ("RESISC45", "4"): 7980,
    ("SUN397", "4"): 7840,
    ("FER2013", "4"): 8080,
    ("OxfordIIITPet", "4"): 8528,
    ("ImageNet", "0.25"): 2493,
}

FAIL_MARKERS = ("FT_FAIL", "BL_FAIL", "FT_PARKED")

# Repo roots of the two rigs that are not the hub.
#
# These are needed because the replicator's `replicate-list` copies only
# `classifier_epoch_N.pt` and `head_epoch_N.pt` -- not `run_meta.json`. A
# checkpoint trained on behemoth therefore arrives on the hub *without the
# evidence of its own budget*, and a purely local check would report "absent"
# forever for two thirds of the wave while looking perfectly healthy. Silence
# that means "I never looked" is the precise failure this script exists to
# prevent, so the metadata is fetched from the rig that wrote it.
#
# The fetch resolves the path with the *remote* rig's own bx_common rather than
# spelling it here: the three rigs mount CHECKPOINT_BASE_PATH differently
# (rig-3090-ti keeps checkpoints on /mnt/WD_4TB), and a hand-spelled path would
# be a guard that drifts from its writer -- the bug class the whole bx_ layer is
# built to avoid.
REMOTE_ROOTS = {
    "behemoth": "/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer",
    "rig-3090-ti": "/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer",
}

# Never `uv run` on behemoth: sm_120 needs its pinned cu128/cu129 torch build.
REMOTE_PY = ".venv/bin/python"

_REMOTE_SNIPPET = (
    "import sys,os;"
    "sys.path.insert(0,'scripts/dispatch/budget_axis');"
    "import bx_common as bx;"
    "p=os.path.join(bx.ckpt_dir(sys.argv[1],sys.argv[2],sys.argv[3]),'run_meta.json');"
    "print(open(p).read() if os.path.isfile(p) else '')"
)


def remote_meta(rig, kind, dataset, mult, timeout=20):
    """`run_meta.json` for one item as the rig that produced it sees it.

    Returns the parsed dict, or None for absent/unreachable/unparseable -- all of
    which mean "try again next tick", never "the budget is wrong". Distinguishing
    those would add noise without changing what anyone would do.
    """
    root = REMOTE_ROOTS.get(rig)
    if root is None:
        return None
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", rig,
        f"cd {root} && {REMOTE_PY} -c {json.dumps(_REMOTE_SNIPPET)} "
        f"{kind} {dataset} {mult}",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if p.returncode != 0 or not p.stdout.strip():
        return None
    try:
        return json.loads(p.stdout)
    except ValueError:
        return None

# ---------------------------------------------------------------------------
# status.txt parsing
#
# The dashboard's text is the interface. Parsing text is fragile in general, but
# here the alternative -- re-running the ssh probes from a second process -- would
# double the load on rigs that are already saturated and would disagree with the
# dashboard the user is reading. Every regex below is anchored on a literal the
# dashboard writes unconditionally, and a parse that finds nothing reports itself
# (see `check` on empty `rigs`) rather than passing quietly.
# ---------------------------------------------------------------------------

_RE_RIG_OK = re.compile(
    r"^(?P<rig>\S+)\s+tmux=(?P<tmux>\d+)\s+gpu\[(?P<gpu>[^\]]*)\]\s+boot=(?P<boot>.+)$"
)
_RE_RIG_BAD = re.compile(r"^(?P<rig>\S+)\s+(?P<state>DOWN|unreachable \(attempt \d+/3\)):")
_RE_GLOBAL = re.compile(
    r"^GLOBAL\s+finetunes\s+(?P<ftd>\d+)/(?P<ftt>\d+)\s+"
    r"baselines\s+(?P<bld>\d+)/(?P<blt>\d+)\s+cells\s+(?P<ced>\d+)/(?P<cet>\d+)"
)
_RE_GRIDS = re.compile(r"G_(?:SS|LL|SL|LS) \d+/\d+")
_RE_QUEUES = re.compile(r"^queues\s+ft=(?P<ft>\d+) left\s+ev=(?P<ev>\d+) left")


def boot_drift_seconds(a, b):
    """Seconds between two `uptime -s` strings, or None if either is unparseable.

    None means "cannot tell", and the caller treats that as a reboot: failing
    toward a spoken alarm is the right direction for a check whose whole purpose
    is to not stay quiet about things it did not verify.
    """
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        ta = time.mktime(time.strptime(a.strip(), fmt))
        tb = time.mktime(time.strptime(b.strip(), fmt))
    except (ValueError, TypeError):
        return None
    return abs(tb - ta)


def parse_status(text):
    """Pull the few facts worth alarming on out of the dashboard's text."""
    out = {"rigs": {}, "global": None, "grids": "", "eta": "", "stamp": ""}
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if line.startswith("BUDGET-AXIS WAVE"):
            out["stamp"] = stripped.split("   ", 1)[-1].strip()
            continue

        # Rig headers sit at column 0; every per-rig detail line is indented, so
        # the anchor is cheap and unambiguous.
        if line and not line[0].isspace():
            m = _RE_RIG_OK.match(stripped)
            if m:
                out["rigs"][m["rig"]] = {
                    "ok": True,
                    "tmux": int(m["tmux"]),
                    "boot": m["boot"].strip(),
                    "state": "up",
                }
                continue
            m = _RE_RIG_BAD.match(stripped)
            if m:
                out["rigs"][m["rig"]] = {
                    "ok": False,
                    "tmux": 0,
                    "boot": "",
                    "state": m["state"],
                }
                continue

        m = _RE_QUEUES.match(stripped)
        if m and out["rigs"]:
            r = out["rigs"][next(reversed(out["rigs"]))]
            r["ft_left"], r["ev_left"] = int(m["ft"]), int(m["ev"])
            continue

        if stripped.startswith("FAIL "):
            # Attribute the failure to the rig whose block it appears in.
            rig = next(reversed(out["rigs"])) if out["rigs"] else "?"
            out["rigs"].setdefault(rig, {}).setdefault("fails", []).append(stripped[5:])
            continue

        m = _RE_GLOBAL.match(stripped)
        if m:
            out["global"] = {k: int(v) for k, v in m.groupdict().items()}
            continue

        if _RE_GRIDS.match(stripped):
            out["grids"] = stripped
            continue

        if "eta" in stripped and ("cells" in stripped or "baselines" in stripped):
            out["eta"] = stripped

    return out


# ---------------------------------------------------------------------------
# Budget verification
# ---------------------------------------------------------------------------

def local_meta(kind, dataset, mult):
    """`run_meta.json` beside a locally-produced checkpoint, or None."""
    path = os.path.join(bx.ckpt_dir(kind, dataset, mult), "run_meta.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        # A truncated file is normal while it is being written; the next tick
        # sees the whole thing. Calling it a budget failure would cry wolf.
        return None


def verify_budget(meta, dataset, mult):
    """Prove that a landed checkpoint realized the budget its path claims.

    Returns (status, detail) where status is "ok" or "bad".

    The axis is only an axis if the budgets are real. A multiplier that silently
    stayed at 1x would not fail anything -- it would produce a complete,
    plausible, wrong grid, and the finding it feeds ("QAT is far more
    budget-sensitive than FP") would be an artifact of a directory name. So each
    checkpoint is checked the moment it appears, against five conditions:

      1. self-consistency: max_steps == epoch_mult * base_epochs * num_batches
      2. the independent expected value from the approved enumeration
      3. base_epochs matches the dataset's 1x entry in DATASET_NAME_TO_EPOCHS
      4. warmup_length < max_steps -- at mult=0.25 the un-rescaled wl=500 is 20%
         of the schedule; if it ever exceeded it the run would be all warmup
      5. epoch_mult agrees with the mult= fragment of the path it was written to

    Condition 1 alone would accept a run whose num_batches was wrong; condition 2
    alone would accept a run that reached the right step count for the wrong
    reason. Together they pin both the schedule and its derivation.
    """
    try:
        max_steps = int(meta["max_steps"])
        base_epochs = int(meta["base_epochs"])
        num_batches = int(meta["num_batches"])
        warmup = int(meta["warmup_length"])
        emult = float(meta["epoch_mult"])
    except (KeyError, TypeError, ValueError) as exc:
        return "bad", f"run_meta.json missing/!int field: {exc}"

    problems = []

    derived = round(emult * base_epochs * num_batches)
    if derived != max_steps:
        problems.append(
            f"max_steps={max_steps} != epoch_mult*base_epochs*num_batches={derived}"
        )

    expected = EXPECTED_MAX_STEPS.get((dataset, str(mult)))
    if expected is None:
        problems.append(f"no expected step count for ({dataset}, mult={mult})")
    elif max_steps != expected:
        problems.append(f"max_steps={max_steps} != expected {expected}")

    base_1x = bx.DATASET_NAME_TO_EPOCHS.get(dataset)
    if base_1x is not None and base_epochs != base_1x:
        problems.append(f"base_epochs={base_epochs} != 1x entry {base_1x}")

    if warmup >= max_steps:
        problems.append(f"warmup_length={warmup} >= max_steps={max_steps}")

    if abs(emult - float(mult)) > 1e-9:
        problems.append(f"epoch_mult={emult} != path mult={mult}")

    if problems:
        return "bad", "; ".join(problems)
    return "ok", f"max_steps={max_steps} warmup={warmup}"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def load_state(path):
    try:
        with open(path) as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = {}
    st.setdefault("rigs", {})        # rig -> last reported condition string
    st.setdefault("budgets", {})     # "kind/ds/mult" -> "ok" | "bad"
    st.setdefault("fails", [])       # failure lines already spoken
    st.setdefault("flags", {})       # one-shot conditions: stale, complete
    return st


def save_state(path, st):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, path)            # never leave a half-written state file


def edge(state_map, key, value):
    """True the first time `key` takes `value` -- how a condition speaks once.

    Recording the *value*, not a boolean, is what lets a condition speak again
    when it changes (up -> DOWN -> up -> DOWN) without repeating itself while it
    holds.
    """
    if state_map.get(key) == value:
        return False
    state_map[key] = value
    return True


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def check(args, st, out):
    status_path = args.status
    now = time.time()

    if not os.path.isfile(status_path):
        if edge(st["flags"], "status_missing", "yes"):
            out.append(f"DASHBOARD MISSING: {status_path} does not exist")
        return
    st["flags"].pop("status_missing", None)

    age = now - os.path.getmtime(status_path)
    # The dashboard rewrites every 120 s. Tolerating a couple of missed cycles
    # keeps a slow ssh probe from being an alarm; beyond that, the loop is gone.
    stale = age > args.stale_after
    if edge(st["flags"], "stale", "yes" if stale else "no") and stale:
        out.append(
            f"DASHBOARD STALE: status.txt last written {age / 60:.1f} min ago "
            f"(> {args.stale_after / 60:.0f} min). The on-rig status loop or "
            f"rig-4090 may be down; the wave itself may still be fine."
        )

    with open(status_path) as f:
        snap = parse_status(f.read())

    if not snap["rigs"]:
        if edge(st["flags"], "unparsed", "yes"):
            out.append("DASHBOARD UNPARSEABLE: no rig blocks found in status.txt")
        return
    st["flags"].pop("unparsed", None)

    # ---- rigs ----
    for rig, expected_tmux in EXPECTED_TMUX.items():
        r = snap["rigs"].get(rig)
        if r is None:
            if edge(st["rigs"], rig, "absent"):
                out.append(f"RIG {rig}: no block in status.txt")
            continue

        if not r["ok"]:
            if edge(st["rigs"], rig, r["state"]):
                out.append(
                    f"RIG {rig} {r['state']}. Not acting: a rebooting rig returns "
                    f"on its own. Recovery, on your word, is "
                    f"`bash scripts/dispatch/budget_axis/bx_reassign.sh "
                    f"{args.wave_id} {rig}`."
                )
            continue

        # A reboot is "interrupted, not failed" -- the item in flight left no
        # artifact and no queue entry, so it needs reconciling against the full
        # enumeration rather than a restart.
        #
        # Compared with a tolerance because the dashboard reads `uptime -s`, which
        # derives boot as now-minus-uptime and therefore jitters by a second or
        # two between samples. An exact comparison reports a reboot on that
        # jitter, and an alarm that cries wolf every few minutes is worse than no
        # alarm -- it teaches its reader to skip the line that will one day be
        # real. The stored value is only replaced when a reboot is *declared*, so
        # jitter cannot accumulate past the tolerance one second at a time.
        boot_key = f"{rig}:boot"
        prev_boot = st["rigs"].get(boot_key)
        if prev_boot is None:
            st["rigs"][boot_key] = r["boot"]
        elif prev_boot != r["boot"]:
            drift = boot_drift_seconds(prev_boot, r["boot"])
            if drift is None or drift > args.boot_tolerance:
                out.append(
                    f"RIG {rig} REBOOTED (boot {prev_boot} -> {r['boot']}). Any "
                    f"item in flight is interrupted, not failed, and must be "
                    f"reconciled against the enumeration, not restarted."
                )
                st["rigs"][boot_key] = r["boot"]

        # Once this rig's finetune queue is empty its finetune lanes are entitled
        # to have exited, so drop them from the expectation rather than reporting
        # a normal shutdown as a failure.
        want = expected_tmux
        if r.get("ft_left") == 0:
            want -= FT_LANES.get(rig, 0)
        cond = "up" if r["tmux"] >= want else f"tmux={r['tmux']}/{want}"
        if edge(st["rigs"], rig, cond) and cond != "up":
            out.append(
                f"RIG {rig} DEGRADED: {r['tmux']} tmux sessions, expected "
                f"{want} (ft.q has {r.get('ft_left', '?')} left, so retired "
                f"finetune lanes are already discounted). A lane died; the wave "
                f"still progresses, so nothing else would report this. "
                f"Relaunch is idempotent: "
                f"`bash scripts/dispatch/budget_axis/waves/{args.wave_id}/{rig}"
                f"/launch_local.sh`."
            )

        for line in r.get("fails", []):
            if any(m in line for m in FAIL_MARKERS) and line not in st["fails"]:
                st["fails"].append(line)
                out.append(f"FAILURE on {rig}: {line}")

    # ---- budgets ----
    #
    # A verified item is never re-read, so the steady state is a handful of
    # stat() calls plus an ssh only for checkpoints that have just landed from a
    # remote rig. `remote_budget_probes` caps the ssh fan-out on the first tick
    # after a long gap, when many items could be pending at once; the rest are
    # picked up on later ticks rather than stalling the check behind a queue of
    # connections to a saturated rig.
    probes_left = args.remote_budget_probes
    for kind, ds, mult in bx.finetune_items():
        key = f"{kind}/{ds}/{mult}"
        if st["budgets"].get(key) == "ok":
            continue                                  # verified once, stays verified

        meta = local_meta(kind, ds, mult)
        if meta is None:
            # Only ask a remote rig once the checkpoint itself is here: before
            # that the run is simply unfinished, and an ssh per tick per pending
            # item would be a self-inflicted load on the rigs doing the work.
            if not os.path.isfile(bx.ckpt_file(kind, ds, mult)) or probes_left <= 0:
                continue
            for rig in REMOTE_ROOTS:
                probes_left -= 1
                meta = remote_meta(rig, kind, ds, mult)
                if meta is not None:
                    break
            if meta is None:
                continue                              # retry on a later tick

        verdict, detail = verify_budget(meta, ds, mult)
        if verdict == "ok":
            st["budgets"][key] = "ok"                 # a pass is silent
        elif edge(st["budgets"], key, "bad"):
            out.append(f"BUDGET UNVERIFIED {kind} {ds} mult={mult}: {detail}")

    # ---- completion ----
    g = snap["global"]
    if g:
        done = (
            g["ftd"] >= g["ftt"] and g["bld"] >= g["blt"] and g["ced"] >= g["cet"]
        )
        if done and edge(st["flags"], "complete", "yes"):
            out.append(
                f"WAVE COMPLETE: finetunes {g['ftd']}/{g['ftt']}, baselines "
                f"{g['bld']}/{g['blt']}, cells {g['ced']}/{g['cet']}. Remaining "
                f"work is tasks 9-10 of the plan: reconcile by artifact, pull "
                f"evaluations to rig-4090, assert the totals locally, then plot."
            )

    # ---- periodic status, regardless of quiet ----
    if args.report_every > 0 and args.tick % args.report_every == 0 and g:
        parts = [
            f"STATUS {snap['stamp']} | finetunes {g['ftd']}/{g['ftt']} "
            f"baselines {g['bld']}/{g['blt']} cells {g['ced']}/{g['cet']}"
        ]
        if snap["grids"]:
            parts.append(f"  {snap['grids']}")
        if snap["eta"]:
            parts.append(f"  {snap['eta']}")
        out.append("\n".join(parts))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wave-id", required=True)
    ap.add_argument("--status", required=True, help="path to the dashboard's status.txt")
    ap.add_argument("--state", required=True, help="JSON file holding what was already said")
    ap.add_argument("--tick", type=int, default=0, help="loop iteration, for --report-every")
    ap.add_argument(
        "--report-every", type=int, default=5,
        help="emit a status line every Nth tick regardless of quiet; 0 disables",
    )
    ap.add_argument(
        "--stale-after", type=float, default=360.0,
        help="seconds after which an unwritten status.txt is itself an alarm",
    )
    ap.add_argument(
        "--boot-tolerance", type=float, default=120.0,
        help="seconds of boot-time movement to treat as `uptime -s` jitter "
             "rather than a reboot",
    )
    ap.add_argument(
        "--remote-budget-probes", type=int, default=6,
        help="max ssh fetches of run_meta.json per tick (the replicator does not "
             "copy it, so remotely-trained checkpoints must be asked about)",
    )
    args = ap.parse_args(argv)

    st = load_state(args.state)
    out = []
    try:
        check(args, st, out)
    except Exception as exc:                                   # noqa: BLE001
        # A check that crashes must say so. Swallowing it would reproduce exactly
        # the failure this script exists to prevent: silence that looks like calm.
        out.append(f"HEALTH CHECK ERROR: {type(exc).__name__}: {exc}")
    else:
        save_state(args.state, st)

    for line in out:
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
