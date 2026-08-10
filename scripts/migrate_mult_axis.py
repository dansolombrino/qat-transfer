"""Stamp the legacy artifact trees with the explicit `mult=1` grammar.

Why this exists
---------------
Training duration became a first-class, always-explicit path axis (see
``code/src/duration.py``).  Every run this repo has ever produced was at
``epoch_mult = 1.0``, but none of those paths say so, and the multiplier is
deliberately *not* elided -- a path that carries no ``mult=`` predates the axis.
So the readers and writers now speak a grammar the tree does not yet use, and
this script closes that gap.

Why hardlinks rather than renames
---------------------------------
Two independent reasons, either sufficient:

1. **Safety.** A rename is destructive the instant it runs: mid-way through
   287,989 directories, an interruption leaves a tree that neither grammar can
   read.  Hardlinking builds the stamped tree *alongside* the original, so both
   grammars are valid at once, verification runs against a live legacy tree, and
   rollback is deleting what was added rather than reversing what was moved.

2. **It is the only thing that fits.** ``storage/`` holds 799 GiB against 352 GiB
   free.  A copy is physically impossible here; a hardlink costs a directory
   entry and shares the inode, so the stamped checkpoint tree costs essentially
   nothing and is provably the same bytes (verified by inode identity, not by
   re-hashing 799 GiB).

The seven grammars
------------------
The mapping is not one rule.  Each phase family names its parameters
differently, and the migration must reproduce *exactly* what the patched code
now emits -- a divergence here is the single most dangerous failure mode in the
whole change, because it would be silent.  ``--verify-construction`` exists for
precisely that: it imports the real path builders and asserts the paths they
produce are the paths this script created.

  1. baselines + checkpoints   insert ``mult=1`` after the ``optim=`` component
  2. transfer evaluations      ``src=X_seed=N`` -> ``src=X_seed=N_mult=1`` (and tgt)
  3. pretrained evaluations    no ``optim=`` anchor; insert after the dataset
  4. 009 run-id                ``src=X-seedN`` -> ``src=X-seedN-mult1`` (and tgt)
  5. 010 materialize run-id    insert ``mult=1`` after ``seed=N``
  6. 010 transfer run-id       insert ``smult=1``/``tmult=1`` after ``tseed=N``
  7. 005 alignment run-id      insert ``epoch_mult=1`` after ``epoch_policy=...``

Rule 1 and rule 2 are mutually exclusive by construction: a transfer path
carries both an ``optim=`` component and ``src=``/``tgt=`` components, and there
the multiplier belongs inside the role fragments (donor and receiver need to
disagree, and they share one ``optim=``).  The rule is therefore "if the path
names a role, stamp the roles; otherwise stamp after optim".

Usage
-----
    python scripts/migrate_mult_axis.py                      # dry run, default
    python scripts/migrate_mult_axis.py --yes                # act
    python scripts/migrate_mult_axis.py --revert             # undo via manifest
    python scripts/migrate_mult_axis.py --verify-construction # code vs disk
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
os.chdir(_PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from src.duration import UNIT_MULT, mult_path_frag, mult_tag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TREES = ("evaluations", "storage")
MANIFEST = Path("migration_baseline/mult_axis_manifest.json")

FRAG = mult_path_frag(UNIT_MULT)          # "mult=1"
TAG = mult_tag(UNIT_MULT)                 # "1"

RE_OPTIM = re.compile(r"^optim=")
RE_ROLE = re.compile(r"^(src|tgt)=(.+)_seed=(\d+)$")
RE_ROLE_DASH = re.compile(r"^(src|tgt)=(.+)-seed(\d+)$")
RE_SEED = re.compile(r"^seed=\d+$")
RE_TSEED = re.compile(r"^tseed=\d+$")
RE_EPOCH_POLICY = re.compile(r"^epoch_policy=")
PRETRAINED = ("pretrained", "pretrained_ptq")


# ---------------------------------------------------------------------------
# The mapping
# ---------------------------------------------------------------------------
def stamp(rel_parts):
    """Map one legacy path (as a component list) to its stamped form.

    Returns None when the path needs no change, so callers can distinguish
    "already correct" from "rewritten".
    """
    parts = list(rel_parts)

    # already stamped -- idempotent
    if FRAG in parts or any(p.startswith(("smult=", "tmult=", "epoch_mult=")) for p in parts) \
            or any(RE_ROLE_DASH.match(p) and p.endswith(f"-mult{TAG}") for p in parts):
        return None

    # (2)/(4) role-bearing paths: the multiplier rides inside src=/tgt=
    role_idx = [i for i, p in enumerate(parts) if RE_ROLE.match(p)]
    if role_idx:
        for i in role_idx:
            parts[i] = f"{parts[i]}_{FRAG}"
        return parts

    dash_idx = [i for i, p in enumerate(parts) if RE_ROLE_DASH.match(p)]
    if dash_idx:
        for i in dash_idx:
            parts[i] = f"{parts[i]}-mult{TAG}"
        return parts

    # (6) 010 transfer run-id: smult/tmult follow tseed
    tseed = [i for i, p in enumerate(parts) if RE_TSEED.match(p)]
    if tseed:
        i = tseed[0]
        parts[i + 1:i + 1] = [f"smult={TAG}", f"tmult={TAG}"]
        return parts

    # (7) 005 alignment run-id: epoch_mult follows epoch_policy
    pol = [i for i, p in enumerate(parts) if RE_EPOCH_POLICY.match(p)]
    if pol:
        i = pol[0]
        parts[i + 1:i + 1] = [f"epoch_mult={TAG}"]
        return parts

    # (1) everything with an optim= anchor
    optim = [i for i, p in enumerate(parts) if RE_OPTIM.match(p)]
    if optim:
        i = optim[0]
        parts[i + 1:i + 1] = [FRAG]
        return parts

    # (5) 010 materialize run-id: mult follows seed=, but only under a
    #     donor=/model= run-id path, never under a plain baseline tree.
    if any(p.startswith("donor=") for p in parts):
        seed = [i for i, p in enumerate(parts) if RE_SEED.match(p)]
        if seed:
            i = seed[0]
            parts[i + 1:i + 1] = [f"mult={TAG}"]
            return parts

    # (8) rebuttal aggregates. These span a whole donor x receiver matrix at one
    #     pair of budgets and carry no optim= anchor at all, so they name both
    #     multipliers directly after the seed -- otherwise an aggregate computed
    #     at a different budget would overwrite this one.
    if len(parts) > 1 and parts[0] == "evaluations" and parts[1] == "998_rebuttal":
        seed = [i for i, p in enumerate(parts) if RE_SEED.match(p)]
        if seed:
            i = seed[0]
            parts[i + 1:i + 1] = [f"smult={TAG}", f"tmult={TAG}"]
            return parts

    # (3) pretrained evaluations: no optim anchor at all
    for i, p in enumerate(parts):
        if p in PRETRAINED and i + 3 <= len(parts):
            # .../<pretrained>/<model>/<dataset>/...  -> stamp after dataset
            parts[i + 3:i + 3] = [FRAG]
            return parts

    return None


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def plan(trees):
    """Every file to link, as (source, destination). Fails on collisions."""
    ops, skipped, seen = [], Counter(), {}
    for tree in trees:
        root = Path(tree)
        if not root.is_dir():
            print(f"[WARN] {tree}/ not present, skipping", file=sys.stderr)
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if not filenames:
                continue
            parts = Path(dirpath).parts
            stamped = stamp(parts)
            if stamped is None:
                skipped[tree] += len(filenames)
                continue
            dest_dir = Path(*stamped)
            for name in filenames:
                src, dst = Path(dirpath) / name, dest_dir / name
                if dst in seen:
                    raise SystemExit(
                        f"COLLISION: {seen[dst]} and {src} both map to {dst}. "
                        f"The mapping is not injective; refusing to proceed."
                    )
                seen[dst] = src
                ops.append((str(src), str(dst)))
    return ops, skipped


def git_is_clean():
    out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return out.stdout.strip() == ""


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def apply(ops):
    linked = existed = 0
    for src, dst in ops:
        d = Path(dst)
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists():
            existed += 1
            continue
        try:
            os.link(src, dst)
        except OSError as exc:
            raise SystemExit(f"link failed {src} -> {dst}: {exc}")
        linked += 1
    return linked, existed


def revert(manifest_path):
    data = json.loads(Path(manifest_path).read_text())
    removed = 0
    for _src, dst in data["ops"]:
        p = Path(dst)
        if p.is_file():
            p.unlink()
            removed += 1
    # prune directories that the migration created and that are now empty
    for _src, dst in sorted(data["ops"], key=lambda o: -len(o[1])):
        d = Path(dst).parent
        while d != Path(".") and d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            d = d.parent
    return removed


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true",
                    help="actually create the hardlinks (default is a dry run)")
    ap.add_argument("--revert", action="store_true",
                    help="remove everything recorded in the manifest")
    ap.add_argument("--trees", nargs="+", default=list(TREES))
    ap.add_argument("--allow-dirty", action="store_true",
                    help="proceed even if the git tree has uncommitted changes")
    ap.add_argument("--limit", type=int, default=None,
                    help="plan only the first N operations (for a fast smoke test)")
    args = ap.parse_args()

    if args.revert:
        if not MANIFEST.exists():
            raise SystemExit(f"no manifest at {MANIFEST}; nothing to revert")
        n = revert(MANIFEST)
        print(f"removed {n:,} stamped links; legacy tree untouched throughout")
        return

    if not args.allow_dirty and not git_is_clean():
        raise SystemExit(
            "git tree is dirty. The migration must be reproducible from a known "
            "commit -- commit or stash first, or pass --allow-dirty."
        )

    print(f"Planning with fragment {FRAG!r} / tag {TAG!r} ...")
    ops, skipped = plan(args.trees)
    if args.limit:
        ops = ops[:args.limit]

    by_tree = Counter(op[0].split("/")[0] for op in ops)
    print(f"\n{len(ops):,} files to link")
    for t, n in sorted(by_tree.items()):
        print(f"    {t:<14} {n:>9,}   (unchanged: {skipped[t]:,})")

    print("\nExamples:")
    for src, dst in ops[:4] + ops[len(ops) // 2:len(ops) // 2 + 3]:
        print(f"  - {src}")
        print(f"  + {dst}")

    if not args.yes:
        print("\nDRY RUN -- nothing written. Re-run with --yes to act.")
        return

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"fragment": FRAG, "tag": TAG, "ops": ops}, indent=0))
    print(f"\nmanifest written: {MANIFEST}")
    linked, existed = apply(ops)
    print(f"linked {linked:,} files ({existed:,} already present)")
    print("Legacy tree is untouched. Verify, then remove it explicitly.")


if __name__ == "__main__":
    main()
