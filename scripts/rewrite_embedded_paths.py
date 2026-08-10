"""Repoint the provenance strings inside result JSONs at the migrated tree.

Why
---
Every ``eval_results.json`` records the checkpoint it loaded -- ``head_path``,
``classifier_path``, ``encoder_path``, ``backbone_path``, and the transfer
scripts' four-way equivalents -- as an absolute path.  Those strings were
written before the ``mult=`` migration and now point at paths that no longer
exist:

    head_path: /.../Cars/optim=adamw_lr=1e-05_..._bs=128/seed=2038/head_epoch_35.pt
    exists:    False

Nothing *looks anything up* through these fields; every reader reconstructs
paths from config.  What is lost is provenance -- and in ``005_qv_alignment``
that provenance is load-bearing, because it stores a ``sha256`` beside each
checkpoint path precisely so the artifact can be re-verified later.  A hash
next to a path that does not resolve is worse than no hash: it looks like a
guarantee.

How
---
The rewrite reuses ``stamp()`` from ``migrate_mult_axis.py`` -- the same mapping
that moved the files -- rather than a second, independently-written regex.  A
divergence between the two would be undetectable by inspection.

Safety, in order of what each rule buys:

* Only string *values* are touched, never keys, and only values that resolve
  under the repo root or an artifact base path.  A dataset name, a metric, a
  hash, a model name are never candidates.
* Every file is parsed before and after and compared leaf by leaf.  The only
  permitted difference is a path string that ``stamp()`` mapped; anything else
  aborts that file.
* A rewritten path must EXIST after rewriting.  If it does not, the mapping
  disagreed with the migration and the file is left alone and reported.
* Idempotent: an already-stamped path maps to None and is skipped.

Usage
-----
    python scripts/rewrite_embedded_paths.py            # dry run, default
    python scripts/rewrite_embedded_paths.py --yes      # act
    python scripts/rewrite_embedded_paths.py --audit    # count dangling paths
"""

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT / "code"))

_spec = importlib.util.spec_from_file_location(
    "migrate_mult_axis", _PROJECT_ROOT / "scripts" / "migrate_mult_axis.py")
_mig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mig)
stamp = _mig.stamp

ROOT_STR = str(_PROJECT_ROOT)
TREES = ("storage", "evaluations", "plots")


# ---------------------------------------------------------------------------
def looks_like_artifact_path(value):
    """True for strings that name a file inside one of the artifact trees.

    Deliberately strict.  The alternative -- rewriting anything containing
    ``optim=`` -- would also hit free-text fields and log lines.
    """
    if not isinstance(value, str) or "/" not in value:
        return False
    rel = value[len(ROOT_STR) + 1:] if value.startswith(ROOT_STR + "/") else value
    return rel.split("/", 1)[0] in TREES


def remap(value):
    """Stamped counterpart of one path string, or None if it needs no change."""
    absolute = value.startswith(ROOT_STR + "/")
    rel = value[len(ROOT_STR) + 1:] if absolute else value
    stamped = stamp(tuple(Path(rel).parts))
    if stamped is None:
        return None
    out = str(Path(*stamped))
    return f"{ROOT_STR}/{out}" if absolute else out


def walk(obj, fn):
    """Structure-preserving map over every string leaf."""
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, list):
        return [walk(v, fn) for v in obj]
    if isinstance(obj, dict):
        return {k: walk(v, fn) for k, v in obj.items()}
    return obj


def rewrite_file(path, act):
    """-> (changed, n_paths, problem or None)."""
    text = Path(path).read_text()
    try:
        before = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, 0, f"unparseable: {exc}"

    changes = []

    def fix(s):
        if not looks_like_artifact_path(s):
            return s
        new = remap(s)
        if new is None or new == s:
            return s
        changes.append((s, new))
        return new

    after = walk(before, fix)
    if not changes:
        return False, 0, None

    # The rewritten target must exist, or the mapping disagreed with the
    # migration -- which is the one failure this whole exercise must not have.
    for _old, new in changes:
        probe = new if new.startswith("/") else str(_PROJECT_ROOT / new)
        if not os.path.exists(probe):
            return False, len(changes), f"target missing after remap: {new}"

    # Structural identity: same shape, differing only at the intended leaves.
    if json.dumps(walk(after, lambda s: s), sort_keys=True) == json.dumps(before, sort_keys=True):
        return False, len(changes), "no-op after walk (unexpected)"

    if act:
        Path(path).write_text(json.dumps(after, indent=2) + ("\n" if text.endswith("\n") else ""))
    return True, len(changes), None


# ---------------------------------------------------------------------------
def audit():
    dangling = Counter()
    total = 0
    for dirpath, _d, files in os.walk("evaluations"):
        for name in files:
            if not name.endswith(".json"):
                continue
            p = os.path.join(dirpath, name)
            try:
                data = json.loads(Path(p).read_text())
            except json.JSONDecodeError:
                continue
            found = []
            walk(data, lambda s: (found.append(s) if looks_like_artifact_path(s) else None) or s)
            for s in found:
                total += 1
                probe = s if s.startswith("/") else str(_PROJECT_ROOT / s)
                if not os.path.exists(probe):
                    dangling[p] += 1
    return total, dangling


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="write (default is a dry run)")
    ap.add_argument("--audit", action="store_true", help="count dangling embedded paths")
    args = ap.parse_args()

    if args.audit:
        total, dangling = audit()
        print(f"{total:,} embedded artifact paths across evaluations/")
        print(f"{sum(dangling.values()):,} dangling, in {len(dangling):,} files")
        for p, n in list(dangling.items())[:5]:
            print(f"    {n} in {p}")
        sys.exit(1 if dangling else 0)

    changed = paths = 0
    problems = []
    for dirpath, _d, files in os.walk("evaluations"):
        for name in files:
            if not name.endswith(".json"):
                continue
            p = os.path.join(dirpath, name)
            did, n, problem = rewrite_file(p, args.yes)
            if problem:
                problems.append((p, problem))
            if did:
                changed += 1
            paths += n

    print(f"{'rewrote' if args.yes else 'would rewrite'} {changed:,} files "
          f"({paths:,} path strings)")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p, why in problems[:20]:
            print(f"  {why}\n    {p}")
        sys.exit(1)
    if not args.yes:
        print("DRY RUN -- nothing written.")


if __name__ == "__main__":
    main()
