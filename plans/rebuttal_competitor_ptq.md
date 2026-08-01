# Plan — Rebuttal: competitor PTQ baselines (Task 1) and QV-on-top-of-strong-PTQ (Task 2)

**Role of this document.** This is the coordination plan for the rebuttal's
baseline/competitor experiments. It is written to be handed to fresh Claude
sessions ("executors") that implement one work package each. Executors: read
this file *and* `CLAUDE.md` (conventions are mandatory) before writing code.
Do not re-derive strategy — the decisions below are already made. If a decision
here conflicts with what you find in the code, stop and report rather than
improvise.

**Status legend:** `[ ]` todo · `[~]` in progress (note the session/date) · `[x]` done.

---

## 1. Background and goal

The paper (quantization vectors, `QV = QAT_D - FP_D`, transferred across tasks
— see `README.md` and `CLAUDE.md` for the full setup) received reviews with two
baseline-related asks:

1. **Task 1 — compare against strong recent PTQ**, not just vanilla RTN
   per-channel PTQ (`apply_ptq_`). Reviewers named GPTQ, VPTQ, SliM-LLM.
2. **Task 2 — show QV patching retains benefit *on top of* stronger
   quantization** (reviewers named GPTQ, AWQ, SmoothQuant; PV-tuning is a
   separate stronger-QAT question, out of scope here).

**Resolution of the "those are LLM methods" problem.** The named methods are
examples, not the requirement; the objection is "your baseline is vanilla RTN".
GPTQ is architecture-agnostic (layer-wise Hessian-compensated quantization of
`nn.Linear` weights) and ports directly to ViT/BERT. VPTQ and SliM-LLM are
LLM-bound by construction (decoder-scale codebooks, salience-based bit
allocation, perplexity eval); we do NOT port them. Instead we cover "recent
strong PTQ for the architectures we study" with ViT-native SOTA (RepQ-ViT
first). The rebuttal text will state this substitution explicitly.

**Target experiment matrix** (3-bit, canonical config from `CLAUDE.md`, one
architecture suffices per reviewer 3HFP; more is better):

```
{RTN, GPTQ, RepQ-ViT[, APHQ-ViT, PTQ4ViT]}  ×  {FP, FP + QV}
```

Row block `× FP` answers Task 1 (is QV+RTN competitive with strong PTQ on FP?).
Column `FP + QV` answers Task 2 (does QV still add gain under strong PTQ?).
The key survivable narrative even if GPTQ(FP) beats QV+RTN: **QV is
complementary to modern PTQ** — so Task 2 numbers matter as much as Task 1.

## 2. History — read before implementing

All of this existed before and was deliberately deleted in commit `3516c46`
("remove all competitor PTQ baselines") because the old implementations were
low-quality LLM-generated code. **We are redoing them from scratch.** The old
tree is still the authoritative map of file names, path grammar, and config
shape:

```
git show 3516c46 --stat                       # what existed
git show 3516c46^:<path>                      # read an old file (read-only reference)
```

Use the old files as an *interface spec only* (script names, YAML keys,
`experiment_type` directory names, eval-path grammar) — do **not** copy their
algorithm code. The old competitor `evaluations/` subtrees were purged and are
not recoverable; all runs will be redone.

## 3. Non-negotiable constraints (from CLAUDE.md, enforced)

- Executors write code only. **Never run** GPU/finetune/experiment scripts;
  they need checkpoints under `storage/` and GPU-hours. The user dispatches
  runs to the rigs separately.
- Hydra script ⇒ matching YAML under `config/` mirroring the code path 1:1.
- `skip_modules` always explicit (`???` in YAML), never defaulted.
- In-place mutators end in `_`; follow the `apply_ptq_` calling convention.
- Eval results: JSON only, one `eval_results.json` per run, path grammar of
  `CLAUDE.md` §"Evaluation paths" with the `experiment_type` directory naming
  the variant (e.g. `fp_gptq`, `fp_gptq_dryrun`).
- Import order / dotenv-first / SLURM-aware logging exactly as in `CLAUDE.md`.
- Module docstrings must state the question and why the methodology answers it
  (998_rebuttal register). One-line docstrings on experiment scripts are a defect.

## 4. Work packages

Execute in order; WP1 unblocks both tasks.

### WP1 — Native GPTQ implementation  `[x]`  (Priority 1)

> Done 2026-08-01 (executor session). Delivered `code/src/gptq.py` (official-fidelity
> sequential GPTQ on the project grid; deviations + impact documented in the module
> docstring) and `code/test/gptq.py` (8 CPU tests, all passing; GPTQ ≈ 2× lower
> layer-output error than RTN at 3-bit/channel on correlated inputs).

**Deliverables**
- `code/src/gptq.py`: from-scratch GPTQ. Public entry point
  `apply_gptq_(model, bits, granularity, skip_modules, calib_loader, device, ...)`
  mirroring `apply_ptq_`'s in-place convention: fake-quantizes `nn.Linear`
  weights in place, replaces no modules (so forward hooks survive, same as
  `apply_ptq_` — see CLAUDE.md on `003_qat_transfer_activ`).
  Algorithm per Frantar et al. 2022/2023 (ICLR 2023): per-layer Hessian
  `H = 2 X Xᵀ` accumulated from calibration activations via forward hooks,
  percdamp damping, Cholesky of `H⁻¹`, column-block quantization with error
  feedback, optional activation ordering (`actorder`). Quantization grid must
  reuse the project's own quantize/dequantize primitives from
  `code/src/quantization.py` so that "3-bit per-channel" means the same thing
  in RTN and GPTQ columns of the table.
- Config knobs (YAML): `gptq: {bits, granularity, skip_modules, num_calib_batches,
  percdamp, actorder, block_size}` — calibration knobs may default, quantization
  knobs may not.
- Calibration data = training split of the *receiver's own* dataset via the
  existing `get_dataset` registry (no new data machinery).

**Acceptance criteria**
- Unit-level: on a random Linear + Gaussian inputs, GPTQ reconstruction error
  ≤ RTN error (strictly lower in the generic case); with `H = I` (or damping
  → ∞) GPTQ output matches plain RTN column-wise rounding. Put these in
  `code/test/`.
- No GPU required for tests (small tensors, CPU).

### WP2 — GPTQ baseline evaluation phase  `[ ]`  (Priority 1, needs WP1)

> Notes from the WP1 executor: (1) `gptq=` fragment — recommend carrying
> `bits/gran/skip` + `ncal`/`percdamp`/`actorder` but NOT `block_size`, which is
> result-invariant (solver batching) and would split identical results across
> paths; coordinator to confirm. (2) Smoke-validated on this box (4090):
> deit3_base/EuroSAT/seed 2038 FP ckpt at 3-bit/channel gives GPTQ(FP)=0.9796
> vs recorded fp_ptq(RTN)=0.8237 and fp=0.9874; 48 layers, ~32 s calib+solve
> per invocation at `num_calib_batches=4`, zero Cholesky retries.

**Deliverables**
- `code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_gptq.py`
  + mirrored YAML. Interface spec: `git show 3516c46^` version of the same
  path (structure only). Loads the FP checkpoint, runs calibration + GPTQ,
  evaluates test accuracy, writes `eval_results.json` under
  `experiment_type=fp_gptq` with the `gptq=` path fragment carrying
  bits/gran/skip + calibration knobs.
- Optional (decide with coordinator): text-family twin
  `evaluate_fp_gptq.py` under the text family.

### WP3 — QV + GPTQ transfer phase (Task 2)  `[ ]`  (Priority 1, needs WP1)

**Deliverables**
- New numbered phase under `code/experiments/vision/ilharco_timm_supervised/`
  (next free number; check the tree) — `qv_transfer_gptq.py` + YAML: build
  `FP_tgt + alpha * QV_src` exactly as `001_qat_transfer/qv_transfer.py` does,
  then apply **GPTQ** instead of `apply_ptq_`, evaluate val+test splits over
  the alpha grid. Path grammar: same as the 001 transfer paths but with a
  `gptq=` fragment in place of / alongside the `ptq=` fragment (keep the
  doubled modality segment; see CLAUDE.md).
- Must include `alpha=0` in the grid (that *is* the GPTQ(FP) baseline on the
  receiver — gives Task 1 and Task 2 numbers from one sweep and guarantees
  identical calibration between the compared cells).

### WP4 — RepQ-ViT  `[ ]`  (Priority 2)

- Vendor the official RepQ-ViT code under `code/src/repqvit/` (it needs
  model-internal access — post-LayerNorm / post-Softmax reparameterization —
  so vendoring beats reimplementation; this is the one exception to
  "from scratch"). Review the vendored code carefully; strip anything unused.
- `evaluate_fp_repqvit.py` + YAML in `000_baselines/`, `experiment_type=fp_repqvit`.
- QV combination (RepQ-ViT on `FP + alpha·QV`) reuses the WP3 script shape.

### WP5 — APHQ-ViT / PTQ4ViT / REx  `[ ]`  (Priority 3, only if time allows)

Same pattern as WP4. Skip unless WP1–WP4 are done and dispatched.

### WP6 — Analysis & tables  `[ ]`  (after runs land)

- Extend/add a `998_rebuttal`-style analysis script comparing the matrix cells
  and a visualization (competitor bar/table). JSON in → JSON out; visualization
  reads the analysis JSON, never recomputes. Follow the
  `<verb>_<noun>_<family>.py` + `*_common.py` split.

## 5. Validation & sanity anchors

- Before trusting any competitor number: GPTQ at 3-bit must beat RTN on at
  least one ViT/dataset pair by a plausible margin (GPTQ paper shows large
  low-bit gains); if it doesn't, suspect the implementation, not the story.
- `_dryrun` smoke runs (limit_num_batches) before real dispatch, never mixed
  with real results.
- Identical quantization primitives across methods (see WP1) so the table's
  columns are comparable.

## 6. Coordination protocol

- One executor session per WP; point it at this file + the WP number
  (e.g. "Read plans/rebuttal_competitor_ptq.md and implement WP1").
- Executor updates this file's checkbox + a one-line note when done, and
  reports back: files created, deviations from spec, open questions.
- The coordinator (manager session) reviews diffs, resolves deviations, and
  owns dispatching GPU runs (multi-rig-dispatch) — executors never run them.
- Journal entries in `journal.md` are written when runs are dispatched/land,
  not when code is merely written.
