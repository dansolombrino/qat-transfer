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

   PV-tuning is still out of scope for *this* plan, but it is no longer
   unaddressed: it is its own line of work under phase `008_pv_transfer`
   (`code/src/pv_tuning.py`, `finetune_pv.py`, `000_baselines/evaluate_pv*.py`).
   It asks whether a stronger *finetuner* yields a better-transferring QV,
   which is orthogonal to this plan's question about stronger *quantizers*, and
   it shares none of their code paths. Answer, on the full 22x22 grid: no —
   cross-task mean PV minus QAT is -0.0020 over n=462, PV better in 43.1% of
   pairs. See journal.md.

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

### WP2 — GPTQ baseline evaluation phase  `[x]`  (Priority 1, needs WP1)

> Code done 2026-08-01 (executor session): `000_baselines/evaluate_fp_gptq.py` +
> mirrored YAML for BOTH the timm vision family and the text family (text twin
> confirmed in-scope by coordinator; it passes a tokenizer-carrying `forward_fn`
> to `apply_gptq_` since text loaders yield raw (texts, labels)). `gptq=`
> fragment per the confirmed rule (bits/gran/skip + ncal/percdamp/actorder, no
> `block_size`); calibration = receiver's own train split; dryrun runs write to
> `fp_gptq_dryrun`. Wave 1 (user-confirmed scope): vit_base_patch16_224.orig_in21k
> x 22 datasets, 3-bit/channel/skip=[head], behemoth GPUs 0/2/4 (WP3's live sweep
> holds 5/6/7 — trees/GPUs disjoint, verified). Wave-1 fp_gptq numbers double as a
> cross-check of 005's alpha=0 self-pair cells.
> **Wave 1 LANDED 2026-08-01**: 22/22, zero failures, mean 45.6 s/run.
> GPTQ(FP) > RTN(FP) on 22/22, mean +0.572; EuroSAT cell matches 005's alpha=0
> bit-for-bit. See journal.md.
> **Text wave LANDED 2026-08-01**: bert-base-uncased x 11 active datasets
> (AmazonPolarity retired — no epochs entry, no RTN twin), 11/11, zero
> failures, mean 38.9 s/run; GPTQ > RTN on 11/11, mean +0.100.
> **2-bit wave LANDED 2026-08-01**: both models, 33/33. GPTQ2 >> RTN2 (which
> is at chance in vision) but collapses vs GPTQ3 (vision means 0.289 vs
> 0.791; text 0.363 vs 0.835) — 2-bit is past one-shot PTQ's reach, i.e. the
> max-headroom regime for QV/QAT (see journal.md). Marked done: the plan's
> "one architecture suffices" bar is met (vit_base full grid + bert-base +
> 2-bit extension), sanity anchor §5 passed 33/33. Optional follow-up, not
> blocking: wider vision model grid (7-model ≈ 1.5-2 h, 12-model ≈ 2.5-3 h
> on behemoth GPUs 0/2/4; runners already bit- and (for the next executor)
> easily model-parameterizable).

> Notes from the WP1 executor: (1) `gptq=` fragment — recommend carrying
> `bits/gran/skip` + `ncal`/`percdamp`/`actorder` but NOT `block_size`, which is
> result-invariant (solver batching) and would split identical results across
> paths; **coordinator confirmed 2026-08-01: exclude `block_size` from the
> fragment; carry bits/gran/skip + ncal/percdamp/actorder. Same rule applies to
> the WP3 `gptq=` fragment.** (2) Smoke-validated on this box (4090):
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

### WP3 — QV + GPTQ transfer phase (Task 2)  `[x]`  (Priority 1, needs WP1)

> Runs landed 2026-08-01: 506/506 cells (484 alpha=1 + 22 alpha=0), behemoth GPUs
> 5/6/7, wall-clock 11:26–16:5x, zero failures. JSONs gathered to the 4090 tree.
> Full-grid Task-2 headline (fp head, cross-task, alpha=1 vs alpha=0): mean
> Delta=-3.2 pts, median -2.2, win rate 9.3% (43/462), best +2.8 — at lambda=1 QV
> does NOT add gain on top of GPTQ; QV+GPTQ nonetheless crushes RTN off-diagonal,
> and GPTQ *hurts* the pure QAT checkpoint on the diagonal (objective mismatch:
> GPTQ reconstructs the FP function, which for a QAT ckpt is the bad one — RTN(QAT)
> remains the proper ceiling). Details + figures: journal.md, two viz scripts under
> code/visualizations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/.

> Code done 2026-08-01 (executor session); smoke-validated on behemoth (pre-quant
> accuracies bit-identical to 001 RTN cells); full 22x22 sweep dispatched 11:26 on
> behemoth GPUs 5/6/7 (tmux qat_005_full_gpu{5,6,7}), see journal.md. Delivered
> `005_qat_transfer_gptq/qv_transfer_gptq.py` + mirrored YAML. Deviations from the
> 001 template, both deliberate: (1) `qv.alphas` is a *list* looped internally
> (alpha=0 runs on the self-pair only — it is the donor-independent GPTQ(FP)
> baseline); (2) `skip_existing: true` resume guard skips cells whose
> eval_results.json exists. Calibration batches are materialized once per receiver
> so every (donor, alpha, head) GPTQ call of that receiver shares bit-identical
> calibration. `gptq=` fragment per the confirmed rule (no `block_size`).
> Sweep scope (user-confirmed): vit_base_patch16_224.orig_in21k, 22x22 grid,
> alphas=[0.0, 1.0], test split, behemoth GPUs 5/6/7.

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
