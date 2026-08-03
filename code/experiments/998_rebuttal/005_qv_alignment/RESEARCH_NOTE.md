# QV alignment validation for reviewer 3HFP

Status: v1.6. The ViT-B/16 Euclidean pilot and the Level-A alpha-response
extension are complete and interpreted. A result-grounded reviewer-response
draft incorporating both experiments is persisted below. Receiver-specific
curvature remains a possible later experiment, not a requirement for reporting
the present H=I result honestly.

This is the canonical, evolving scientific record for rebuttal experiment
`005_qv_alignment`. It should be updated as the methodology is refined and as
results arrive. `journal.md` remains the chronological project log; this note
holds the detailed reasoning, definitions, diagnostics, and reviewer-response
draft that would be too large for a journal entry.

## Revision history

- **v1.6 — 2026-08-03:** Interprets the complete Level-A alpha response and
  adds reviewer-response draft v2. The sweep reveals a family-wise-significant
  scale-dependent alignment profile, but the Euclidean predicted optimum is
  severely miscalibrated and anticorrelated with the empirical optimum, while
  squared cosine does not predict grid-best validation recovery. The resulting
  interpretation is that scale matters and unit scale hides structure, but the
  Euclidean cosine-squared law is not quantitatively validated.
- **v1.5 — 2026-08-02:** Completes real Level-A execution in CPU-only wave
  `20260802-211910`. The 12,331,488-byte golden statistics artifact passes
  source-hash, schema, population, QAP-null, and quartile-count validation.
  All three real figures render as PDF plus 300-dpi PNG and pass visual
  inspection. Scientific interpretation is deliberately deferred.
- **v1.4 — 2026-08-02:** Implements the approved
  `006_alignment_alpha_response` analyzer and three render-only figures. Five
  targeted tests pass, including validation of both real source contracts,
  directional scale/tie logic, deterministic max-stat QAP, quadratic-fit
  behavior, and synthetic PDF/PNG rendering. No golden statistics or real
  figures have been produced; dispatch remains separately gated.
- **v1.3 — 2026-08-02:** Proposes the separately gated
  `006_alignment_alpha_response` Level-A experiment. Audits complete existing
  alpha-curve coverage and precommits the correlation profile, scale-calibration
  diagnostics, QAP multiplicity control, tie handling, artifacts, run identity,
  and CPU-only execution policy. No implementation or execution yet.
- **v1.2 — 2026-08-02:** Completes Step 6. Distinguishes the unit-scale gain
  equation from the optimized-scale cosine-squared law, interprets all four
  predeclared comparisons and influence diagnostics, records figure-readiness
  issues, and adds a reviewer-response draft that leads with the primary null
  result.
- **v1.1 — 2026-08-02:** Completes Step 5. Records the successful 22-task
  geometry artifact, analyzer retry, raw predeclared headline statistics, and
  six verified real figure files. Interpretation is deliberately deferred to
  Step 6.
- **v1.0 — 2026-08-02:** Completes Step 4 verification without checkpoint
  access: 10 targeted tests pass, including actual ViT module discovery,
  existing outcome-source validation, synthetic QAP/influence checks, and
  temporary PDF/PNG rendering. Records and fixes two floating-point boundary
  issues found by the first test pass.
- **v0.9 — 2026-08-02:** Implements Step 3 without execution: Hydra-configured
  geometry and analysis producers plus the three render-only visualization
  scripts. Verification and all artifact generation remain separately gated.
- **v0.8 — 2026-08-02:** Completes the Step 2 contract audit and clarifies that
  the first implementation produces only the reviewer-literal Euclidean
  association; projected-QV transfer remains a separately gated later step.
- **v0.7 — 2026-08-02:** Locks the fixed 22-task alphabetical order, canonical
  final epoch for every task, and exact FP/QAT checkpoint templates. The task
  set remains an invariant rather than a config or run-ID key.
- **v0.6 — 2026-08-02:** Locks the reviewer-literal Euclidean measurement to
  one global cosine over the selected quantized-Linear weight subspace, with
  no layer-wise or channel-wise cosine averaging, and approves the three
  Euclidean pilot figures in PDF and PNG.
- **v0.5 — 2026-08-02:** Disables Weights & Biases for both deterministic
  Euclidean stages; golden JSON artifacts and atomic status records are the
  complete logging contract.
- **v0.4 — 2026-08-02:** Approves the Euclidean pilot's input-checkpoint
  provenance, atomic golden-artifact completion rule, and deliberate absence
  of partial runtime checkpointing or resume.
- **v0.3 — 2026-08-02:** Persists the approved modern `005` contract,
  producer/analyzer boundary, ViT-B/16 pilot, ordered producer and analyzer run
  identities, the complete meaning of `reviewer_3hfp_v1`, and the durable
  location and schema of the joined per-pair points.
- **v0.2 — 2026-08-02:** Records approval of the initial framing and introduces
  mandatory step boundaries. Every step ends with a user review; authorization
  for one step never implies authorization for the next.
- **v0.1 — 2026-08-02:** Initial framing. Defines the quantized-Linear QV
  subspace, separates the reviewer-literal association from a subspace-consistent
  theoretical test, specifies Euclidean, STE-Hessian, and generalized
  Gauss--Newton analyses, and precommits the result-dependent interpretations.

## Working protocol and approval gates

This experiment is deliberately incremental. At the end of every step, work
stops so the user can inspect, store, commit, annotate, or revise the result.
The next step begins only after an explicit greenlight. A greenlight is scoped
to the named next step and does not carry forward.

1. **Persist and approve the scientific framing.** Maintain this note as the
   canonical rationale and record the staged workflow. No code or execution.
2. **Elect the experiment contract.** Decide the run identity and parameter
   order, producer/aggregator boundaries, artifact schemas and golden completion
   signals, architecture rollout, checkpoint/resume policy, and W&B policy. No
   experiment implementation or execution.
3. **Implement the Euclidean pilot.** Write only the ViT-B/16 quantized-Linear
   QV producer, statistical aggregator, and visualization code agreed in Step 2.
   Do not run the real 22×22 experiment.
4. **Verify the Euclidean implementation.** Run code-level tests and synthetic
   or minimal numerical checks agreed in Step 2. Review key selection,
   precision, path provenance, joins, diagonal exclusion, and permutation logic.
   Do not launch the real pilot unless separately approved.
5. **Run the ViT-B/16 Euclidean pilot.** Produce the approved alignment,
   statistics, and figures only. Stop before extending the analysis or changing
   the reviewer-response language.
6. **Interpret and document the pilot.** Inspect results with the user, update
   this note, fill only the supported parts of the reviewer response, and decide
   whether the methodology needs revision.
7. **Design the projected-QV transfer phase.** Resolve the subspace/outcome
   consistency control in operational detail. Implementation and execution are
   separate later gates.
8. **Design the curvature phase.** Resolve the data split and sample count,
   STE objective, evaluation point, stationarity and PSD criteria, GGN
   construction, loss-curve protocol, and compute budget. Implementation and
   execution are separate later gates.
9. **Consider architecture replication.** Extend beyond the ViT-B/16 headline
   only after the preceding results and costs have been reviewed.

Current stopping point: Steps 1--4 are complete. Step 4 verified the
implementation without reading real checkpoints or creating experiment
artifacts. Step 5, the first real ViT-B/16 Euclidean pilot run, remains
unauthorized until its explicit user greenlight.

### Step 4 verification record

Command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  code/test/qv_alignment_rebuttal.py
```

Final result: `10 passed in 11.05s`.

The targeted suite verifies:

- Both Hydra configs satisfy their frozen contracts and preserve the elected
  producer/analyzer run-ID order.
- The module selector returns exactly the same Linear names as `apply_ptq_` on
  a synthetic nested model while excluding a skipped `head` subtree.
- Architecture discovery with `pretrained=False` selects 48 ViT-B/16 Linear
  weight matrices, excludes the classification head, and covers 84,934,656
  scalar coordinates. For 22 float32 QVs, the real producer's temporary map is
  7,474,249,728 bytes (approximately 6.96 GiB).
- The locked FP and QAT checkpoint templates resolve exactly, including the
  dataset-final epoch and the QAT quantization tag, using only a synthetic
  checkpoint root.
- Float64 blockwise Gram accumulation agrees to numerical precision with one
  explicitly concatenated global vector; no layer or channel averaging enters.
- Tie-aware ranks, constant-vector handling, simultaneous-axis QAP reindexing,
  full-null persistence, Monte Carlo exceedance counting, and all influence
  population sizes match the predeclared design.
- The existing outcome summary validates as exactly 484 unique, complete
  ViT-B/16 donor--receiver records with matching metadata.
- All three visualization scripts consume a synthetic golden statistics JSON
  and successfully render six nonempty temporary artifacts (PDF and PNG for
  each script). The temporary directory is removed after the test.

The first pass produced `7 passed, 2 failed`. Both failures exposed numerical
boundaries and were fixed before the clean run:

1. A mathematically perfect correlation evaluated as `1.0000000000000002`.
   Correlation and QAP coefficients are now clipped only at floating-point
   overflow beyond their mathematically valid `[-1, 1]` range.
2. The QAP unit test recomputed its exceedance expectation through a separately
   rounded correlation path. It now applies the declared formula directly to
   the exact persisted null values; the independently computed null vector is
   still checked elementwise against explicit simultaneous-axis reindexing.

No test resolved `CHECKPOINT_BASE_PATH`, loaded a real checkpoint, invoked a
producer `main`, wrote beneath `evaluations/998_rebuttal/005_qv_alignment`, or
wrote beneath `plots/998_rebuttal/005_qv_alignment`.

### Step 3 implementation inventory

The following source files now implement the approved contract:

- `code/experiments/998_rebuttal/005_qv_alignment/compute_euclidean_alignment.py`
  plus its mirrored Hydra config. It discovers the `apply_ptq_` Linear-weight
  set without pretrained downloads, writes each float32 checkpoint QV into
  temporary memory-mapped storage, and accumulates the single global Gram
  matrix in fixed blocks and float64. The temporary map is never a completion
  artifact and is removed after use.
- `code/experiments/998_rebuttal/005_qv_alignment/analyze_euclidean_alignment.py`
  plus its mirrored Hydra config. It hard-validates the 484-cell join, persists
  the 462 cross-task and 22 diagonal records, computes every predeclared
  Spearman/Pearson result, uses one reproducible shared set of 10,000 QAP task
  permutations, and writes the full null vectors and influence summaries.
- `visualizations/998_rebuttal/005_qv_alignment/plot_alignment_heatmap.py`,
  `plot_alignment_associations.py`, and `plot_alignment_influence.py`, with a
  shared validation/render helper. They accept only the golden statistics JSON
  and emit the approved PDF and PNG artifacts beneath mirrored run identities.

Both producers refuse to overwrite an existing golden artifact. No runtime
resume state, model checkpoint, W&B integration, dataset loading, or inference
was introduced. This inventory records implementation presence, not
correctness: syntax, unit, synthetic, and contract verification belong to the
separately gated Step 4.

## Step 2 experiment contract (approved portion)

### Infrastructure and pilot boundary

`005_qv_alignment` uses the modern research contract without migrating rebuttal
phases `001`--`004`:

- Hydra-configured artifact producers with mirrored config files.
- `code/common/run_id.py` for percent-encoded run paths and full-config
  collision guards.
- `code/common/status.py` for atomic lifecycle and progress reporting.
- Visualization code in the plugin-standard top-level `visualizations/` tree;
  visualization scripts read `evaluations/` only.
- ViT-B/16 (`vit_base_patch16_224.orig_in21k`) is the sole first pilot. Other
  backbones require a later greenlight.

The Euclidean phase has two artifact-producing stages:

1. `compute_euclidean_alignment` reads the matched FP/QAT checkpoints and emits
   reusable QV geometry without reading any transfer outcome.
2. `analyze_euclidean_alignment` reads that geometry and the existing
   `win_loss_ilharco_timm_supervised.json`, performs the exact pair join and
   statistical analysis, and persists the points used by visualization.

This boundary prevents checkpoint processing from being repeated when an
outcome definition, inference method, or plot changes.

### Checkpoint provenance, completion, and resume contract

The word *checkpoint* has two distinct meanings in this experiment.

The producer's immutable model inputs are the canonical final FP and QAT
classifier checkpoints for each of the 22 tasks. The FP/QAT pair defines each
task's QV before projection to the selected quantized-`nn.Linear` weight
subspace. For every input checkpoint, `euclidean_alignment.json` records the
resolved path and SHA-256 digest. A changed checkpoint is therefore detectable
even if its filename is unchanged.

The Euclidean pilot deliberately has no intermediate runtime checkpoints and
no partial-resume protocol:

- The producer computes all 22 projected vectors and the complete 22-by-22
  geometry, validates it, and atomically publishes `euclidean_alignment.json`.
  An interrupted or failed producer restarts from the beginning.
- The analyzer validates both immutable source artifacts, computes the complete
  join and analysis, and atomically publishes `euclidean_statistics.json`. An
  interrupted or failed analyzer also restarts from the beginning.
- A run is complete only when its golden JSON artifact exists and passes its
  schema and invariant checks. Temporary, partial, or status files are never
  accepted as scientific output.
- Neither stage creates a model checkpoint, and there is no runtime-checkpoint
  retention policy because no runtime checkpoints exist.

This contract applies to the single-model Euclidean pilot. If later curvature
or multi-architecture work makes restart cost material, its resume contract
must be elected separately rather than silently added to this run identity.

### Experiment logging

Weights & Biases is disabled for both `compute_euclidean_alignment` and
`analyze_euclidean_alignment` (`use_wandb=false`). These are deterministic
offline artifact producers rather than iterative training jobs: there are no
epochs, learning curves, optimizer states, or gradients to monitor. The golden
JSON artifacts carry the scientific provenance and results, while atomic
status records carry lifecycle, elapsed time, progress, completion, and error
information. A later training-like or curvature experiment must elect its own
logging contract rather than inheriting this decision implicitly.

### Euclidean aggregation rule

The reviewer-literal measurement is one global Euclidean cosine over the
direct sum of all selected quantized-`nn.Linear.weight` coordinates. For
selected layers `ell`, the producer computes

\[
c_{\mathrm{global}}(D,R)=
\frac{\sum_\ell\langle\rho_{D,\ell},\rho_{R,\ell}\rangle_F}
{\sqrt{\sum_\ell\|\rho_{D,\ell}\|_F^2}
 \sqrt{\sum_\ell\|\rho_{R,\ell}\|_F^2}}.
\]

This is algebraically identical to flattening each selected weight tensor,
concatenating the tensors in the same canonical order, and taking one cosine.
The implementation need not materialize the concatenated vectors: it sums
per-layer dot products and squared norms in `float64` and normalizes once.

This is not a flattening of the entire model. Parameters outside the locked
quantized-Linear weight subspace remain excluded. It is also not an average of
layer cosines or output-row/channel cosines. Such averages assign equal votes
to architecture- or quantizer-defined groups regardless of their QV energy
and therefore define a different statistic from the requested `H=I` cosine.
No layer-averaged, channel-averaged, or quantization-step-normalized alignment
is part of `reviewer_3hfp_v1`.

Per-output-row quantization still enters the study causally: it is the
quantizer that produced the QAT displacement and it determines the affected
Linear weight subspace. It does not change how the identity metric aggregates
the selected scalar coordinates.

### Euclidean visualization contract

Three visualization scripts read `euclidean_statistics.json` only; they never
reload checkpoints, reconstruct the pair join, or recompute statistics:

1. `visualizations/998_rebuttal/005_qv_alignment/plot_alignment_heatmap.py`
   renders the 22-by-22 signed global-cosine matrix. The diagonal is visibly
   identified as algebraically one but excluded from the off-diagonal
   color-range determination.
2. `visualizations/998_rebuttal/005_qv_alignment/plot_alignment_associations.py`
   renders a two-panel cross-task association figure: signed global cosine
   versus unit-scale test `delta`, and squared global cosine versus
   `recovery_best`. Both panels contain exactly 462 neutral-colored points and
   report their persisted Spearman, Pearson, and QAP results. They report no
   IID correlation p-value.
3. `visualizations/998_rebuttal/005_qv_alignment/plot_alignment_influence.py`
   renders the 22 leave-one-receiver-out and 22 leave-one-donor-out
   coefficients against the full-matrix reference coefficient.

Every figure is emitted as vector PDF for the paper and 300-DPI PNG for quick
inspection and rebuttal-system compatibility. HTML is not a required pilot
artifact. Plot output paths mirror the producer and analyzer identities under
`plots/998_rebuttal/005_qv_alignment/<script_stem>/` so figures remain
traceable to their exact source artifact.

### Geometry-producer run identity

The ordered `RUN_ID_PARAMS` for `compute_euclidean_alignment` are:

1. `family`
2. `model_name`
3. `seed`
4. `optim`
5. `lr`
6. `wd`
7. `ls`
8. `wl`
9. `max_grad_norm`
10. `batch_size`
11. `qat_bits`
12. `qat_granularity`
13. `qat_skip_modules`
14. `ptq_skip_modules`
15. `checkpoint_kind`
16. `epoch_policy`
17. `vector_scope`
18. `module_selector`
19. `accumulation_dtype`

The ordered 22-task set is a fixed invariant of `005_qv_alignment`, not a
config key and not a run-ID component. Every geometry artifact records the
ordered list verbatim. Changing the set requires an explicit run-ID re-election
or a new sub-experiment; it may not be changed in place.

The canonical order follows the existing rebuttal outcome artifact and is
alphabetical. The corresponding canonical final epochs are locked alongside
it:

| index | task | final epoch |
|---:|---|---:|
| 0 | CIFAR10 | 6 |
| 1 | CIFAR100 | 6 |
| 2 | Cars | 35 |
| 3 | DTD | 76 |
| 4 | EMNIST | 2 |
| 5 | EuroSAT | 12 |
| 6 | FER2013 | 10 |
| 7 | FashionMNIST | 5 |
| 8 | Flowers102 | 147 |
| 9 | Food101 | 4 |
| 10 | GTSRB | 11 |
| 11 | ImageNet | 1 |
| 12 | KMNIST | 5 |
| 13 | MNIST | 5 |
| 14 | OxfordIIITPet | 82 |
| 15 | PCAM | 1 |
| 16 | RESISC45 | 15 |
| 17 | RenderedSST2 | 39 |
| 18 | STL10 | 60 |
| 19 | SUN397 | 14 |
| 20 | SVHN | 4 |
| 21 | TinyImageNet | 4 |

The implementation may import `DATASET_NAME_TO_EPOCHS`, but it must compare
the resolved mapping to this locked table and fail on any difference. It never
silently adopts a changed task set, order, or epoch policy.

With `<root> = CHECKPOINT_BASE_PATH`, `<model> =
vit_base_patch16_224_orig_in21k`, and `<epoch>` from the table, immutable model
inputs resolve as:

```text
FP:
<root>/vision/ilharco_timm_supervised/fp/<model>/<task>/
  optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/
  seed=2038/classifier_epoch_<epoch>.pt

QAT:
<root>/vision/ilharco_timm_supervised/qat/<model>/<task>/
  optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/
  qat=bits=3_gran=channel_skip=head/
  seed=2038/classifier_epoch_<epoch>.pt
```

Before loading any tensor, the producer requires all 44 paths to exist and
reports every missing path together in one failure. The resolved absolute path,
file size, and SHA-256 digest of every checkpoint are persisted in the golden
artifact.

The producer's golden artifact is
`euclidean_alignment.json`. It contains the task order, selected parameter
names and shapes, resolved checkpoint provenance, per-QV norms, the dot-product
matrix, signed cosine matrix, and numerical diagnostics.

### Analyzer run identity

The analyzer lives below its producer run directory and therefore inherits the
complete geometry identity. Its ordered `RUN_ID_PARAMS` are:

1. `ptq_bits`
2. `ptq_granularity`
3. `outcome_protocol`
4. `outcome_split`
5. `unit_alpha`
6. `analysis_spec`
7. `n_permutations`
8. `permutation_seed`

For the pilot these resolve to `3`, `channel`, `full_qv`, `test`, `1.0`,
`reviewer_3hfp_v1`, `10000`, and `2038`, respectively.

The analyzer's golden artifact is `euclidean_statistics.json`.

### Normative definition of `reviewer_3hfp_v1`

This section is the authoritative meaning of the version token. Code may refer
to `reviewer_3hfp_v1`, but it must implement the full contract below; the token
is never an excuse to leave methodology implicit. Any change to these rules
requires a new analysis-spec version and a run-ID re-election.

#### Population and join

- Require exactly the fixed 22 tasks in both the alignment artifact and the
  outcome source.
- Require all 484 ordered donor--receiver cells. Missing, duplicated, or extra
  cells are a hard error; the analyzer never correlates over an intersection or
  silently drops incomplete data.
- Retain the 22 same-task cells for audit and heatmap display, but exclude them
  from every inferential correlation. The inferential population is exactly
  the 462 ordered cross-task cells.
- Join only by the exact `(donor, receiver)` task-name pair.
- Verify that the source metadata match the analyzer config: family, training
  seed, QAT configuration, PTQ configuration, split, and unit alpha.

#### Predeclared comparisons

There is one reviewer-literal primary comparison:

1. Signed Euclidean cosine versus unit-scale test-accuracy `delta` from the
   full-QV transfer matrix. Spearman is the primary coefficient; Pearson is the
   secondary functional-form coefficient.

There is one theory-adjacent secondary comparison:

2. Squared Euclidean cosine versus `recovery_best`, the test-set recovery ratio
   at the receiver-validation-selected positive lambda. This is explicitly an
   approximation to Proposition 1 because the metric is Euclidean rather than
   Hessian-weighted, the outcome is Top-1 accuracy rather than loss, and the
   practical lambda is selected from a bounded positive grid.

Two additional pairings are persisted as descriptive diagnostics, never
promoted to the headline after looking at their values:

- Signed cosine versus `delta_best`.
- Squared cosine versus unit-scale `recovery`.

All four pairings receive Spearman and Pearson coefficients. Only the first is
the primary inferential claim. The secondary and diagnostic results must be
identified as such in tables, figures, and reviewer text.

#### QAP permutation test

- Let `A[D,R]` be the alignment matrix and `Y[D,R]` the fixed outcome matrix.
- For each permutation, draw one uniform permutation `pi` of the 22 task
  labels and reindex both axes of alignment together, producing
  `A_pi[D,R] = A[pi(D), pi(R)]`. Keep `Y` fixed.
- Remove the same 22 diagonal positions and recompute the coefficient over the
  462 remaining cells.
- Use 10,000 independently drawn permutations from NumPy's documented seeded
  generator with seed 2038.
- Report the two-sided Monte Carlo p-value
  `(1 + count(abs(stat_perm) >= abs(stat_observed))) / (10000 + 1)`.
- Persist the observed statistic, exceedance count, p-value, permutation count,
  seed, generator identifier, and summary quantiles of the null distribution.
  The full 10,000-value null vector is also persisted so later plots cannot
  silently regenerate a different test.
- Apply the QAP procedure separately to Spearman and Pearson for every
  predeclared pairing. The reviewer-literal signed-cosine/unit-delta Spearman
  test remains the sole primary p-value; all others are secondary or
  descriptive and are labeled accordingly.

#### Influence and stratified summaries

- Leave-one-receiver-out: for each of 22 receivers, remove its 21 incoming
  cross-task cells and recompute Spearman and Pearson for every pairing.
- Leave-one-donor-out: for each of 22 donors, remove its 21 outgoing cross-task
  cells and recompute the same coefficients.
- Per receiver: correlate its 21 donor alignments with its 21 outcomes.
- Per donor: correlate its 21 receiver alignments with its 21 outcomes.
- A coefficient that is undefined because either vector is constant is stored
  as `null` with an explicit reason; it is never converted to zero.
- No IID Fisher-z interval or IID correlation p-value is reported. The matrix
  cells share donors and receivers, so those calculations would claim more
  independent evidence than exists.

#### Durable point persistence

The producer artifact lives at

```text
evaluations/998_rebuttal/005_qv_alignment/euclidean_alignment/
  <producer_run_id_path>/euclidean_alignment.json
```

The analyzer artifact, including the exact plot points, lives at

```text
evaluations/998_rebuttal/005_qv_alignment/euclidean_alignment/
  <producer_run_id_path>/analysis/<analyzer_run_id_path>/
  euclidean_statistics.json
```

`euclidean_statistics.json` contains at minimum:

- `provenance`: resolved config snapshots, source paths, SHA-256 hashes of both
  source JSONs, task order, and analysis-spec version.
- `points`: exactly 462 ordered cross-task records. Each record contains donor,
  receiver, signed cosine, squared cosine, `delta`, `delta_best`, `recovery`,
  `recovery_best`, unit alpha, best alpha, receiver baseline accuracy, and
  receiver ceiling delta.
- `diagonal_audit`: exactly 22 same-task records, never included in inference.
- `statistics`: observed Spearman/Pearson coefficients, QAP results and null
  vectors, leave-one-out results, and per-donor/per-receiver summaries.
- `missing`: an empty list in every successful v1 artifact. Any missing pair is
  a hard failure before the golden artifact is written.

Visualization code reads these persisted records and statistics verbatim. It
does not reload checkpoints, rejoin the outcome matrix, recompute a
correlation, or rerun a permutation test.

## Step 5 execution record (complete; interpretation pending)

The first real wave, `20260802-162236`, computed all 22 QVs and the complete
Euclidean alignment matrix. The producer selected 48 quantized
`nn.Linear.weight` matrices (84,934,656 scalar coordinates per QV), completed
in 68.1 seconds, and wrote a nonempty golden `euclidean_alignment.json`.

The analyzer invocation in that wave failed before reading data because the
generated Hydra override did not retain Hydra-level quotes around an
`outcome_path` containing embedded `=` delimiters. This was an invocation bug,
not a scientific or numerical failure. The shared `hydra_override_arg` helper
now JSON-quotes string values; a regression test was added, and the combined
targeted suite passes 18/18 tests. The failed wave script remains unchanged as
the historical execution record.

Analyzer-only retry wave `20260802-190820` reused the successful geometry
artifact. It completed all four predeclared comparisons in 2.5 seconds and
wrote a valid 3,220,749-byte `euclidean_statistics.json`. The two headline
results available at the Step 5 boundary are:

- Primary, signed Euclidean cosine versus unit-scale `delta`: Spearman
  `rho = 0.01496`, QAP `p = 0.86051`; Pearson `r = -0.02039`, QAP
  `p = 0.77162`.
- Theory-adjacent secondary, squared Euclidean cosine versus `recovery_best`:
  Spearman `rho = 0.15841`, QAP `p = 0.02930`; Pearson `r = 0.00919`, QAP
  `p = 0.81162`.

These values are recorded here as raw outputs, not yet interpreted. In
particular, the discrepancy between the rank and linear statistics for the
theory-adjacent comparison must be examined through the persisted points,
influence diagnostics, and plots during Step 6 before any reviewer-facing
claim is drafted.

All three real visualizations were rendered and verified as nonempty PDF and
300-dpi PNG files: alignment heatmap, two-panel association figure, and
donor/receiver influence figure. One initial association-render command used
an incomplete input path and failed before reading data; its log was retained,
and the corrected invocation succeeded. Matplotlib's fallback to a temporary
cache directory was harmless.

Step 5 stops here. No curvature experiment, projected-QV transfer run,
cross-architecture expansion, or reviewer-response interpretation is
authorized by these artifacts alone.

## Step 6 interpretation and reviewer-facing reasoning

### 6.1 Direct answer to the reviewer's empirical question

For the reviewer-literal comparison, the answer is **no**: global Euclidean
cosine over the quantized-Linear QV coordinates does not correlate with
unit-scale transfer gain across the 462 cross-task cells. The primary
Spearman coefficient is 0.01496 with task-label QAP p = 0.86051; the secondary
linear summary is Pearson r = -0.02039 with QAP p = 0.77162. Both effect sizes
are essentially zero, the scatter has no visible trend, and the conclusion is
stable to omitting any one receiver or donor.

This is a meaningful negative result: the raw global Euclidean angle is not a
useful predictor of the observed unit-scale 22-by-22 transfer matrix for this
ViT-B/16 pilot.

### 6.2 Why unit-scale gain is not the cosine-squared law

The reviewer's suggested correlation is informative, but it is not itself the
exact prediction of Proposition 1. Under the proposition's quadratic model,

```text
g_R(delta) = g_R(rho_R) + 0.5 ||delta - rho_R||^2_{H_R}.
```

The gain from applying a donor at an arbitrary scale lambda is therefore

```text
g_R(0) - g_R(lambda rho_D)
  = lambda <rho_D, rho_R>_{H_R}
    - 0.5 lambda^2 ||rho_D||^2_{H_R}.
```

At fixed lambda = 1, the gain depends on the inner product, the donor norm,
and whether unit scale overshoots; cosine alone cannot determine it. Only
after optimizing over an unconstrained real lambda,

```text
lambda* = <rho_D, rho_R>_{H_R} / ||rho_D||^2_{H_R},
```

does the recovered fraction reduce exactly to
`cos^2_{H_R}(rho_D, rho_R)`. Consequently, the primary null result does not
formally falsify Proposition 1. It does show that Euclidean angle alone does
not explain the practical unit-scale transfer matrix.

### 6.3 The H=I proxy has a structural limitation

Euclidean cosine is symmetric:
`cos_I(rho_D, rho_R) = cos_I(rho_R, rho_D)`. Transfer is directional, and the
two cells `Delta(D,R)` and `Delta(R,D)` can be very different. For example,
the largest off-diagonal cosine is the symmetric CIFAR10--STL10 value 0.23272,
whereas the unit-scale gains are +0.24310 for STL10 to CIFAR10 and -0.07700
for CIFAR10 to STL10.

The metric in Proposition 1 is receiver-specific. In general,
`cos_{H_R}(rho_D, rho_R)` need not equal
`cos_{H_D}(rho_R, rho_D)`, so the proper curvature-weighted quantity can encode
the directionality that H=I necessarily discards. This makes Euclidean cosine
a deliberately coarse proxy rather than a substitute for the proposition's
metric.

### 6.4 Observed Euclidean geometry

The off-diagonal cosines are highly concentrated near zero:

- mean 0.02246 and median 0.01814;
- 34/462 (7.4%) are negative;
- 52/462 have absolute cosine below 0.01;
- only 22/462 exceed 0.05;
- the maximum is 0.23272 for the CIFAR10--STL10 pair.

Squaring compresses the bulk further: the median cosine-squared value is
0.000329, the 95th percentile is 0.002485, and the maximum is 0.05416. Thus
most matched QVs are nearly orthogonal in the global Euclidean geometry, with
a small number of semantically plausible high-alignment pairs such as
CIFAR10--STL10, MNIST--KMNIST, and KMNIST--FashionMNIST.

### 6.5 Complete predeclared association results

| comparison | role | Spearman rho | QAP p | Pearson r | QAP p |
|---|---|---:|---:|---:|---:|
| signed cosine vs. unit-scale `delta` | primary | 0.01496 | 0.86051 | -0.02039 | 0.77162 |
| squared cosine vs. tuned `recovery_best` | theory-adjacent secondary | 0.15841 | 0.02930 | 0.00919 | 0.81162 |
| signed cosine vs. tuned `delta_best` | descriptive diagnostic | 0.25164 | 0.01000 | 0.12274 | 0.09519 |
| squared cosine vs. unit-scale `recovery` | descriptive diagnostic | -0.00523 | 0.95280 | -0.03819 | 0.32327 |

The QAP p-values come from 10,000 simultaneous donor/receiver task-label
permutations. They are used instead of IID correlation tests because cells
share donors and receivers and because the same symmetric cosine appears in
both directed cells.

### 6.6 What the tuned-scale results do and do not show

The theory-adjacent tuned comparison has a weak positive rank association:
larger squared Euclidean alignment tends to correspond to a better ordinal
position in validation-selected test recovery. Its Spearman result is not
driven by a single task: all 22 leave-one-receiver-out coefficients remain
positive (0.12253--0.19377), as do all 22 leave-one-donor-out coefficients
(0.05569--0.18701).

However, this is not evidence for a quantitative cosine-squared law:

- the effect is small (rho = 0.158);
- Pearson correlation is essentially zero (r = 0.009) and remains near zero
  under every leave-one-task recomputation;
- the relationship is heterogeneous within tasks: only 12/22 per-receiver
  and 12/22 per-donor Spearman coefficients are positive;
- 21/462 tuned accuracy-recovery ratios exceed one (maximum 4.059), whereas
  Proposition 1's exact best-donor loss-recovery fraction is bounded by one;
  this directly demonstrates that the accuracy ratio is a proxy rather than
  the proposition's mathematical quantity;
- `recovery_best` has large values at small cosine-squared values, including
  4.059 for ImageNet to TinyImageNet and 2.429 for Food101 to RenderedSST2;
- the empirical outcome is validation-grid-selected test Top-1 accuracy,
  whereas the proposition concerns an unconstrained optimum of a smooth local
  loss.

The tuned `delta_best` diagnostic is somewhat stronger in rank
(rho = 0.25164, QAP p = 0.01000), and its leave-one-task coefficients remain
positive. This supports the qualitative statement that direction becomes more
informative after scale selection. It remains a predeclared diagnostic, not a
replacement primary endpoint, and its Pearson/QAP result does not meet the
same evidential threshold (r = 0.12274, p = 0.09519).

The defensible summary is therefore: **scale tuning reveals a modest ordinal
alignment signal, but the Euclidean experiment neither predicts unit-scale
transfer nor validates the quantitative cosine-squared recovery law.**

### 6.7 Why the proposition must be presented as an organizing idealization

The exact statement requires all of the following:

1. receiver-specific positive-definite `H_R` geometry;
2. a locally quadratic, smooth receiver loss around `rho_R`;
3. an unconstrained best real-valued scale;
4. gain measured in that same loss;
5. donor and receiver vectors and the outcome defined in the same parameter
   subspace.

The pilot instead uses H=I, finite QAT displacements, hard 3-bit PTQ and
discrete Top-1 accuracy, a bounded positive validation grid from 0.15 to 1.5,
and the existing full-QV transfer outcome against geometry projected onto the
quantized `nn.Linear.weight` subspace. Each mismatch is scientifically
relevant. The theory can motivate what quantities and failure modes to examine,
but it should not be described as an empirically established predictive law
for the 22-by-22 accuracy matrix.

### 6.8 Reviewer-response draft v1

> We thank the reviewer for requesting this direct validation. For the
> ViT-B/16 22-by-22 vision matrix, we computed the Euclidean cosine between
> every matched donor and receiver QV over the 48 `nn.Linear.weight` tensors
> acted on by our 3-bit per-channel quantizer (84,934,656 coordinates), using
> one global inner product rather than averaging layer-wise cosines. We retain
> the 22 same-task cells in the heatmap as an audit, but exclude them from
> association statistics because their cosine is identically one and they are
> not cross-task transfer.
>
> Across the 462 cross-task cells, signed Euclidean cosine does not predict the
> observed unit-scale test-accuracy gain: Spearman rho = 0.015 (task-label QAP
> p = 0.861) and Pearson r = -0.020 (QAP p = 0.772). Thus the answer for the
> reviewer's H=I proxy is negative: raw Euclidean QV angle is not an empirical
> predictor of the unit-scale transfer matrix.
>
> We also evaluated a predeclared theory-adjacent comparison after selecting
> the patch scale on receiver validation data. Squared Euclidean cosine has a
> weak positive rank association with normalized test recovery (Spearman
> rho = 0.158, QAP p = 0.029), but no linear association (Pearson r = 0.009,
> QAP p = 0.812). A related descriptive diagnostic, signed cosine versus the
> validation-selected test gain, gives Spearman rho = 0.252 (QAP p = 0.010).
> We interpret these tuned-scale results as modest ordinal evidence that scale
> and direction interact, not as validation of the quantitative cosine-squared
> law.
>
> Proposition 1 is exact for the best unconstrained scale under a smooth local
> quadratic receiver loss in the receiver-specific `H_R` metric. The present
> proxy instead uses H=I, a bounded positive scale grid, finite QAT
> displacements, hard PTQ/Top-1 outcomes, and a quantized-Linear projection of
> the QVs. Moreover, H=I cosine is symmetric while transfer is directional;
> the receiver-specific `H_R` metric need not be symmetric under donor/receiver
> exchange. We therefore revise the theoretical account to describe
> Proposition 1 as a local organizing model that motivates alignment and scale,
> rather than as an empirically validated predictor of the full transfer
> matrix. A direct quantitative test would additionally require a
> receiver-specific curvature metric and a subspace-matched loss outcome.

### 6.9 Figure-readiness audit

The current figures are numerically correct but should not yet be treated as
final reviewer figures:

- the heatmap color range is dominated by the 0.23272 CIFAR10--STL10 value,
  making most off-diagonal structure visually faint;
- the association panels show all points honestly, but the same extreme
  cosine pair compresses the x-axis bulk, while large recovery ratios at small
  cosine-squared values dominate the y-axis;
- the influence plot uses the fixed interval [-1,1] even though all primary
  leave-one-out coefficients lie between -0.05379 and 0.06526, so the
  robustness pattern is unnecessarily compressed.

A reviewer-ready refinement should preserve the full-range view while adding
a bulk zoom or inset, and should tighten the influence x-axis around the
observed range. This is a visualization-only next step and must not recompute
statistics or change the golden JSON.

### 6.10 Step-6 decision

The current pilot is sufficient to answer the literal H=I question honestly.
It is not sufficient to claim direct validation of Proposition 1. Before a
stronger theoretical claim, the next scientific phase should pair
receiver-specific GGN/Hessian-weighted geometry with a smooth loss endpoint
and should make the measured vector subspace match the applied transfer
subspace. Those are new experiments, not reinterpretations of this pilot.

## Proposed Step 7: Level-A alpha-response analysis

### 7.1 Name and authorization boundary

The proposed next experiment is
`998_rebuttal/006_alignment_alpha_response`. It is a new read-only aggregation
experiment rather than a change to `005_qv_alignment` or
`003_lambda_sensitivity`.

This section is a design proposal only. Approval of the scientific motivation
does not authorize implementation or execution until the user explicitly
approves the name, run-ID election, endpoints, artifacts, and placement below.

### 7.2 Existing-data audit

No model forward pass is required for Level A. Two existing golden sources are
complete for ViT-B/16:

1. `005_qv_alignment/euclidean_alignment.json` provides the exact 22-task
   order, global Euclidean cosine matrix, dot-product matrix, and QV norms over
   the 48 quantized `nn.Linear.weight` tensors.
2. `003_lambda_sensitivity/.../split=val/baseline=fp_ptq/grid=shared/`
   `lambda_curves_ilharco_timm_supervised.json` provides all 484 directed
   donor--receiver validation curves with no missing cells. Each curve has the
   11 measured positive scales
   `{0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00, 1.05, 1.20, 1.35, 1.50}` and
   uses the exact unpatched `alpha=0, Delta=0` anchor against receiver FP+PTQ.

The raw tree contains exactly 5,324 corresponding validation evaluations
(484 cells times 11 measured scales). The new analyzer should consume the
already-aggregated `003` JSON and hash both sources, rather than rescan those
5,324 files.

### 7.3 Fixed scope and invariants

- Model: `vit_base_patch16_224.orig_in21k` only.
- Tasks: the same fixed 22-task alphabetical invariant as `005`; it is not a
  config or run-ID key.
- Seed and training/QAT contract: 2038, AdamW, the canonical final epochs, and
  the existing 3-bit per-channel QAT/QV protocol.
- PTQ: 3-bit per-channel, head skipped.
- Geometry: `reviewer_3hfp_v1`, one global H=I inner product over the
  quantized-Linear weight subspace.
- Curves: validation split, `fp_ptq` baseline, shared 11-point positive grid,
  plus the exact no-patch anchor at alpha zero.
- Unit scale: 1.0.
- Analysis version: `reviewer_3hfp_alpha_v1`.
- QAP: 10,000 simultaneous donor/receiver task-label permutations, seed 2038,
  using the same dependency-preserving construction as `005`.
- Weights & Biases: disabled.

Any future change to these invariants requires a run-ID re-election or a new
analysis-spec version before another run.

### 7.4 Per-cell derived quantities

For each of the 462 cross-task cells, persist the full validation
`Delta_alpha` curve and compute:

1. The Euclidean predicted optimal scale

   ```text
   alpha_pred_I(D,R) = <rho_D,rho_R> / ||rho_D||^2
                     = cosine_I(D,R) ||rho_R|| / ||rho_D||.
   ```

   Unlike cosine alone, this quantity is directional because the donor norm is
   in the denominator.
2. The operationally clipped prediction on the available nonnegative range,
   `clip(alpha_pred_I, 0, 1.5)`, while retaining the raw value.
3. The best observed nonnegative validation gain over
   `{0, 0.15, ..., 1.5}`. Alpha zero participates in the optimum so a harmful
   positive patch can always be rejected.
4. Every maximizing grid alpha within absolute accuracy tolerance `1e-12`,
   plus the tie count, lowest maximizer, highest maximizer, and midpoint of the
   maximizing interval. No arbitrary single tie winner is hidden.
5. Left/right boundary-censoring flags when the maximum includes 0 or 1.5.
6. Unit-scale regret, positive-gain interval, plateau width, sign-change count,
   and existing `003` unimodality diagnostic.
7. An unconstrained quadratic least-squares fit to the 12 anchored accuracy
   points, storing coefficients, R-squared, concavity, and the vertex only when
   the fit is concave. A poor or convex fit is reported, never repaired or
   silently discarded.
8. Grid-best validation recovery normalized by the receiver's same-task
   unit-scale validation gain. A non-positive denominator is a hard failure,
   not a missing value or zero.

The 22 same-task curves are persisted as a separate audit and excluded from
cross-task inference.

### 7.5 Predeclared statistical questions

#### Primary: alignment association as a function of alpha

At each of the 11 measured positive alphas, correlate signed Euclidean cosine
with `Delta_val(D,R; alpha)` over the 462 cross-task cells. Spearman is primary
and Pearson secondary.

All alphas are reported. No alpha may be selected after looking at the
correlations and presented as though it had been preselected. The 10,000 shared
QAP permutations produce:

- pointwise two-sided QAP p-values for description;
- the null distribution of `max_alpha |rho_alpha|`;
- one family-wise max-stat p-value for the claim that any tested positive alpha
  exhibits an association.

The unit-scale alpha is marked explicitly so the validation profile can be
compared with the existing test-split primary result without conflating the two
splits.

#### Theory-adjacent secondary: optimized recovery

Correlate squared Euclidean cosine with grid-best validation recovery using
Spearman and Pearson plus the same task-label QAP. This is labeled
theory-adjacent and selection-optimistic because both the maximizing alpha and
its accuracy are measured on validation. It diagnoses the upper envelope of
the existing positive-grid curves; it is not a new generalization-performance
claim.

#### Mechanistic diagnostics

- Compare raw and clipped `alpha_pred_I` with the tie-aware empirical optimum
  midpoint using Spearman/Pearson and absolute-error summaries.
- Report whether the prediction falls inside the empirical maximizing interval,
  plateau, and positive-gain interval.
- Report results separately for interior and boundary-censored optima, but do
  not promote a filtered subset over the all-cell result.
- Summarize concave-fit fraction, R-squared, vertex censoring, unit-scale
  regret, and unimodality.
- Show descriptive median Delta curves by precomputed cosine quartile. These
  bands receive no IID confidence interval.

Diagnostics are not additional primary hypotheses and cannot replace a null
primary profile after results are observed.

### 7.6 Producer, artifact, and completion contract

One deterministic Hydra-configured analyzer is proposed:

```text
code/experiments/998_rebuttal/006_alignment_alpha_response/
  analyze_alpha_response.py
config/experiments/998_rebuttal/006_alignment_alpha_response/
  analyze_alpha_response.yaml
```

It validates the exact task order, 484 curve cells, 462 off-diagonal cells,
11-point grid, source configs, and SHA-256 hashes before analysis. It uses
`guard_run_config` and `StatusWriter`, writes atomically, refuses to overwrite a
golden artifact, and has no partial resume.

Golden completion signal:

```text
evaluations/998_rebuttal/006_alignment_alpha_response/
  <run_id_path>/alpha_response_statistics.json
```

The JSON persists source provenance, all 462 cross-task point/curve records,
22 diagonal audits, the complete correlation profile, full QAP null vectors,
tie/boundary diagnostics, quadratic fits, and an empty `missing` list.

No checkpoints are produced: this is a short deterministic aggregation, so
there are no checkpoint filenames, training-state contents, retention policy,
or resume support. The golden statistics JSON is the sole completion signal.

### 7.7 Proposed run-ID election

All config parameters are persisted in `.run_config.json`, but the proposed
ordered identifying subset is:

1. `model_name`
2. `curve_split`
3. `curve_baseline`
4. `curve_grid`
5. `analysis_spec`
6. `n_permutations`
7. `permutation_seed`

The fixed task set, seed 2038, optimizer/QAT/PTQ contract, unit alpha,
geometry spec, tie tolerance, source paths, and `use_wandb=false` remain
experiment invariants rather than run-ID axes. This keeps the flat identity
within the filesystem component limit. Varying any invariant later requires
explicit run-ID evolution before execution.

### 7.8 Proposed figures

All plots read only `alpha_response_statistics.json`, use argparse, and write
PDF plus 300-dpi PNG:

1. `plot_alpha_correlation_profile.py`: Spearman and Pearson versus alpha,
   unit-scale marker, pointwise QAP values, and the family-wise max-stat result.
2. `plot_alpha_geometry_calibration.py`: predicted versus tie-aware empirical
   optimal alpha, plus cosine-squared versus grid-best recovery; full-range and
   bulk-zoom views preserve boundary points.
3. `plot_alpha_curve_strata.py`: descriptive median Delta(alpha) curves by
   fixed cosine quartile, with cell counts and no IID uncertainty claim.

Visualization scripts never rescan the 5,324 source files, refit statistics,
or rerun QAP.

### 7.9 Proposed execution

After implementation and synthetic/contract verification, dispatch one
CPU-only run on local rig-4090; `gpu0` is only the standard lane identity and
CUDA is not invoked. Estimated analyzer runtime is under one minute, followed
by render-only figure generation. Placement, implementation, execution, and
the exact wave remain pending user approval.

### 7.10 Interpretation gate

The possible outcomes are precommitted:

- A broad association profile away from alpha=1, agreement between predicted
  and empirical optima, and improved optimized-recovery association would
  support scale mismatch as a concrete explanation of the unit-scale null.
- A profile that is null at every alpha would strengthen the conclusion that
  Euclidean geometry is insufficient, even after exposing the full positive
  response curve.
- A narrow isolated alpha peak without family-wise support is not evidence.
- Good curve shape but poor H=I scale prediction would support the scale
  mechanism while motivating receiver-specific `H_R` geometry.
- Poor quadratic shape would directly document failure of the local accuracy
  proxy, regardless of whether one rank correlation is small-p.

No reviewer-response language is changed until the golden artifact and figures
are inspected with the user in a separately greenlit interpretation step.

### 7.11 Approved implementation and verification

The user approved the complete Step-7 contract, the CPU-only rig-4090/gpu0
lane, and the proposed journal entry. The implementation now exists at the
paths in Section 7.6, together with the three render-only scripts in Section
7.8 and `code/test/alpha_response_alignment.py`.

Five targeted tests pass. They validate the locked Hydra config and both real
source artifacts without generating results; verify the directional donor-norm
formula, the analytic alpha-zero anchor, complete tie intervals, quadratic
concavity rules, deterministic shared-permutation QAP and max-stat adjustment;
and render all three figures to temporary PDF/PNG files from synthetic golden
statistics. Python compilation and `git diff --check` also pass.

No real `alpha_response_statistics.json`, wave scripts, logs, or plot files
were generated in this step. The next separately gated action is generation
and launch of the single CPU-only analysis wave, followed by render-only figure
generation. Interpretation remains a later gate.

### 7.12 Real execution and artifact validation

With explicit user approval, the single CPU-only run was generated and
launched on the rig-4090 gpu0 lane as wave `20260802-211910`. The self-guarded
lane ran from 2026-08-02 21:22:47 to 21:22:50 CEST and completed in 2.9 seconds
with progress `statistics and artifact 4/4; artifact written`. The status,
golden artifact, and timestamped log agree, and the log contains no traceback.

The golden artifact is:

```text
evaluations/998_rebuttal/006_alignment_alpha_response/
  model_name=vit_base_patch16_224.orig_in21k/
  curve_split=val/curve_baseline=fp_ptq/curve_grid=shared/
  analysis_spec=reviewer_3hfp_alpha_v1/
  n_permutations=10000/permutation_seed=2038/
  alpha_response_statistics.json
```

It is 12,331,488 bytes with SHA-256
`ead61f94074b9aa03e13845c1bc4fbcfbd74294006cc1d72c29c36bf55cb07f4`.
Both recorded source hashes were independently recomputed and match. Schema
validation confirms 462 cross-task records, 22 diagonal audits, 11 primary
alpha values, 10,000 persisted null draws per alpha and coefficient, four
cosine quartiles totaling 462 cells, and an empty `missing` list.

The three render-only scripts produced six nonempty real files: one PDF and
one 300-dpi PNG each for the correlation profile, geometry/scale calibration,
and cosine-quartile response curves. The PNG dimensions are 2284x2134,
3424x2734, and 2344x1594 respectively. Visual inspection confirms that labels,
full-range and zoom panels, boundary points, legends, and unit-alpha markers
are legible. Matplotlib used a temporary writable cache because the default
user cache is unavailable; this did not affect outputs.

This step records only execution and artifact validity. No coefficient,
p-value, curve shape, or reviewer-response implication is interpreted here.
The next separately gated step is the precommitted scientific interpretation
and update of the evolving reviewer answer.

## Step 8: Level-A interpretation and reviewer-response update

### 8.1 What the alpha sweep changes

The original reviewer-literal result remains unchanged: at the paper's
unit-scale transfer setting, signed global Euclidean cosine does not predict
cross-task **test** gain. The Step-6 estimate is Spearman `rho = 0.01496`
with task-label QAP `p = 0.86051`; Pearson is `r = -0.02039`, `p = 0.77162`.

The complete **validation** alpha profile shows why this null result should not
be read as “alignment never matters.” The association is strongly
scale-dependent:

| alpha | Spearman rho | pointwise QAP p | max-stat adjusted p | Pearson r | Pearson adjusted p |
|---:|---:|---:|---:|---:|---:|
| 0.15 | 0.38948 | 0.00010 | 0.00020 | 0.25771 | 0.00310 |
| 0.30 | 0.36328 | 0.00030 | 0.00030 | 0.23436 | 0.00710 |
| 0.45 | 0.33530 | 0.00030 | 0.00040 | 0.19157 | 0.02720 |
| 0.60 | 0.26507 | 0.00500 | 0.01090 | 0.13576 | 0.15368 |
| 0.75 | 0.15807 | 0.06979 | 0.18618 | 0.06874 | 0.63024 |
| 0.90 | 0.05670 | 0.49775 | 0.85891 | 0.00146 | 1.00000 |
| 1.00 | -0.02271 | 0.78292 | 0.99230 | -0.03593 | 0.91151 |
| 1.05 | -0.04265 | 0.58954 | 0.93461 | -0.05253 | 0.77782 |
| 1.20 | -0.12888 | 0.06399 | 0.33607 | -0.09457 | 0.40286 |
| 1.35 | -0.18396 | 0.00380 | 0.10279 | -0.12289 | 0.21238 |
| 1.50 | -0.24401 | 0.00020 | 0.02040 | -0.14771 | 0.10889 |

The global shared-permutation test for any Spearman association across the 11
reported scales gives `max |rho| = 0.38948`, `p = 0.00020`; the analogous
Pearson test gives `max |r| = 0.25771`, `p = 0.00310`. Thus the small-alpha
positive association is not an uncorrected scan artifact. Spearman remains
significant after max-stat correction at alpha 0.15, 0.30, 0.45, and 0.60,
and the negative association at alpha 1.50 also survives correction. The
association passes through zero near unit scale.

This is a more informative outcome than either “cosine predicts transfer” or
“cosine is irrelevant.” Euclidean alignment predicts where a pair lies on a
scale-dependent response pattern: high-alignment pairs tend to respond more
strongly at small alpha, peak earlier, and deteriorate more rapidly when the
same patch is enlarged. A single unit-scale matrix intersects these response
curves near the point at which the cross-pair rank association vanishes.

### 8.2 Descriptive response-curve evidence

The cosine-quartile curves make the sign change concrete. These are
descriptive medians over the fixed quartiles, without IID confidence bands:

| signed-cosine quartile | n | median empirical best alpha | peak of the median curve | median gain at alpha=1 | median gain at alpha=1.5 |
|---|---:|---:|---:|---:|---:|
| Q1, lowest | 116 | 0.975 | 0.04085 at 1.00 | 0.04085 | 0.02994 |
| Q2 | 114 | 0.900 | 0.07574 at 0.90 | 0.07047 | 0.03066 |
| Q3 | 116 | 0.750 | 0.08160 at 0.60 | 0.04900 | -0.00131 |
| Q4, highest | 116 | 0.600 | 0.10730 at 0.60 | 0.05891 | -0.01781 |

Across individual cells, the empirical best-alpha midpoint has median 0.75
and interquartile range 0.60--1.05. At unit scale, 342/462 cross-task cells
have positive validation gain, five are exactly zero, and 115 are negative.
Allowing alpha zero and the full positive grid yields a positive best gain for
445/462 cells; the remaining 17 select no patch. Relative to unit scale, the
grid-best validation gain improves by a median 0.0214 and mean 0.0399 absolute
accuracy (2.14 and 3.99 percentage points), with a maximum regret of 28.9
points. These are validation-set upper-envelope quantities and are therefore
mechanistic/descriptive, not unbiased test-performance estimates.

There is genuine support for a curved scale response: 413/462 quadratic fits
are concave, the median quadratic `R^2` is 0.932, and 445/462 have `R^2 >= 0.5`.
However, only 224/462 curves are strictly unimodal on the anchored grid, and
49 concave fits place their vertex outside the measured range. Top-1
discretization, finite displacement, and local irregularities therefore remain
material; a good aggregate parabola is not the same as satisfying every
assumption of Proposition 1.

### 8.3 The Euclidean optimal-scale prediction fails

Under the H=I quadratic proxy, the predicted optimum is

```text
alpha_pred_I(D,R) = <rho_D,rho_R> / ||rho_D||^2
                  = cosine_I(D,R) ||rho_R|| / ||rho_D||.
```

This is the more theory-specific scale diagnostic because it uses the dot
product and donor/receiver norm ratio, not cosine alone. It fails strongly:

- raw `alpha_pred_I` has median 0.0190 and range -0.0236--0.2532, whereas the
  empirical best-alpha midpoint has median 0.75 and range 0--1.5;
- 34/462 raw predictions are negative, and none exceeds 0.2532;
- raw prediction versus empirical optimum gives Spearman `rho = -0.29476`,
  QAP `p = 0.00010`, and Pearson `r = -0.21711`, `p = 0.00060`;
- clipping predictions to [0,1.5] gives Spearman `rho = -0.29821`,
  `p = 0.00010`, and Pearson `r = -0.22652`, `p = 0.00030`;
- the clipped prediction lies inside the exact empirical maximizing interval
  for only 4/462 cells and inside no cell's 90%-of-maximum plateau;
- its median absolute error is 0.7304 alpha units;
- the negative rank association remains in the 420 interior-optimum cells
  (`rho = -0.30708`), so it is not produced by the 42 boundary-censored cells.

The negative calibration is consistent with the quartile curves: higher
Euclidean alignment is associated descriptively with an **earlier**, not
later, empirical peak. Therefore the sweep supports scale as a practical
mechanism but rejects H=I as a quantitatively adequate geometry for predicting
the optimum scale.

### 8.4 The cosine-squared recovery test remains unsupported

For the Level-A upper envelope, squared Euclidean cosine versus grid-best
validation recovery gives Spearman `rho = 0.11467`, QAP `p = 0.12619`, and
Pearson `r = 0.00171`, `p = 0.96480`. Recovery ranges up to 4.520, again
violating the [0,1] range of the exact loss-recovery fraction and confirming
that normalized Top-1 accuracy recovery is only a proxy.

The earlier Step-6 endpoint—validation-selected **test** recovery—had a weak
positive rank association (`rho = 0.15841`, `p = 0.02930`) but essentially no
linear association. The complete validation upper-envelope analysis does not
replicate that rank result. The honest synthesis is not to select the more
favorable endpoint: there is no stable quantitative evidence that Euclidean
`cosine^2` predicts recovered gain.

### 8.5 Consequence for Proposition 1

The sweep separates two claims that would otherwise be conflated:

1. **Scale is important.** This is well supported. The association changes
   systematically across alpha, unit scale incurs substantial regret for many
   cells, cosine quartiles peak at different scales, and most curves admit a
   good concave quadratic fit.
2. **The Euclidean local-quadratic geometry predicts the response.** This is
   not supported. Unit-scale cosine is null, H=I predicted optima are grossly
   miscalibrated and anticorrelated with empirical optima, and Euclidean
   cosine-squared does not predict grid-best recovery.

This does not logically falsify Proposition 1, because the proposition uses a
receiver-specific `H_R`, a smooth local loss, matching vector/outcome
subspaces, and an unconstrained real optimum. H=I cannot encode receiver
directionality, and accuracy is not the proposition's loss. But the data also
do not validate the proposition. The correct theoretical positioning is:

> Proposition 1 is a local organizing model that correctly highlights
> alignment and scale as coupled variables. For finite QAT displacements and
> hard-quantized Top-1 transfer, global Euclidean geometry is not a calibrated
> predictive law. Testing the stronger claim requires receiver-specific
> curvature and a smooth, subspace-matched loss endpoint.

This is the precommitted “good curve shape but poor H=I scale prediction”
outcome from Section 7.10. It motivates `H_R` rather than permitting us to
claim that an unmeasured `H_R` would necessarily succeed.

### 8.6 Reviewer-response draft v2

> We thank the reviewer for requesting this direct validation. For the
> ViT-B/16 22-by-22 vision matrix, we computed the Euclidean cosine between
> every matched donor and receiver QV over the 48 `nn.Linear.weight` tensors
> acted on by our 3-bit per-channel quantizer (84,934,656 coordinates), using
> one global inner product rather than averaging layer-wise or channel-wise
> cosines. We retain the 22 same-task cells as an audit but exclude them from
> cross-task association statistics, leaving 462 directed transfer cells.
>
> For the reviewer's literal H=I comparison, signed Euclidean cosine does not
> predict the observed unit-scale test-accuracy gain: Spearman rho = 0.015
> (task-label QAP p = 0.861) and Pearson r = -0.020 (p = 0.772). Thus raw
> Euclidean angle is not an empirical predictor of the paper's unit-scale
> transfer matrix.
>
> We additionally analyzed the complete pre-existing 11-point validation
> scale sweep, reporting every alpha and controlling the scan with shared
> task-label permutations. The alignment association is strongly
> scale-dependent: Spearman rho is 0.389 at alpha=0.15, decreases to -0.023
> at alpha=1, and reverses to -0.244 at alpha=1.5. The global max-|rho| QAP
> test gives p=0.0002; the small-alpha associations through alpha=0.60 and the
> negative alpha=1.5 association survive max-stat correction. Descriptively,
> the highest-alignment quartile peaks at alpha=0.60, whereas the lowest
> quartile peaks near alpha=1. This explains why the unit-scale slice can be
> null even though alignment and scale interact systematically.
>
> This pattern does not validate the quantitative cosine-squared law. Under
> H=I, the quadratic theory predicts
> `alpha*=<rho_D,rho_R>/||rho_D||^2`. Its median prediction is 0.019 versus an
> empirical median optimum of 0.75, and it is anticorrelated with the empirical
> optimum (Spearman rho=-0.298, QAP p=0.0001 after clipping to the measured
> range). Squared Euclidean cosine also does not predict grid-best validation
> recovery (Spearman rho=0.115, p=0.126; Pearson r=0.002, p=0.965). Although
> 413/462 response-curve fits are concave and the median R^2 across all curves
> is 0.932, the Euclidean geometry does not calibrate their optima or recovered
> gain.
>
> We therefore revise the theoretical interpretation. Proposition 1 is exact
> for a smooth local quadratic receiver loss in the receiver-specific H_R
> metric and at the best unconstrained real scale. Our experiment instead uses
> H=I, finite QAT displacements, hard PTQ/Top-1 outcomes, a bounded positive
> grid, and geometry restricted to the quantized-Linear subspace. Moreover,
> H=I cosine is symmetric while transfer is directional. The results support
> alignment and scale as useful organizing variables, but not Euclidean
> cosine-squared as a validated predictive law for the 22-by-22 accuracy
> matrix. A direct test of the stronger claim requires receiver-specific
> curvature and a smooth, subspace-matched loss endpoint.

### 8.7 Recommended manuscript/response packaging

The reviewer answer should lead with the negative unit-scale test result, not
with the significant small-alpha profile. The alpha profile then explains the
null without claiming that the proposition has been rescued. Recommended
packaging is:

1. Put the correlation-profile figure in the rebuttal or main appendix. It
   directly answers whether alignment depends on scale and visibly marks the
   unit setting and family-wise QAP results.
2. Put the geometry-calibration and quartile-curve figures in supplementary
   material. The former documents failure of the H=I predicted optimum and
   cosine-squared recovery; the latter gives an intuitive response-shape view.
3. Add a short methods sentence defining the one global cosine over the 48
   quantized Linear weights and the exclusion of the 22 algebraic diagonal
   cells from inference.
4. Revise any language that presents the cosine-squared law as empirically
   established. Call it a local explanatory idealization and state the exact
   assumptions needed for a direct test.
5. Do not imply that receiver-specific curvature has been validated. It is a
   clearly motivated next experiment, not a result of this sweep.

### 8.8 Final Level-A decision

The alpha sweep was worth running. It turns the unit-scale null into a precise
mechanistic result: Euclidean alignment carries scale-dependent ordinal
information, and unit alpha happens to lie near the profile's zero crossing.
At the same time, the stronger diagnostics rule out an overly favorable
interpretation. H=I predicts neither the correct optimum scale nor the
empirical grid-best recovery proxy expected to track cosine-squared alignment.
The reviewer can therefore be answered with new empirical evidence while the
theoretical claim is narrowed to what the data actually support.

## 1. Reviewer request

Reviewer 3HFP asks whether the central prediction of Proposition 1 is
empirically visible:

> The cos²-alignment law in Proposition 1 is the central prediction of the
> local theory, but the quantity cosH(ρ_D, ρ_R) is never computed or validated
> empirically. Even the Euclidean (H=I) cosine similarity between matched donor
> and receiver QVs would be informative. Does it correlate with observed
> Δ(D,R) across the 22×22 vision matrix? If not, how should the theoretical
> account be interpreted?

Proposition 1 predicts that, under a local quadratic model of the receiver's
post-quantization objective, the fraction of receiver-side QAT gain recovered
by the optimally scaled donor patch is

\[
\cos^2_{H_R}(\rho_D,\rho_R),
\]

where \(H_R\) is the receiver-specific loss Hessian.

The study must distinguish three increasingly theory-specific questions:

1. Does ordinary Euclidean QV alignment predict the observed Top-1 transfer
   gain?
2. Does receiver-specific curvature-weighted alignment predict it better?
3. Are the assumptions needed for Proposition 1--local stationarity, positive
   curvature, and approximately quadratic behavior--empirically reasonable?

These questions are related but are not interchangeable.

## 2. Locked QV definition for this study

For this analysis, project each QV onto exactly the weights quantized by the
project's PTQ operator:

\[
\widetilde{\rho}_D=P_{\mathcal S}
\left(\theta_{D,\mathrm{QAT}}-\theta_{D,\mathrm{FP}}\right),
\]

where \(\mathcal S\) contains:

- The `.weight` tensors of `nn.Linear` modules touched by `apply_ptq_`.
- No classification-head parameters, because the head is explicitly skipped.
- No Linear biases, because `apply_ptq_` does not quantize them.
- No LayerNorm parameters, embeddings, patch-convolution weights, or other
  tensors.

This choice focuses the analysis on the parameter subspace directly subjected
to 3-bit rounding.

It must be described as **quantized-subspace QV alignment**, not as the cosine
of the paper's complete backbone QV. The paper's transferred QV also contains
non-Linear backbone parameters.

## 3. Subspace/outcome consistency

The existing \(\Delta(D,R)\) matrix was produced by transferring the complete
backbone QV, whereas the proposed cosine uses only its quantized-Linear
projection. There are consequently two distinct analyses.

### 3.1 Reviewer-literal association test

Compare quantized-subspace cosine with the existing paper results:

\[
\cos_I(\widetilde{\rho}_D,\widetilde{\rho}_R)
\quad\text{versus}\quad
\Delta_{\mathrm{full}}(D,R).
\]

This directly answers whether similarity in the weights touched by PTQ predicts
the already reported transfer outcomes.

It is informative, but it is not an exact test of the cos² law: the explanatory
variable uses a projected QV while the outcome was produced by transferring the
full QV.

### 3.2 Subspace-consistent theoretical test

Transfer only the projected QV:

\[
\theta_{R\leftarrow D}^{\mathcal S}
=\theta_R+\lambda\widetilde{\rho}_D,
\]

leaving every parameter outside \(\mathcal S\) at its FP receiver value. This
produces a matched outcome:

\[
\Delta_{\mathcal S}(D,R)
=\operatorname{Acc}\left(
\operatorname{PTQ}(\theta_R+\lambda\widetilde{\rho}_D)
\right)
-\operatorname{Acc}\left(\operatorname{PTQ}(\theta_R)\right).
\]

The corresponding receiver-side ceiling must also use the projected receiver
QV:

\[
\Delta_{\mathcal S,\mathrm{ceiling}}(R)
=\Delta_{\mathcal S}(R,R).
\]

This second matrix requires new evaluations, but it is the scientifically
controlled test if the alignment is restricted to quantized Linear weights.

The first Euclidean implementation and pilot use only the existing full-patch
\(\Delta\), answering the reviewer's literal question. Producing projected-patch
\(\Delta_{\mathcal S}\) to remove the subspace/outcome mismatch remains Step 7
of the gated plan. It is not part of the first implementation or run and
requires its own later design review and greenlight.

## 4. Euclidean test

The Euclidean alignment is

\[
c_I(D,R)=
\frac{\widetilde{\rho}_D^\top\widetilde{\rho}_R}
{\lVert\widetilde{\rho}_D\rVert_2
 \lVert\widetilde{\rho}_R\rVert_2}.
\]

This matrix is symmetric:

\[
c_I(D,R)=c_I(R,D).
\]

The outcome matrix is not symmetric because transfer is evaluated on the
receiver:

\[
\Delta(D,R)\ne\Delta(R,D).
\]

Euclidean alignment therefore cannot explain every directional effect. At
best, it identifies pairs whose QVs share a common direction before accounting
for receiver-specific geometry.

### 4.1 Primary comparisons

For unit-scale positive patching, compare the signed cosine with the observed
unit-scale gain:

\[
c_I(D,R)
\quad\text{versus}\quad
\Delta(D,R;\lambda=1).
\]

The sign matters because the deployed patch uses positive \(\lambda\). Squaring
the cosine would make aligned and anti-aligned vectors indistinguishable.

For the theory-motivated best-scale analysis, compare

\[
c_I(D,R)^2
\quad\text{versus}\quad
\operatorname{recovery}(D,R).
\]

The paper's selected \(\lambda\) is restricted to a positive finite grid,
whereas Proposition 1 optimizes over an unconstrained scalar. This comparison
is therefore approximate and must be labeled accordingly.

## 5. What the Hessian-weighted result requires

Proposition 1 uses

\[
c_{H_R}(D,R)=
\frac{\widetilde{\rho}_D^\top H_R\widetilde{\rho}_R}
{\sqrt{\widetilde{\rho}_D^\top H_R\widetilde{\rho}_D}
 \sqrt{\widetilde{\rho}_R^\top H_R\widetilde{\rho}_R}}.
\]

Unlike the Euclidean matrix, this is receiver-specific. In general,

\[
c_{H_R}(D,R)\ne c_{H_D}(R,D).
\]

This asymmetry is desirable because it can potentially explain the asymmetric
transfer matrix.

Prediction logits alone are insufficient. The experiment must know how the
logits change when parameters move along every QV direction:

\[
J_\theta f_R(x)\widetilde{\rho}_D.
\]

For each receiver, this requires:

- The receiver model and QAT checkpoint.
- Receiver inputs and labels.
- The 22 projected QVs.
- A differentiable scalar loss, using cross-entropy.
- Forward and backward or directional-derivative computations.

The full Hessian and full parameter Jacobian do not need to be materialized or
stored.

## 6. Why the exact hard-PTQ Hessian is problematic

Hard rounding is piecewise constant in the weights. Its ordinary gradient is
zero almost everywhere and undefined at quantization boundaries. Top-1
accuracy is also non-differentiable.

There is consequently no useful classical Hessian of the literal "hard PTQ
followed by Top-1 accuracy" computation.

A curvature experiment must use an explicitly declared smooth surrogate. The
closest available surrogate in this repository is the QAT forward pass with
the straight-through estimator:

- Forward: use the same fake-quantized weights.
- Backward: treat the quantizer approximately as the identity.
- Objective: receiver cross-entropy rather than Top-1 accuracy.

The resulting matrix must be called the **STE-surrogate Hessian**, not the exact
Hessian of hard PTQ.

## 7. Curvature measurements

### 7.1 Restricted STE Hessian

For receiver \(R\), place the 22 projected QVs in the columns of \(V\). Compute
only

\[
K_R=V^\top H_R^{\mathrm{STE}}V.
\]

This is a 22×22 matrix even though the full Hessian has billions of entries. It
can be obtained using Hessian-vector products.

Before using it as a metric, test:

- Symmetry of \(K_R\).
- Eigenvalues of its symmetric part.
- Whether \(\widetilde{\rho}^\top H_R\widetilde{\rho}>0\).
- Directional gradients \(V^\top\nabla g_R\), testing whether the proposed
  receiver point is locally stationary.

If \(K_R\) is not positive semidefinite, it does not define a valid norm and
its "cosine" is not generally meaningful. That would be evidence that
Proposition 1's positive-definite local-quadratic assumption does not hold in
this restricted subspace. This is a scientific result, not an implementation
failure.

### 7.2 Generalized Gauss--Newton geometry

For each example \(i\), let

\[
z_{i,D}=J_\theta f_R(x_i)\widetilde{\rho}_D
\]

be the directional change in its logits. For cross-entropy, let

\[
C_i=\operatorname{diag}(p_i)-p_ip_i^\top
\]

be the loss curvature in logit space. Accumulate

\[
G_R[D,E]=\frac{1}{N}\sum_i z_{i,D}^\top C_i z_{i,E}.
\]

The GGN matrix is positive semidefinite by construction and gives a valid
cosine:

\[
c_{G_R}(D,R)=\frac{G_R[D,R]}
{\sqrt{G_R[D,D]G_R[R,R]}}.
\]

This is the proposed main curvature-aware measurement. It captures how the
receiver's predictions respond to donor and receiver QVs.

It is still a proxy for \(H_R\), so it must be reported as "GGN-weighted
alignment," not as an exact measurement of the theoretical Hessian.

## 8. Theory-faithful outcome

Proposition 1 concerns objective improvement, not Top-1 accuracy. A proper
curvature test should therefore record receiver cross-entropy as well as
accuracy.

For a donor direction, define loss recovery:

\[
\mathcal R_{\mathrm{loss}}(D,R)=
\frac{L_R(0)-L_R(\lambda^\star\widetilde{\rho}_D)}
{L_R(0)-L_R(\widetilde{\rho}_R)}.
\]

Within the projected subspace, the denominator must use the projected receiver
patch. Using the complete QAT gain would again mix two different parameter
spaces.

The strongest test is then

\[
c_{G_R}(D,R)^2
\quad\text{versus}\quad
\mathcal R_{\mathrm{loss}}(D,R).
\]

Additional checks should compare:

- Curvature-predicted optimal scale

  \[
  \lambda^\star_{\mathrm{GGN}}=
  \frac{\widetilde{\rho}_D^\top G_R\widetilde{\rho}_R}
  {\widetilde{\rho}_D^\top G_R\widetilde{\rho}_D}.
  \]

- Empirically optimal scale from a loss curve.
- Predicted loss recovery \(c_{G_R}^2\).
- Observed loss recovery.
- Observed Top-1 \(\Delta\), as the deployment-facing quantity.

This separates "does the local quadratic law describe the loss?" from "does
that loss geometry predict discrete changes in classification accuracy?"

## 9. Statistical treatment

The 462 off-diagonal cells are not 462 independent observations. Every task
appears repeatedly as both donor and receiver.

Therefore:

- Exclude the 22 diagonal cells from correlations. On the diagonal, cosine is
  one algebraically and the outcome is receiver-side QAT, not cross-task
  transfer.
- Report Spearman correlation as the primary association statistic.
- Report Pearson correlation as a secondary functional-form check.
- Use task-label permutations--a quadratic-assignment permutation test--rather
  than an IID correlation p-value.
- Report leave-one-receiver-out and leave-one-donor-out sensitivity.
- If multiple architectures are included, report them separately. Do not
  manufacture significance by naively pooling thousands of dependent cells.

## 10. Planned figures

The reviewer-facing minimum should be:

1. **Alignment heatmap:** signed Euclidean cosine over the quantized Linear
   subspace, with the diagonal visible but excluded from color-range estimation.
2. **Literal reviewer scatter:** signed Euclidean cosine versus unit-scale
   Top-1 \(\Delta\), excluding diagonal cells and showing Spearman, Pearson, and
   permutation significance.
3. **Theory scatter:** squared alignment versus loss-recovery ratio at the
   appropriate scale.
4. **Geometry comparison:** Euclidean versus GGN-weighted alignment, showing
   whether receiver curvature materially changes pair ordering.
5. **Assumption diagnostics:** restricted-Hessian eigenvalues,
   directional-gradient residuals, and representative
   measured-versus-quadratic loss curves.

If several backbones are evaluated, add a compact forest plot of per-backbone
correlations rather than pooling all points into one unreadable scatter.

## 11. Precommitted interpretation

The interpretation must be fixed before seeing results.

### 11.1 Euclidean and curvature alignment both correlate

This supports the local alignment account and shows that the effect is already
visible in raw weight geometry.

### 11.2 Euclidean is weak, but GGN is predictive

This would be a particularly coherent result: ordinary parameter-space
similarity misses the effect, while receiver-sensitive loss geometry recovers
it. Proposition 1 specifically predicts \(H_R\)-alignment, not \(H=I\).

### 11.3 Euclidean correlates, but GGN does not

This requires investigation. Possible causes include an unsuitable curvature
surrogate, insufficient receiver data, lack of stationarity, or numerical
problems. The more complicated metric must not be preferred automatically.

### 11.4 Neither predicts transfer

The theory should then be presented as a local organizing idealization rather
than an empirically predictive model of the 22×22 matrix. Likely sources of
mismatch include:

- QVs are not local displacements.
- The QAT checkpoint is not a stationary point of the chosen evaluation loss.
- Hard quantization and Top-1 accuracy are not smooth.
- The restricted QV subspace omits causally important parameters.
- The positive-definite quadratic approximation is violated.
- The practical \(\lambda\) search is positive and bounded.
- A single local loss metric may not explain dataset-level accuracy changes.

### 11.5 Restricted Hessian is indefinite or nonstationary

This directly indicates that the assumptions of Proposition 1 are not
satisfied at the evaluated point and scale. The result must be reported rather
than forcing a Hessian cosine.

## 12. Initial reviewer-response template

> We thank the reviewer for suggesting this direct validation. We computed
> pairwise alignment between the matched donor and receiver QVs over the
> `nn.Linear` weight tensors directly affected by our 3-bit per-channel
> quantizer. We first evaluated the requested Euclidean \(H=I\) cosine across
> the 22×22 vision matrix. We excluded the 22 same-task diagonal cells from
> correlation statistics because their cosine equals one algebraically and
> they correspond to receiver-side QAT rather than cross-task transfer.
>
> Across the 462 cross-task pairs, signed Euclidean cosine had Spearman
> correlation **[rho]** and Pearson correlation **[r]** with unit-scale
> transfer gain \(\Delta(D,R)\), with a task-label permutation p-value of
> **[p]**. Squared Euclidean cosine had correlation **[rho/r]** with
> **[loss or accuracy recovery quantity]**.
>
> Because Proposition 1 concerns the receiver-specific \(H_R\)-geometry rather
> than \(H=I\), we additionally estimated a generalized Gauss--Newton metric
> from receiver validation data. Its squared cosine correlated
> **[more/less/similarly]** with receiver loss recovery: **[statistics]**. The
> restricted STE-Hessian was **[positive semidefinite / indefinite]**, and the
> directional stationarity diagnostic was **[result]**.
>
> These results imply **[precommitted interpretation corresponding to the
> outcomes above]**. We have revised the manuscript to distinguish the exact
> local-quadratic statement from its empirical validity over finite QAT
> displacements and discrete Top-1 accuracy.

No result language may be filled in until the corresponding data exist.

## 13. Historical open decisions before implementation

The following decisions remain unresolved and must be made explicitly before
any code is written or run:

1. **Outcome consistency:** whether to produce the recommended
   projected-QV-only transfer matrix in addition to correlating against the
   existing full-QV matrix.
2. **Architecture coverage:** whether ViT-B/16 is the sole 22×22 headline or
   the same analysis is replicated separately across the ten vision backbones
   already represented in `win_loss_ilharco_timm_supervised.json`.
3. **Curvature data:** receiver split, sample count, batching, and convergence
   criterion for the STE-Hessian/GGN estimates.
4. **Operational objective:** exact definition of the cross-entropy surrogate
   and the point at which curvature and stationarity are evaluated.
5. **Scale protocol:** empirical loss-curve grid and treatment of negative or
   out-of-grid curvature-predicted optima.
6. **Experiment mechanics:** run identity, final artifacts, checkpoint/resume
   policy, and whether Weights & Biases is used. These must be elected with the
   user under the research experiment-design protocol.

The decisions needed for the Euclidean ViT-B/16 pilot were resolved in the
approved Step 2 contract above. Projected-QV transfer, curvature, and broader
architecture coverage remain later, separately gated decisions.

## 14. Strict matching-row replication

The approved row-wise extension is an additive one-variable replication of the
completed ViT-B/16 Euclidean pilot. The global artifacts and their normative
interpretation remain untouched. Every model, checkpoint, task, QV coordinate,
selected module, outcome cell, statistical routine, random permutation, and
figure design is held fixed.

The sole numerical substitution is the alignment matrix. For each selected
Linear-weight matrix `ell` and matching output-row index `i`, define

\[
c_{\ell,i}(D,R)=
\frac{\langle\rho_{D,\ell}[i,:],\rho_{R,\ell}[i,:]\rangle}
{\|\rho_{D,\ell}[i,:]\|_2\|\rho_{R,\ell}[i,:]\|_2}.
\]

The replicated similarity is

\[
c_{\mathrm{row}}(D,R)=\frac{1}{82944}\sum_{\ell,i}c_{\ell,i}(D,R).
\]

All 82,944 rows receive equal weight. There is no per-layer averaging, scalar
coordinate weighting, QV-energy weighting, or quantization-step
normalization. The 73,728 width-768 rows and 9,216 width-3072 rows each receive
one vote. A zero-norm row is an error rather than an omitted or imputed value;
the preimplementation audit found zero such rows across all 22 QVs.

Downstream analysis is unchanged. In particular, the theory-adjacent field is
`cosine_sq = c_row(D,R) ** 2`; it is not the mean of squared row cosines. The
same four `reviewer_3hfp_v1` comparisons, 462-cell population, tie handling,
10,000 simultaneous-axis QAP permutations with seed 2038, influence analyses,
and plotting layouts are reused verbatim. The distinct token
`reviewer_3hfp_rowwise_v1` records only which alignment matrix entered that
unchanged analysis.

### Matching-row result

Wave `20260803-120627` completed from source revision
`943f1bec835baf0be8ca0bcae4c4942790415bd1` and included all 82,944 rows
from the same 48 tensors and 44 FP/QAT checkpoints, with zero zero-norm rows.
The primary signed row-cosine versus unit-scale gain comparison has Spearman
rho `0.0725815` (QAP `p = 0.401560`) and Pearson `r = 0.0731162` (QAP
`p = 0.315268`). The theory-adjacent squared row-cosine versus
validation-selected recovery comparison has Spearman rho `0.228536` (QAP
`p = 0.00159984`) and Pearson `r = 0.0751924` (QAP `p = 0.115188`).

The artifact audit confirmed exact checkpoint identities, task/tensor/cell
order, outcome source, non-similarity fields, comparison definitions, and the
original QAP permutation digest. Every `cosine_sq` is exactly the square of
the aggregated row-wise cosine. All six row-wise PDF/PNG files rendered and
validated, so the alignment aggregation is the sole scientific change.

The geometry stage is `rowwise_alignment`, with golden artifact
`rowwise_alignment.json`; its analysis artifact is
`rowwise_statistics.json`. The producer run identity is the original ordered
19 parameters followed by `aggregation_spec=row_cosine_mean_v1`. The analyzer
run identity remains the original ordered eight parameters, with only the
analysis-spec value changed for provenance. Both stages remain deterministic,
CPU-only, non-resumable, and W&B-free.

## 15. Validation-selected best-alpha presentation

The best-alpha presentation is an additive rendering of quantities already
produced by `reviewer_3hfp_v1`; it is not a new model experiment. It holds the
ViT-B/16 checkpoints, 22-task population, 48 selected `nn.Linear.weight`
tensors, 84,934,656-coordinate global cosine, and 462-cell off-diagonal
analysis population fixed.

For every ordered donor--receiver pair, `alpha_best` was selected by the
pre-existing `val_accuracy_fp_head_ptq` result. `delta_best` is the test-set
gain measured at that validation-selected alpha relative to the same FP+PTQ
receiver baseline. Neither the alpha selection nor any test inference was
repeated for this extension, and alpha was never selected on test.

The completed `euclidean_statistics.json` already contains the required
comparison. Signed global cosine versus validation-selected test gain has
Spearman rho `0.2516408968` and Pearson `r = 0.1227361819` across the 462
cross-task cells. Squared global cosine versus validation-selected test
recovery has Spearman rho `0.1584121246` and Pearson `r = 0.0091900528`.
The signed-cosine/best-gain Spearman coefficient remains positive in every
leave-one-task-role-out check: `0.15968--0.28781` when omitting receivers and
`0.16257--0.28316` when omitting donors.

Three render-only scripts consume the existing statistics artifact:

- `plot_best_alpha_heatmap.py` reproduces the complete global-cosine heatmap
  as a dedicated member of the best-alpha result suite. Its numerical matrix
  is necessarily identical to the original heatmap because cosine is
  independent of alpha.
- `plot_best_alpha_associations.py` reproduces the original two-panel
  association layout, substituting `delta_best` for the unit-scale `delta` in
  the signed-cosine panel while retaining the already best-scale
  `recovery_best` panel.
- `plot_best_alpha_influence.py` reproduces the leave-one-receiver-out and
  leave-one-donor-out layout for `signed_cosine_vs_delta_best`.

The new figure annotations report only Spearman and Pearson coefficients;
they intentionally omit QAP p-values to match the selected reviewer-response
presentation. The underlying statistics artifact is not altered.

All three scripts were rendered plainly on rig-4090 from the real completed
JSON. They produced six nonempty artifacts beneath
`plots/998_rebuttal/005_qv_alignment/<script_stem>/<mirrored_run_id_path>/`:
PDF plus 300-DPI PNG for each script. The heatmap PNG is 3302x2970, the
association PNG is 3634x1504, and the influence PNG is 3274x2254. Visual
inspection confirmed legible axes, task labels, coefficient annotations, and
all-cell reference lines. The full `code/test/qv_alignment_rebuttal.py` suite
passes (`20 passed`), including synthetic rendering and rejection of missing
best-alpha fields or comparison statistics.

### Matching-row best-alpha counterpart

The row-wise best-alpha presentation is likewise render-only. It reads the
completed `rowwise_statistics.json`, whose outcomes and validation-selected
alphas are exactly the same as the global-cosine artifact, and changes only
the plotted alignment geometry to the uniform mean of all 82,944 matching-row
cosines.

Signed mean-row cosine versus validation-selected test gain has Spearman rho
`0.2933233035` (QAP `p = 0.00239976`) and Pearson `r = 0.1977755079` (QAP
`p = 0.00969903`) over the same 462 cross-task cells. Squared mean-row cosine
versus validation-selected recovery has Spearman rho `0.2285356756` (QAP
`p = 0.00159984`) and Pearson `r = 0.0751923611` (QAP `p = 0.11518848`).
The signed-cosine/best-gain Spearman coefficient remains positive in every
leave-one-task-role-out check: `0.20649--0.32622` when omitting receivers and
`0.20947--0.31742` when omitting donors.

Three additive scripts—`plot_rowwise_best_alpha_heatmap.py`,
`plot_rowwise_best_alpha_associations.py`, and
`plot_rowwise_best_alpha_influence.py`—write into distinct full-provenance
paths beneath their respective script stems. All six real PDF/PNG outputs are
valid and nonempty; the PDFs are one page, and the PNG dimensions are
3324x2970, 3634x1504, and 3274x2254. Visual inspection confirmed complete
labels, unclipped annotations, and all 44 influence points. The full 005 test
suite passes (`29 passed`), including strict row-wise schema, missing-field,
missing-comparison, influence-cardinality, and unsafe-path rejection tests.
