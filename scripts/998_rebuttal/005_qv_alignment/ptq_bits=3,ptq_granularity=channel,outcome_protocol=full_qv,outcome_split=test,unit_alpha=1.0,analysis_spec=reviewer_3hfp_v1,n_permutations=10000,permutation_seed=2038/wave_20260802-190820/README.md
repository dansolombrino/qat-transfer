# wave 20260802-190820 — analyzer quoting-fix retry

Dispatched 2026-08-02 19:08 from rig-4090 after explicit user approval.

Why this wave: retry only `analyze_euclidean_alignment` after wave
`20260802-162236` failed before analysis because an `outcome_path` value with
embedded `=` delimiters was not quoted in Hydra's override grammar. The shared
argument helper now preserves Hydra-level string quoting, and the targeted
suite passes 18/18 tests.

This run:
`ptq_bits=3,ptq_granularity=channel,outcome_protocol=full_qv,outcome_split=test,unit_alpha=1.0,analysis_spec=reviewer_3hfp_v1,n_permutations=10000,permutation_seed=2038`
→ rig-4090, CPU-only (`gpu0` is the required lane identity; the code never
invokes CUDA).

The successful geometry artifact from wave `20260802-162236` is reused and
must not be recomputed. Real visualizations run only after the analyzer's
golden `euclidean_statistics.json` is present and validated.
