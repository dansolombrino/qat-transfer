# wave 20260802-162236 — reviewer 3HFP Euclidean QV-alignment pilot

Dispatched 2026-08-02 16:22 from rig-4090.

Why this wave: first real `reviewer_3hfp_v1` analysis after the geometry
producer and synthetic verification gates.

This run:
`ptq_bits=3,ptq_granularity=channel,outcome_protocol=full_qv,outcome_split=test,unit_alpha=1.0,analysis_spec=reviewer_3hfp_v1,n_permutations=10000,permutation_seed=2038`
→ rig-4090, CPU-only (`gpu0` is the required lane identity; the code never
invokes CUDA).

It is queued after the geometry producer in the same lane. Full wave: two
scientific stages, sequentially on the rig-4090 CPU lane, followed by three
render-only visualization commands on the same host.
