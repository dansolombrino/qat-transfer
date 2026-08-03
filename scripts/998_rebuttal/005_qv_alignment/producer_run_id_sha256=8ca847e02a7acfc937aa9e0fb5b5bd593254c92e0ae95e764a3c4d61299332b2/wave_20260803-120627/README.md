# wave 20260803-120627 — strict matching-row replication of 005

Dispatched 2026-08-03 from rig-4090.

Source tag: `wave--20260803-120627` (the annotated tag and `master` must
resolve to the same commit on rig-4090).

Why this wave: reproduce reviewer 3HFP's completed ViT-B/16 alignment analysis
while changing only the similarity computation from one global flattened
cosine to the unweighted mean of all 82,944 matching output-row cosines.

This run: row-wise geometry producer → rig-4090, CPU-only gpu0 lane. The
canonical flat run ID is 451 bytes, so this folder uses its approved SHA-256
alias; the complete scientific identity remains in the script and evaluation
path.

Full wave: two sequential runs on rig-4090 gpu0 — row-wise geometry, then the
unchanged reviewer analysis. The three row-wise figure replicas are rendered
after the statistics artifact validates.
