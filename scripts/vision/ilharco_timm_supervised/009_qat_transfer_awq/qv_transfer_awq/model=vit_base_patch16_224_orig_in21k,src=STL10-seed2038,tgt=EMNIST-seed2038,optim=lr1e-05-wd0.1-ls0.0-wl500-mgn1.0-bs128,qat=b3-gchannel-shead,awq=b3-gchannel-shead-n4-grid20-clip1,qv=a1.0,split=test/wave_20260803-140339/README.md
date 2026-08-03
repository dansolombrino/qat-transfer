# wave 20260803-140339 — complete the phase-009 unit-alpha AWQ grid

Prepared 2026-08-03T14:08+02:00 on rig-4090; production launch remains manual.

Source tag: `wave--20260803-140339` (the feature branch and annotated tag must resolve to the same commit on both isolated execution clones).

Why this wave: add the 462 missing donor-receiver cells required for the full 22×22 `001_009` PTQ-versus-AWQ comparison. The existing 22-cell ImageNet-donor row is reused.

This run: `model=vit_base_patch16_224_orig_in21k,src=STL10-seed2038,tgt=EMNIST-seed2038,optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128,qat=b3-gchannel-shead,awq=b3-gchannel-shead-n4-grid20-clip1,qv=a1.0,split=test` → rig-3090-ti, gpu 0.

Full wave: 462 runs — 308 on rig-4090 gpu0 and 154 on rig-3090-ti gpu0, one sequential lane per rig.
