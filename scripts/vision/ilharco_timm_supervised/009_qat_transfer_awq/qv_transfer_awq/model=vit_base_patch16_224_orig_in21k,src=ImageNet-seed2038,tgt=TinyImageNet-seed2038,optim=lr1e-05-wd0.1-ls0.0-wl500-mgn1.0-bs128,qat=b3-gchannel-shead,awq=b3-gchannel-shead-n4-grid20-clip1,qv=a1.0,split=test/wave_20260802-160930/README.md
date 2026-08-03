# wave 20260802-160930 — test QAT-QV patching on top of AWQ at lambda=1

Dispatched from rig-4090 to rig-3090-ti GPU 0.

Why this wave: reviewer-3HFP AWQ pilots with ImageNet as the sole donor.

This run: `model=vit_base_patch16_224_orig_in21k,src=ImageNet-seed2038,tgt=TinyImageNet-seed2038,optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128,qat=b3-gchannel-shead,awq=b3-gchannel-shead-n4-grid20-clip1,qv=a1.0,split=test` → rig-3090-ti, gpu 0.

Full wave: 45 runs on rig-3090-ti, gpu 0, sequentially.
