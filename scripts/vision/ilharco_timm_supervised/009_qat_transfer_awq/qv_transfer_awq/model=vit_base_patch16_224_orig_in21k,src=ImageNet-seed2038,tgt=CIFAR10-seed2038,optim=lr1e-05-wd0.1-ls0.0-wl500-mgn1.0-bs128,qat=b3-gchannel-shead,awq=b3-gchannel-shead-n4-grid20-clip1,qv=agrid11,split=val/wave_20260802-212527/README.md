# ImageNet-only strong-PTQ alpha sweep

- Wave: `20260802-212527`
- Stage: validation alpha grid
- Method: `AWQ`
- Donor: `ImageNet`, seed 2038
- Receiver: `CIFAR10`, seed 2038
- Alpha grid: `[0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 1.05, 1.2, 1.35, 1.5]`
- Placement: `behemoth` GPU `4`
- Golden outputs: 11 `eval_results.json` files, one per alpha
- Resume: the producer skips existing cells; the wrapper succeeds only when all 11 artifacts are nonempty
- Selection metric: `val_accuracy_fp_head_awq`
- Follow-up: evaluate only the validation-selected alpha on test

The process evaluates the full row so all alphas reuse the same materialized
receiver calibration batches.  Behemoth placement is covered by the user's
explicit authorization for GPUs `0,2,4,5,6,7` in this wave.
