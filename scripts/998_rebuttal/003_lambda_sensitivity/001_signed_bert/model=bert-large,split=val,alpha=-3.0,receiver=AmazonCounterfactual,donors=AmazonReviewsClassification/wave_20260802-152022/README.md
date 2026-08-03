# wave 20260802-152022 — signed BERT-large lambda rebuttal experiment

Dispatched 2026-08-02 19:06 from rig-4090.

Why this wave: measure the negative and zero lambda arms for BERT-large, then
confirm negative validation selections on test for reviewer 3HFP.

This run: `model=bert-large,split=val,alpha=-3.0,receiver=AmazonCounterfactual,donors=AmazonReviewsClassification` → behemoth, gpu 5.

Full wave: 4 currently materialized runs on behemoth GPUs
0,2,4,5,6,7; GPUs 2,4,5,6,7 were user-authorized on 2026-08-02 for this
wave only. Conditional selected-test runs may be added under this same approved
wave after validation aggregation.
