# Summary

## Problem

Post-training quantization (PTQ) of transformers degrades accuracy non-uniformly across inputs: most inputs are preserved, but a small minority is reliably broken by quantization, and that minority dominates the accuracy gap. The question this paper asks is whether the broken inputs are predictable at deployment time from cheap features, well enough to route them to a full-precision fallback and recover most of the lost accuracy. If yes, this enables an input-aware mixed-precision serving pattern where PTQ is the cheap default and FP is reserved for selective use — recovering accuracy without paying the FP cost on every input.

## Motivation

Whether FP inference is available is rarely the question in production; the question is whether its cost should be paid on every request. Always-FP wastes compute on the easy majority of inputs that PTQ already handles correctly; always-PTQ hurts quality disproportionately on the broken minority. Routing concentrates FP compute on the minority where it actually buys back accuracy, leaving the cheap PTQ path as the default. The pattern is most useful in services with fixed latency or cost budgets, in multi-tenant or bursty workloads where average-load reductions buy tail-latency headroom, and in two-tier setups (edge + server, or cheap + expensive accelerator) where the cheap path is the default and the expensive path is a selective fallback. Conceptually, the router is a systems specialization of selective prediction with deferral: trust a cheap default predictor on easy cases and escalate hard cases to a more reliable option — with the escalation target being full-precision inference rather than human review.

## Contribution

We frame the work as feasibility-then-refinement. First, we evaluate the canonical selective-prediction baseline — max-softmax-prob (MSP) — as a PTQ-routing signal, to our knowledge for the first time. It works well enough to establish that the problem is tractable. Second, we propose a 3-feature logistic regression on the quantized model's confidence signals (margin, max softmax, entropy) as a richer upgrade and measure when the richer predictor actually helps. Both predictors are evaluated under leave-one-out cross-task transfer (no per-task retraining), on three backbones spanning two modalities. The paper additionally characterizes when input-aware routing is worth doing at all — including a clean negative result on input-domain features and a careful diagnosis of when the routing claim narrows.

## Results

**MSP establishes feasibility.** Across three backbones spanning two modalities (ViT-B/16, ViT-L/16, and Qwen3 Embedding 0.6B), MSP — threshold the quantized model's top-1 softmax probability — recovers 90% of the FP-to-PTQ accuracy gap at roughly one-fifth to one-third routing budget, several times better than random. This validates that PTQ fragility is predictable from Q-side confidence on every backbone we tested.

**The 3-feature LogReg upgrades MSP, with diminishing returns at scale.** Adding the top-1/top-2 logit margin and the softmax entropy to MSP, combined in a per-task-z-scored logistic regression, improves the routing budget by ~6 pp on ViT-B/16 and ~4 pp on ViT-L/16, but ties MSP on Qwen3 (within noise). The shrinking improvement is mechanistically interpretable: as the backbone grows, the softmax distribution sharpens and the additional features collapse into monotonic transforms of the top-1 softmax. The result has practical content: ship the multi-feature LogReg on smaller vision backbones; ship MSP on the larger or NLP ones.

**It transfers without retraining.** A single classifier trained on pooled source tasks performs within a few percentage points of a target-task-specific classifier, on every individual task.

**It transfers across PTQ recipes.** The predictor mechanism is recipe-invariant under both per-channel and per-group_128 round-to-nearest quantization; the location of the "recoverable" regime depends on the recipe (better recipes push the boundary to lower bit-widths), but the predictor adapts to either boundary with no protocol change.

**Negative result on input-domain features.** Standard input statistics (image-pixel features for vision; tokenizer-side text statistics for NLP) carry no measurable predictive signal on any backbone. PTQ fragility is a model-level phenomenon, not a low-level input descriptor.

**Regime structure polarizes with scale.** As the backbone grows, the easy ("recoverable") regime gets easier and shrinks (fewer tasks have enough PTQ damage to route), while the hard ("catastrophic") regime, when present, gets harsher. This polarization is recipe-invariant.

## Caveats

**The deployable claim has scope qualifiers.** Routing is only useful where PTQ produces enough broken inputs for the accuracy recovery to be meaningful. We make this explicit by reporting an eligibility filter; tasks where PTQ barely degrades accuracy are excluded because routing them is uninformative.

**Online deployment narrows on the smallest vision backbone.** On ViT-B/16, label-free online deployment with a fixed threshold is unreliable because the per-task recovery metric is intrinsically noisy on tasks near the eligibility boundary. The diagnosis is that this reflects the eligible-task population, not the predictor; on the larger backbones (ViT-L, Qwen3) the same recipe works cleanly without labeled calibration.

**The "W3 is catastrophic" claim is recipe-specific.** Under naive per-channel round-to-nearest quantization, W3 produces a regime in which no input-aware routing recovers compute. Under per-group_128 quantization (a one-line change to the same naive method), most of that catastrophe disappears. We frame this as a recoverable-regime boundary that depends on the recipe rather than as a property of the bit-width.

**No comparison to state-of-the-art PTQ methods.** We test naive RTN at two granularities; methods that additionally modify the rounding (GPTQ, AWQ, SmoothQuant, QuaRot) would presumably extend the recoverable regime further, but are not tested.

**No wall-clock latency benchmark.** Compute savings are reported parametrically in the per-input PTQ-vs-FP speedup ratio, not as an end-to-end measurement on specific hardware.

**Single-seed.** The current numbers come from one finetune seed per backbone per task; multi-seed error bars are not reported.