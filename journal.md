# Models

## open_clip

| Model | Pretrained |
|---|---|
| ViT-B-16 | laion2b_s34b_b88k |
| ViT-L-14 | laion2b_s32b_b82k |
| ViT-H-14 | laion2b_s32b_b79k |

## timm

| Model | Pretrained |
|---|---|
| deit3_base_patch16_224 | fb_in1k |
| deit3_large_patch16_224 | fb_in1k |
| swin_base_patch4_window7_224 | ms_in22k_ft_in1k |
| swin_large_patch4_window7_224 | ms_in22k_ft_in1k |
| vit_base_patch16_224 | orig_in21k |
| vit_large_patch16_224 | orig_in21k |
| vit_huge_patch14_224 | orig_in21k |

---

# Competitors Overview

## Comparison table

| Method | Venue | Quant type | Bits | Data-free? | Scope |
|---|---|---|---|---|---|
| FT+PTQ (ours) | — | W-only | W3 | Yes | All models |
| QV Patching (ours) | — | W-only | W3 | Yes | All models |
| QAT+PTQ (ours) | — | W-only | W3 | No (full QAT) | All models |
| GPTQ | ICLR 2023 | W-only | W3 | No (128 samples) | All models |
| RepQ-ViT | ICCV 2023 | W+A | W3/A3 | No (32 images) | Vision only |
| MimiQ | AAAI 2025 | W+A | W3/A3 | Yes | Vision only |
| AdaLog | ECCV 2024 | W+A | W3/A3, W3/A8 | No (calib) | Vision only |
| APHQ-ViT | CVPR 2025 | W+A | W3/A3, W3/A8 | No (calib) | Vision only |
| DFQ-ViT | arXiv 2025 | W+A | W3/A8 | Yes | Vision only (blocked: no public code) |

Note: RepQ-ViT, MimiQ, AdaLog, APHQ-ViT, and DFQ-ViT quantize activations too (W+A), strictly harder than our W3-only setup. This favors QV patching in the comparison.

## What each competitor tests

| Competitor | Tests this advantage of QV patching |
|---|---|
| GPTQ | Zero-shot vs. calibration-based. Same quantization type (weight-only). |
| RepQ-ViT | Zero-shot vs. calibration-based. Simpler approach vs. ViT-specific engineering. |
| MimiQ | Both data-free. Weight-only simplicity vs. synthetic-data generation + W+A quantization. |
| AdaLog | Zero-shot vs. calibration-based. Simpler approach vs. adaptive log quantizer + block reconstruction. |
| APHQ-ViT | Zero-shot vs. calibration-based. Simpler approach vs. Hessian-based block reconstruction. |
| DFQ-ViT | Both data-free. Weight-only simplicity vs. synthetic-data + W+A quantization. |

## Methods considered but excluded

| Method | Why excluded |
|---|---|
| AWQ (MLSys 2024) | LLM decoder-only. No ViT/encoder support. |
| PTQ4ViT (ECCV 2022) | Lowest usable bit-width is W6/A6. No 3-bit results. |
| DFQ-ViT (2025) | Moved to competitor table (blocked: no public code). |
| CLAMP-ViT (ECCV 2024) | Data-free W+A, but no confirmed 3-bit results. |
| PSAQ-ViT (ECCV 2022) | Data-free W+A, but lowest is W4/A8. No 3-bit. |
| OBC/OBQ (NeurIPS 2022) | Predecessor to GPTQ; strictly dominated by it. |

**Note on PTQ4ViT:** Already implemented and tracked below (runs pending), but the brainstorming analysis flagged it as not having published 3-bit results — its lowest supported config is W6/A6. Keeping the existing runs for completeness but it may be dropped from the paper's competitor table.

## Competitor details

- **GPTQ:** code via AutoGPTQ / GPTQModel. Default is asymmetric per-group — configure to symmetric per-channel for apples-to-apples, or note the difference. Ref: Frantar et al., arXiv:2210.17323
- **MimiQ:** code at github.com/iamkanghyunchoi/mimiq. Data-free (generates synthetic calibration samples). Ref: Choi et al., arXiv:2407.20021
- **RepQ-ViT:** already implemented. code at github.com/zkkli/RepQ-ViT. Ref: Li et al., arXiv:2212.08254
- **AdaLog:** code at github.com/GoatWu/AdaLog. W+A PTQ with adaptive logarithm quantizer. Ref: Wu et al., ECCV 2024, arXiv:2407.12951
- **APHQ-ViT:** code at github.com/GoatWu/APHQ-ViT. W+A PTQ with Average Perturbation Hessian reconstruction. Ref: Wu et al., CVPR 2025, arXiv:2504.02508
- **DFQ-ViT:** no public code. Data-free W+A PTQ with E2H synthetic data generation + Activation Correction Matrix. Ref: arXiv:2507.14481

---

# RepQ-ViT Baseline

**Status:** Implemented for vision models (ilharco_timm_supervised, ilharco_open_clip). TBD for text (ilharco_automodelforsequenceclassification) — RepQ-ViT targets ViT architectures and would not be a 1:1 application to text models.

## Experiment progress

- [x] timm — all 7 models x 22 datasets (w_bits=3, a_bits=3, seed=2038)
      # DONE all models 4090
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_repqvit.py -m model_name=deit3_base_patch16_224.fb_in1k,deit3_large_patch16_224.fb_in1k,swin_base_patch4_window7_224.ms_in22k_ft_in1k,swin_large_patch4_window7_224.ms_in22k_ft_in1k,vit_base_patch16_224.orig_in21k,vit_large_patch16_224.orig_in21k,vit_huge_patch14_224.orig_in21k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 repqvit.w_bits=3 repqvit.a_bits=3 'repqvit.skip_modules=[head]' repqvit.calib_batch_size=128
      ```

- [x] open_clip — ViT-B-16 / laion2b_s34b_b88k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
      # DONE 3090 Ti
      ```
      uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp_repqvit.py -m model_name=ViT-B-16 pretrained=laion2b_s34b_b88k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 repqvit.w_bits=3 repqvit.a_bits=3 'repqvit.skip_modules=[classification_head]' repqvit.calib_batch_size=128
      ```

- [x] open_clip — ViT-L-14 / laion2b_s32b_b82k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
      # DONE 3090 Ti
      ```
      uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp_repqvit.py -m model_name=ViT-L-14 pretrained=laion2b_s32b_b82k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 repqvit.w_bits=3 repqvit.a_bits=3 'repqvit.skip_modules=[classification_head]' repqvit.calib_batch_size=128
      ```

- [x] open_clip — ViT-H-14 / laion2b_s32b_b79k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
      # DONE 3090 Ti
      ```
      uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp_repqvit.py -m model_name=ViT-H-14 pretrained=laion2b_s32b_b79k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 repqvit.w_bits=3 repqvit.a_bits=3 'repqvit.skip_modules=[classification_head]' repqvit.calib_batch_size=128
      ```

---

# REx Baseline

**Status:** Implemented for vision models (ilharco_timm_supervised, ilharco_open_clip). Uses residual expansion (REx) PTQ with separate correction terms (RexLinear / RexMultiheadAttention wrappers).

## Experiment progress

- [x] open_clip — ViT-B-16 / laion2b_s34b_b88k x 22 datasets (bits=3, channel, order=2, sparsity=0.125/0.25/0.5/0.75/1.0, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_rex.py -m model_name=ViT-B-16 pretrained=laion2b_s34b_b88k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 rex.bits=3 rex.granularity=channel rex.order=2 rex.sparsity=0.125,0.25,0.5,0.75,1.0 'rex.skip_modules=[classification_head]'
      ```
      # DONE 4090

- [x] open_clip — ViT-L-14 / laion2b_s32b_b82k x 22 datasets (bits=3, channel, order=2, sparsity=0.125/0.25/0.5/0.75/1.0, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_rex.py -m model_name=ViT-L-14 pretrained=laion2b_s32b_b82k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 rex.bits=3 rex.granularity=channel rex.order=2 rex.sparsity=0.125,0.25,0.5,0.75,1.0 'rex.skip_modules=[classification_head]'
      ```
      # DONE 4090

- [x] open_clip — ViT-H-14 / laion2b_s32b_b79k x 22 datasets (bits=3, channel, order=2, sparsity=0.125/0.25/0.5/0.75/1.0, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_rex.py -m model_name=ViT-H-14 pretrained=laion2b_s32b_b79k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 rex.bits=3 rex.granularity=channel rex.order=2 rex.sparsity=0.125,0.25,0.5,0.75,1.0 'rex.skip_modules=[classification_head]'
      ```
      # DONE 4090

- [x] timm — deit3_base_patch16_224.fb_in1k x 22 datasets (bits=3, channel, order=2, sparsity=0.125/0.25/0.5/0.75/1.0, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_rex.py -m model_name=deit3_base_patch16_224.fb_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 rex.bits=3 rex.granularity=channel rex.order=2 rex.sparsity=0.125,0.25,0.5,0.75,1.0 'rex.skip_modules=[head]'
      ```
      # DONE 3090 Ti

- [x] timm — deit3_large_patch16_224.fb_in1k x 22 datasets (bits=3, channel, order=2, sparsity=0.125/0.25/0.5/0.75/1.0, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_rex.py -m model_name=deit3_large_patch16_224.fb_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 rex.bits=3 rex.granularity=channel rex.order=2 rex.sparsity=0.125,0.25,0.5,0.75,1.0 'rex.skip_modules=[head]'
      ```
      # DONE 4090

- [x] timm — swin_base_patch4_window7_224.ms_in22k_ft_in1k x 22 datasets (bits=3, channel, order=2, sparsity=0.125/0.25/0.5/0.75/1.0, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_rex.py -m model_name=swin_base_patch4_window7_224.ms_in22k_ft_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 rex.bits=3 rex.granularity=channel rex.order=2 rex.sparsity=0.125,0.25,0.5,0.75,1.0 'rex.skip_modules=[head]'
      ```
      # DONE 3090 Ti

- [x] timm — swin_large_patch4_window7_224.ms_in22k_ft_in1k x 22 datasets (bits=3, channel, order=2, sparsity=0.125/0.25/0.5/0.75/1.0, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_rex.py -m model_name=swin_large_patch4_window7_224.ms_in22k_ft_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 rex.bits=3 rex.granularity=channel rex.order=2 rex.sparsity=0.125,0.25,0.5,0.75,1.0 'rex.skip_modules=[head]'
      ```
      # DONE 4090

- [x] timm — vit_base_patch16_224.orig_in21k x 22 datasets (bits=3, channel, order=2, sparsity=0.125/0.25/0.5/0.75/1.0, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_rex.py -m model_name=vit_base_patch16_224.orig_in21k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 rex.bits=3 rex.granularity=channel rex.order=2 rex.sparsity=0.125,0.25,0.5,0.75,1.0 'rex.skip_modules=[head]'
      ```
      # DONE 3090 Ti

- [x] timm — vit_large_patch16_224.orig_in21k x 22 datasets (bits=3, channel, order=2, sparsity=0.125/0.25/0.5/0.75/1.0, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_rex.py -m model_name=vit_large_patch16_224.orig_in21k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 rex.bits=3 rex.granularity=channel rex.order=2 rex.sparsity=0.125,0.25,0.5,0.75,1.0 'rex.skip_modules=[head]'
      ```
      # DONE 3090 Ti

- [x] timm — vit_huge_patch14_224.orig_in21k x 22 datasets (bits=3, channel, order=2, sparsity=0.125/0.25/0.5/0.75/1.0, seed=2038)
      # DONE 3090 Ti
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_rex.py model_name=vit_huge_patch14_224.orig_in21k dataset_name=ImageNet batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 rex.bits=3 rex.granularity=channel rex.order=2 rex.sparsity=1.0 'rex.skip_modules=[head]'
      ```

- [x] text — google-bert/bert-base-uncased x 11 datasets (bits=3, channel, order=2, sparsity=0.125/0.25/0.5/0.75/1.0, seed=2038)
      # 3090 Ti
      ```
      uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_rex.py -m model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 lr=1e-5 wd=0.1 ls=0.0 max_grad_norm=1.0 max_length=128 seed=2038 gpu=0 rex.bits=3 rex.granularity=channel rex.order=2 rex.sparsity=0.125,0.25,0.5,0.75,1.0 'rex.skip_modules=[classifier]'
      ```

- [x] text — google-bert/bert-large-uncased x 11 datasets (bits=3, channel, order=2, sparsity=0.125/0.25/0.5/0.75/1.0, seed=2038)
      # 3090 Ti
      ```
      uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_rex.py -m model_name=google-bert/bert-large-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 lr=1e-5 wd=0.1 ls=0.0 max_grad_norm=1.0 max_length=128 seed=2038 gpu=0 rex.bits=3 rex.granularity=channel rex.order=2 rex.sparsity=0.125,0.25,0.5,0.75,1.0 'rex.skip_modules=[classifier]'
      ```

- [x] text — google/embeddinggemma-300m x 11 datasets (bits=3, channel, order=2, sparsity=0.125/0.25/0.5/0.75/1.0, seed=2038)
      # 3090 Ti
      ```
      uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_rex.py -m model_name=google/embeddinggemma-300m dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 lr=1e-5 wd=0.1 ls=0.0 max_grad_norm=1.0 max_length=128 seed=2038 gpu=0 rex.bits=3 rex.granularity=channel rex.order=2 rex.sparsity=0.125,0.25,0.5,0.75,1.0 'rex.skip_modules=[score]'
      ```

- [x] text — Qwen/Qwen3-Embedding-0.6B x 11 datasets (bits=3, channel, order=2, sparsity=0.125/0.25/0.5/0.75/1.0, seed=2038)
      # 3090 Ti
      ```
      uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_rex.py -m model_name=Qwen/Qwen3-Embedding-0.6B dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 lr=1e-5 wd=0.1 ls=0.0 max_grad_norm=1.0 max_length=128 seed=2038 gpu=0 rex.bits=3 rex.granularity=channel rex.order=2 rex.sparsity=0.125,0.25,0.5,0.75,1.0 'rex.skip_modules=[score]'
      ```

---

# PTQ4ViT Baseline -- Excluded for now because it's not advised for 3 bits

**Status:** Implemented for vision models (ilharco_timm_supervised, ilharco_open_clip). Uses Hessian-guided PTQSL search for optimal quantization intervals with Split-of-Softmax (SoS) for attention score matrices and PostGeLU handling for MLP layers.

## Experiment progress

- [ ] open_clip — ViT-B-16 / laion2b_s34b_b88k x 22 datasets (bit=3, hessian, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_ptq4vit.py -m model_name=ViT-B-16 pretrained=laion2b_s34b_b88k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq4vit.bit=3 ptq4vit.calibrator=hessian ptq4vit.calib_num_samples=32 ptq4vit.calib_batch_size=32 ptq4vit.hessian_batch_size=4 ptq4vit.metric=hessian ptq4vit.search_round=3 ptq4vit.eq_alpha=0.01 ptq4vit.eq_beta=1.2 ptq4vit.eq_n=100 ptq4vit.n_V=1 ptq4vit.n_H=1 ptq4vit.n_a=1 ptq4vit.n_G_A=1 ptq4vit.n_V_A=1 ptq4vit.n_H_A=1 ptq4vit.n_G_B=1 ptq4vit.n_V_B=1 ptq4vit.n_H_B=1 'ptq4vit.skip_modules=[classification_head]'
      ```

- [ ] open_clip — ViT-L-14 / laion2b_s32b_b82k x 22 datasets (bit=3, hessian, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_ptq4vit.py -m model_name=ViT-L-14 pretrained=laion2b_s32b_b82k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq4vit.bit=3 ptq4vit.calibrator=hessian ptq4vit.calib_num_samples=32 ptq4vit.calib_batch_size=32 ptq4vit.hessian_batch_size=4 ptq4vit.metric=hessian ptq4vit.search_round=3 ptq4vit.eq_alpha=0.01 ptq4vit.eq_beta=1.2 ptq4vit.eq_n=100 ptq4vit.n_V=1 ptq4vit.n_H=1 ptq4vit.n_a=1 ptq4vit.n_G_A=1 ptq4vit.n_V_A=1 ptq4vit.n_H_A=1 ptq4vit.n_G_B=1 ptq4vit.n_V_B=1 ptq4vit.n_H_B=1 'ptq4vit.skip_modules=[classification_head]'
      ```

- [ ] open_clip — ViT-H-14 / laion2b_s32b_b79k x 22 datasets (bit=3, hessian, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_ptq4vit.py -m model_name=ViT-H-14 pretrained=laion2b_s32b_b79k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq4vit.bit=3 ptq4vit.calibrator=hessian ptq4vit.calib_num_samples=32 ptq4vit.calib_batch_size=32 ptq4vit.hessian_batch_size=4 ptq4vit.metric=hessian ptq4vit.search_round=3 ptq4vit.eq_alpha=0.01 ptq4vit.eq_beta=1.2 ptq4vit.eq_n=100 ptq4vit.n_V=1 ptq4vit.n_H=1 ptq4vit.n_a=1 ptq4vit.n_G_A=1 ptq4vit.n_V_A=1 ptq4vit.n_H_A=1 ptq4vit.n_G_B=1 ptq4vit.n_V_B=1 ptq4vit.n_H_B=1 'ptq4vit.skip_modules=[classification_head]'
      ```

- [ ] timm — deit3_base_patch16_224.fb_in1k x 22 datasets (bit=3, hessian, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_ptq4vit.py -m model_name=deit3_base_patch16_224.fb_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq4vit.bit=3 ptq4vit.calibrator=hessian ptq4vit.calib_num_samples=32 ptq4vit.calib_batch_size=32 ptq4vit.hessian_batch_size=4 ptq4vit.metric=hessian ptq4vit.search_round=3 ptq4vit.eq_alpha=0.01 ptq4vit.eq_beta=1.2 ptq4vit.eq_n=100 ptq4vit.n_V=1 ptq4vit.n_H=1 ptq4vit.n_a=1 ptq4vit.n_G_A=1 ptq4vit.n_V_A=1 ptq4vit.n_H_A=1 ptq4vit.n_G_B=1 ptq4vit.n_V_B=1 ptq4vit.n_H_B=1 'ptq4vit.skip_modules=[head]'
      ```

- [ ] timm — deit3_large_patch16_224.fb_in1k x 22 datasets (bit=3, hessian, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_ptq4vit.py -m model_name=deit3_large_patch16_224.fb_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq4vit.bit=3 ptq4vit.calibrator=hessian ptq4vit.calib_num_samples=32 ptq4vit.calib_batch_size=32 ptq4vit.hessian_batch_size=4 ptq4vit.metric=hessian ptq4vit.search_round=3 ptq4vit.eq_alpha=0.01 ptq4vit.eq_beta=1.2 ptq4vit.eq_n=100 ptq4vit.n_V=1 ptq4vit.n_H=1 ptq4vit.n_a=1 ptq4vit.n_G_A=1 ptq4vit.n_V_A=1 ptq4vit.n_H_A=1 ptq4vit.n_G_B=1 ptq4vit.n_V_B=1 ptq4vit.n_H_B=1 'ptq4vit.skip_modules=[head]'
      ```

- [ ] timm — swin_base_patch4_window7_224.ms_in22k_ft_in1k x 22 datasets (bit=3, hessian, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_ptq4vit.py -m model_name=swin_base_patch4_window7_224.ms_in22k_ft_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq4vit.bit=3 ptq4vit.calibrator=hessian ptq4vit.calib_num_samples=32 ptq4vit.calib_batch_size=32 ptq4vit.hessian_batch_size=4 ptq4vit.metric=hessian ptq4vit.search_round=3 ptq4vit.eq_alpha=0.01 ptq4vit.eq_beta=1.2 ptq4vit.eq_n=100 ptq4vit.n_V=1 ptq4vit.n_H=1 ptq4vit.n_a=1 ptq4vit.n_G_A=1 ptq4vit.n_V_A=1 ptq4vit.n_H_A=1 ptq4vit.n_G_B=1 ptq4vit.n_V_B=1 ptq4vit.n_H_B=1 'ptq4vit.skip_modules=[head]'
      ```

- [ ] timm — swin_large_patch4_window7_224.ms_in22k_ft_in1k x 22 datasets (bit=3, hessian, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_ptq4vit.py -m model_name=swin_large_patch4_window7_224.ms_in22k_ft_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq4vit.bit=3 ptq4vit.calibrator=hessian ptq4vit.calib_num_samples=32 ptq4vit.calib_batch_size=32 ptq4vit.hessian_batch_size=4 ptq4vit.metric=hessian ptq4vit.search_round=3 ptq4vit.eq_alpha=0.01 ptq4vit.eq_beta=1.2 ptq4vit.eq_n=100 ptq4vit.n_V=1 ptq4vit.n_H=1 ptq4vit.n_a=1 ptq4vit.n_G_A=1 ptq4vit.n_V_A=1 ptq4vit.n_H_A=1 ptq4vit.n_G_B=1 ptq4vit.n_V_B=1 ptq4vit.n_H_B=1 'ptq4vit.skip_modules=[head]'
      ```

- [ ] timm — vit_base_patch16_224.orig_in21k x 22 datasets (bit=3, hessian, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_ptq4vit.py -m model_name=vit_base_patch16_224.orig_in21k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq4vit.bit=3 ptq4vit.calibrator=hessian ptq4vit.calib_num_samples=32 ptq4vit.calib_batch_size=32 ptq4vit.hessian_batch_size=4 ptq4vit.metric=hessian ptq4vit.search_round=3 ptq4vit.eq_alpha=0.01 ptq4vit.eq_beta=1.2 ptq4vit.eq_n=100 ptq4vit.n_V=1 ptq4vit.n_H=1 ptq4vit.n_a=1 ptq4vit.n_G_A=1 ptq4vit.n_V_A=1 ptq4vit.n_H_A=1 ptq4vit.n_G_B=1 ptq4vit.n_V_B=1 ptq4vit.n_H_B=1 'ptq4vit.skip_modules=[head]'
      ```

- [ ] timm — vit_large_patch16_224.orig_in21k x 22 datasets (bit=3, hessian, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_ptq4vit.py -m model_name=vit_large_patch16_224.orig_in21k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq4vit.bit=3 ptq4vit.calibrator=hessian ptq4vit.calib_num_samples=32 ptq4vit.calib_batch_size=32 ptq4vit.hessian_batch_size=4 ptq4vit.metric=hessian ptq4vit.search_round=3 ptq4vit.eq_alpha=0.01 ptq4vit.eq_beta=1.2 ptq4vit.eq_n=100 ptq4vit.n_V=1 ptq4vit.n_H=1 ptq4vit.n_a=1 ptq4vit.n_G_A=1 ptq4vit.n_V_A=1 ptq4vit.n_H_A=1 ptq4vit.n_G_B=1 ptq4vit.n_V_B=1 ptq4vit.n_H_B=1 'ptq4vit.skip_modules=[head]'
      ```

- [ ] timm — vit_huge_patch14_224.orig_in21k x 22 datasets (bit=3, hessian, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_ptq4vit.py -m model_name=vit_huge_patch14_224.orig_in21k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq4vit.bit=3 ptq4vit.calibrator=hessian ptq4vit.calib_num_samples=32 ptq4vit.calib_batch_size=32 ptq4vit.hessian_batch_size=4 ptq4vit.metric=hessian ptq4vit.search_round=3 ptq4vit.eq_alpha=0.01 ptq4vit.eq_beta=1.2 ptq4vit.eq_n=100 ptq4vit.n_V=1 ptq4vit.n_H=1 ptq4vit.n_a=1 ptq4vit.n_G_A=1 ptq4vit.n_V_A=1 ptq4vit.n_H_A=1 ptq4vit.n_G_B=1 ptq4vit.n_V_B=1 ptq4vit.n_H_B=1 'ptq4vit.skip_modules=[head]'
      ```

---

# GPTQ Baseline

**Status:** Not yet implemented. Weight-only PTQ (layer-wise second-order optimal rounding). Applies to all models (vision + text). Code via AutoGPTQ / GPTQModel. Default is asymmetric per-group — configure to symmetric per-channel for apples-to-apples with our setup, or note the difference. Uses 128 calibration samples (not zero-shot). Ref: Frantar et al., arXiv:2210.17323

## Experiment progress

- [ ] timm — all 7 models x 22 datasets (bits=3, seed=2038)
- [ ] open_clip — ViT-B-16 / laion2b_s34b_b88k x 22 datasets (bits=3, seed=2038)
- [ ] open_clip — ViT-L-14 / laion2b_s32b_b82k x 22 datasets (bits=3, seed=2038)
- [ ] open_clip — ViT-H-14 / laion2b_s32b_b79k x 22 datasets (bits=3, seed=2038)
- [ ] text — google-bert/bert-base-uncased x 11 datasets (bits=3, seed=2038)
- [ ] text — google-bert/bert-large-uncased x 11 datasets (bits=3, seed=2038)
- [ ] text — google/embeddinggemma-300m x 11 datasets (bits=3, seed=2038)
- [ ] text — Qwen/Qwen3-Embedding-0.6B x 11 datasets (bits=3, seed=2038)

---

# MimiQ Baseline

**Status:** Not yet implemented. Data-free W+A PTQ for ViTs (generates synthetic calibration data via inter-head attention similarity). Vision-only — does not apply to text encoders. Run at W3/A3. Code at github.com/iamkanghyunchoi/mimiq. Ref: Choi et al., arXiv:2407.20021

## Experiment progress

- [ ] timm — all 7 models x 22 datasets (w_bits=3, a_bits=3, seed=2038)
- [ ] open_clip — ViT-B-16 / laion2b_s34b_b88k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
- [ ] open_clip — ViT-L-14 / laion2b_s32b_b82k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
- [ ] open_clip — ViT-H-14 / laion2b_s32b_b79k x 22 datasets (w_bits=3, a_bits=3, seed=2038)

---

# AdaLog Baseline

**Status:** Not yet implemented. W+A PTQ with adaptive logarithm quantizer and Fast Progressive Combining Search (FPCS). Vision-only. Run at both W3/A3 and W3/A8. Code at github.com/GoatWu/AdaLog. Ref: Wu et al., ECCV 2024, arXiv:2407.12951

## Experiment progress

- [ ] timm — all 7 models x 22 datasets (w_bits=3, a_bits=3, seed=2038)
- [ ] timm — all 7 models x 22 datasets (w_bits=3, a_bits=8, seed=2038)
- [ ] open_clip — ViT-B-16 / laion2b_s34b_b88k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
- [ ] open_clip — ViT-B-16 / laion2b_s34b_b88k x 22 datasets (w_bits=3, a_bits=8, seed=2038)
- [ ] open_clip — ViT-L-14 / laion2b_s32b_b82k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
- [ ] open_clip — ViT-L-14 / laion2b_s32b_b82k x 22 datasets (w_bits=3, a_bits=8, seed=2038)
- [ ] open_clip — ViT-H-14 / laion2b_s32b_b79k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
- [ ] open_clip — ViT-H-14 / laion2b_s32b_b79k x 22 datasets (w_bits=3, a_bits=8, seed=2038)

---

# APHQ-ViT Baseline

**Status:** Implemented for vision models (ilharco_timm_supervised). TBD for ilharco_open_clip. W+A PTQ with Average Perturbation Hessian (APH) based importance estimation and MLP Reconstruction. Vision-only. Three-stage pipeline reproduced faithfully from the reference: (1) MLP reconstruction (GELU→ReLU + Hessian-perturb fine-tune, toggleable via `aphq_vit.reconstruct_mlp`), (2) wrap modules with APHQ-ViT quant layers + MSE-guided calibration (always on), (3) AdaRound block reconstruction with `hessian_perturb` metric + QDrop (toggleable via `aphq_vit.optimize`). Default config (both toggles on) reproduces the paper's headline W4/A4 / W3/A3 numbers. Architecture support extended beyond the reference's `model_zoo` to include the project's `deit3_*`, `vit_huge_patch14_224`, and `swin_large_patch4_window7_224` variants (same `timm.models.vision_transformer.VisionTransformer` / `swin_transformer.SwinTransformer` families with the same `Attention` / `WindowAttention` hookable modules); model-name matching strips the `.<pretrained-tag>` suffix (e.g. `.orig_in21k`, `.fb_in1k`, `.ms_in22k_ft_in1k`) so all current timm checkpoints validate. See `SUPPORTED_MODELS` in `code/src/aphq_vit/adapter.py`. Run at both W3/A3 and W3/A8. Code at github.com/GoatWu/APHQ-ViT. Ref: Wu et al., CVPR 2025, arXiv:2504.02508

## Experiment progress

- [ ] timm — deit3_base_patch16_224.fb_in1k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_aphq_vit.py -m model_name=deit3_base_patch16_224.fb_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 aphq_vit.w_bit=3 aphq_vit.a_bit=3 aphq_vit.qhead_a_bit=3 'aphq_vit.skip_modules=[head]'
      ```

- [ ] timm — deit3_base_patch16_224.fb_in1k x 22 datasets (w_bits=3, a_bits=8, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_aphq_vit.py -m model_name=deit3_base_patch16_224.fb_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 aphq_vit.w_bit=3 aphq_vit.a_bit=8 aphq_vit.qhead_a_bit=8 'aphq_vit.skip_modules=[head]'
      ```

- [ ] timm — deit3_large_patch16_224.fb_in1k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_aphq_vit.py -m model_name=deit3_large_patch16_224.fb_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 aphq_vit.w_bit=3 aphq_vit.a_bit=3 aphq_vit.qhead_a_bit=3 'aphq_vit.skip_modules=[head]'
      ```

- [ ] timm — deit3_large_patch16_224.fb_in1k x 22 datasets (w_bits=3, a_bits=8, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_aphq_vit.py -m model_name=deit3_large_patch16_224.fb_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 aphq_vit.w_bit=3 aphq_vit.a_bit=8 aphq_vit.qhead_a_bit=8 'aphq_vit.skip_modules=[head]'
      ```

- [ ] timm — swin_base_patch4_window7_224.ms_in22k_ft_in1k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_aphq_vit.py -m model_name=swin_base_patch4_window7_224.ms_in22k_ft_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 aphq_vit.w_bit=3 aphq_vit.a_bit=3 aphq_vit.qhead_a_bit=3 'aphq_vit.skip_modules=[head]'
      ```

- [ ] timm — swin_base_patch4_window7_224.ms_in22k_ft_in1k x 22 datasets (w_bits=3, a_bits=8, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_aphq_vit.py -m model_name=swin_base_patch4_window7_224.ms_in22k_ft_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 aphq_vit.w_bit=3 aphq_vit.a_bit=8 aphq_vit.qhead_a_bit=8 'aphq_vit.skip_modules=[head]'
      ```

- [ ] timm — swin_large_patch4_window7_224.ms_in22k_ft_in1k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_aphq_vit.py -m model_name=swin_large_patch4_window7_224.ms_in22k_ft_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 aphq_vit.w_bit=3 aphq_vit.a_bit=3 aphq_vit.qhead_a_bit=3 'aphq_vit.skip_modules=[head]'
      ```

- [ ] timm — swin_large_patch4_window7_224.ms_in22k_ft_in1k x 22 datasets (w_bits=3, a_bits=8, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_aphq_vit.py -m model_name=swin_large_patch4_window7_224.ms_in22k_ft_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 aphq_vit.w_bit=3 aphq_vit.a_bit=8 aphq_vit.qhead_a_bit=8 'aphq_vit.skip_modules=[head]'
      ```

- [ ] timm — vit_base_patch16_224.orig_in21k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_aphq_vit.py -m model_name=vit_base_patch16_224.orig_in21k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 aphq_vit.w_bit=3 aphq_vit.a_bit=3 aphq_vit.qhead_a_bit=3 'aphq_vit.skip_modules=[head]'
      ```

- [ ] timm — vit_base_patch16_224.orig_in21k x 22 datasets (w_bits=3, a_bits=8, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_aphq_vit.py -m model_name=vit_base_patch16_224.orig_in21k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 aphq_vit.w_bit=3 aphq_vit.a_bit=8 aphq_vit.qhead_a_bit=8 'aphq_vit.skip_modules=[head]'
      ```

- [ ] timm — vit_large_patch16_224.orig_in21k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_aphq_vit.py -m model_name=vit_large_patch16_224.orig_in21k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 aphq_vit.w_bit=3 aphq_vit.a_bit=3 aphq_vit.qhead_a_bit=3 'aphq_vit.skip_modules=[head]'
      ```

- [ ] timm — vit_large_patch16_224.orig_in21k x 22 datasets (w_bits=3, a_bits=8, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_aphq_vit.py -m model_name=vit_large_patch16_224.orig_in21k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 aphq_vit.w_bit=3 aphq_vit.a_bit=8 aphq_vit.qhead_a_bit=8 'aphq_vit.skip_modules=[head]'
      ```

- [ ] timm — vit_huge_patch14_224.orig_in21k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_aphq_vit.py -m model_name=vit_huge_patch14_224.orig_in21k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 aphq_vit.w_bit=3 aphq_vit.a_bit=3 aphq_vit.qhead_a_bit=3 'aphq_vit.skip_modules=[head]'
      ```

- [ ] timm — vit_huge_patch14_224.orig_in21k x 22 datasets (w_bits=3, a_bits=8, seed=2038)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_aphq_vit.py -m model_name=vit_huge_patch14_224.orig_in21k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 aphq_vit.w_bit=3 aphq_vit.a_bit=8 aphq_vit.qhead_a_bit=8 'aphq_vit.skip_modules=[head]'
      ```

- [ ] open_clip — ViT-B-16 / laion2b_s34b_b88k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
- [ ] open_clip — ViT-B-16 / laion2b_s34b_b88k x 22 datasets (w_bits=3, a_bits=8, seed=2038)
- [ ] open_clip — ViT-L-14 / laion2b_s32b_b82k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
- [ ] open_clip — ViT-L-14 / laion2b_s32b_b82k x 22 datasets (w_bits=3, a_bits=8, seed=2038)
- [ ] open_clip — ViT-H-14 / laion2b_s32b_b79k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
- [ ] open_clip — ViT-H-14 / laion2b_s32b_b79k x 22 datasets (w_bits=3, a_bits=8, seed=2038)

---

# DFQ-ViT Baseline

**Status:** Blocked — no public code available. Data-free W+A PTQ (W3/A8) with Easy-to-Hard (E2H) synthetic data generation and Activation Correction Matrix (ACM). Vision-only. Will implement if/when code is released. Ref: arXiv:2507.14481

## Experiment progress

(Blocked on public code release)

---

# 002 QV Transfer Reversed (timm)

**Status:** Implemented. Reversed QV transfer: computes QV = QAT_src - FP_src (same as 001), but applies it in the opposite direction to the PTQ of the QAT target checkpoint:

```
patched = PTQ(QAT_{S2,Q}^{D2}) - alpha * QV
```

Uses the same `cfg.ptq` config for both the PTQ(QAT_tgt) base construction and the final post-patching PTQ evaluation. Alpha = 1.00. Evaluates with both FP and QAT target heads, with and without PTQ.

## Experiment progress

- [ ] timm — deit3_base_patch16_224.fb_in1k x 22 datasets (bits=3, channel, skip=head, seed=2038, eval_split=test)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/002z_qat_transfer_reversed_ptq_after_reverse/qv_transfer.py -m model_name=deit3_base_patch16_224.fb_in1k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.00 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' eval_split=test gpu=0
      ```

- [ ] timm — deit3_large_patch16_224.fb_in1k x 22 datasets (bits=3, channel, skip=head, seed=2038, eval_split=test)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/002z_qat_transfer_reversed_ptq_after_reverse/qv_transfer.py -m model_name=deit3_large_patch16_224.fb_in1k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.00 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' eval_split=test gpu=0
      ```

- [ ] timm — swin_base_patch4_window7_224.ms_in22k_ft_in1k x 22 datasets (bits=3, channel, skip=head, seed=2038, eval_split=test)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/002z_qat_transfer_reversed_ptq_after_reverse/qv_transfer.py -m model_name=swin_base_patch4_window7_224.ms_in22k_ft_in1k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.00 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' eval_split=test gpu=0
      ```

- [ ] timm — swin_large_patch4_window7_224.ms_in22k_ft_in1k x 22 datasets (bits=3, channel, skip=head, seed=2038, eval_split=test)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/002z_qat_transfer_reversed_ptq_after_reverse/qv_transfer.py -m model_name=swin_large_patch4_window7_224.ms_in22k_ft_in1k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.00 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' eval_split=test gpu=0
      ```

- [ ] timm — vit_base_patch16_224.orig_in21k x 22 datasets (bits=3, channel, skip=head, seed=2038, eval_split=test)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/002z_qat_transfer_reversed_ptq_after_reverse/qv_transfer.py -m model_name=vit_base_patch16_224.orig_in21k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.00 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' eval_split=test gpu=0
      ```

- [ ] timm — vit_large_patch16_224.orig_in21k x 22 datasets (bits=3, channel, skip=head, seed=2038, eval_split=test)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/002z_qat_transfer_reversed_ptq_after_reverse/qv_transfer.py -m model_name=vit_large_patch16_224.orig_in21k batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.00 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' eval_split=test gpu=0
      ```

- [ ] timm — vit_huge_patch14_224.orig_in21k x 22 datasets (bits=3, channel, skip=head, seed=2038, eval_split=test)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/002z_qat_transfer_reversed_ptq_after_reverse/qv_transfer.py -m model_name=vit_huge_patch14_224.orig_in21k batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.00 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' eval_split=test gpu=0
      ```

---

# 2026-07-30/31 — bitwidth sweep for the rebuttal: 4-bit and 2-bit

Reviewer question: does QV transfer hold at bitwidths other than 3? Everything in
the paper was `qat.bits=3 / ptq.bits=3, channel`. Decision: **matched-bit**
(train QAT at B bits, deploy PTQ at B bits), α = 1.0, `split=test` only.

Fixed throughout: `seed=2038`, `granularity=channel`.
Vision: `vit_base_patch16_224.orig_in21k`, 22 datasets, `bs=128`, `skip=[head]`.
Text: 4 models × 11 datasets, `bs=32`, `max_length=128`, no warmup.
  `google-bert/bert-base-uncased`, `google-bert/bert-large-uncased` → `skip=[classifier]`
  `google/embeddinggemma-300m`, `Qwen/Qwen3-Embedding-0.6B` → `skip=[score]`
  (AmazonPolarity excluded — commented out of `DATASET_NAME_TO_EPOCHS`.)

## What was RUN

| Modality | Bits | Stage 1 QAT | Stage 2 gate | Stage 3 grid (α=1, test) |
|---|---|---|---|---|
| Vision (timm, 1 backbone) | **4** | 22 runs ✅ | 22 rows ✅ | 22×22 = **484 cells ✅ verified** |
| Text (4 models) | **4** | 44 runs ✅ | 44 rows ✅ | 11×11 ×4 = **484 cells ✅ verified** (121 per model) |
| Vision (timm, 1 backbone) | **2** | 22 runs ✅ | 22 rows ✅ | 22×22 = 484 pairs — **464 on disk**, 20 missing |
| Text (4 models) | **2** | 22 runs ✅ (BERTs only) | 44 rows ✅ | 11×11 ×3 = 363 pairs — **215 on disk**, 148 missing |

Counts above are `find`-verified against the local `evaluations/` tree on 2026-07-31,
not inferred from what was dispatched. Re-verify the same way rather than trusting
the ✅: everything here is written by whichever rig ran the job, so a row can read
"done" locally while a third of it still sits on behemoth or the 3090 Ti.

### 2026-07-31 — text 4-bit grid repair

The text 4-bit grid was **not** the 484 cells this table originally claimed. Two
separate problems, both now fixed:

1. **Never rsynced.** Only 125 of 484 cells were on the 4090; the rest were still
   on behemoth / the 3090 Ti. `scripts/dispatch/gather_results.sh` blocks forever on
   a `RECONCILE COMPLETE` marker in a dead session's scratchpad log, so it never ran
   its two rsync lines. Running those two lines by hand brought the tree to 461.
2. **23 cells genuinely never evaluated.** The gaps were not random — they were one
   receiver per model, truncated mid-sweep (the signature of a runner killed inside
   its inner donor loop):
   - `bert-large-uncased`, `tgt=ToxicConversations` — 9 donors missing
   - `embeddinggemma-300m`, `tgt=IMDB` — 9 donors missing
   - `Qwen3-Embedding-0.6B`, `tgt=ToxicConversations` — 5 donors missing

   All FP and 4-bit QAT checkpoints for those receivers were intact on behemoth, so
   this was 3 `runners/t_qv.sh` invocations (GPUs 4, 5, 2), no retraining. Grid is
   now 121/121 per model = **484**.

Note there is **no `split=val` anywhere in the 4-bit tree** (by design — see "What
was deliberately NOT run"). Any best-α / λ* figure is therefore impossible at 4 bits;
only fixed α=1.0 on `split=test` can be plotted.

## What was REUSED, not re-run

- vision `fp_ptq` + `pretrained_ptq` @ 2 bits — 22 each, pre-existing
- text 2-bit QAT checkpoints for `embeddinggemma-300m` and `Qwen3-Embedding-0.6B` — 11 each
- text matched 2-bit grid for `Qwen3-Embedding-0.6B` — 121 pairs

## What was deliberately NOT run

- **α / λ* sweeps at either bitwidth.** The 3-bit protocol sweeps 11 α values
  (vision) / 40 (text) on `split=val` then reports λ* on test. Skipped: at 4 bits
  Δ_ceiling averages +1.08, so tuning λ to recover a ~1-point ceiling is not worth
  ~2.6× the α=1 cost. Only `qv.alpha=1.0` (the data-free setting) was run.
- 3-bit anything — already on disk.
- Other backbones / families (open_clip, hf_clip, the frozen-ablation timm variant).

## Headline result: the 4-bit Δ_ceiling gate (22/22 vision datasets)

`Δ_ceiling = acc(ptq(QAT)) − acc(ptq(FP))`, the receiver's own QAT gain.

|  | 3-bit | 4-bit |
|---|---|---|
| mean Δ_ceiling | **+45.26** | **+1.08** |
| range | +2.42 … +83.70 | −1.01 … +5.08 |
| datasets with Δ ≤ 0 | 0 | 2 |
| datasets with Δ > 2 pts | 22 | 2 |

At 3 bits PTQ collapses the model and QAT rescues it; at 4 bits PTQ is already
near-lossless so there is almost nothing for a QV to recover. Recovery ratio
`Δ/Δ_ceiling` is ill-conditioned at 4 bits — two denominators are negative.

**Mechanism, visible across all 22:** the only two datasets with real 4-bit
headroom (KMNIST +5.08, EMNIST +4.15) are exactly the two where 4-bit PTQ still
degrades the model (82.67 / 76.41 post-PTQ, vs 89–98 elsewhere). The size of the
QAT benefit tracks how much quantization actually hurts. That reframes the 4-bit
result from "the method fails at 4 bits" to "at 4 bits there is nothing to
recover, and where there is, the method still has room" — a scope boundary with a
mechanism, not a null result.

Suggested framing: the **baseline Δ_ceiling table is the stronger rebuttal
artifact**, with the α=1 grid reported beneath it.

## Infrastructure notes

Ran across behemoth (4× RTX PRO 6000 Blackwell, GPUs 0/2/6/7), rig-4090,
rig-3090-ti. Traps hit and their fixes are captured in
`.claude/skills/multi-rig-dispatch/SKILL.md`; helper scripts in
`scripts/dispatch/` (`rssh.sh`, `reconcile.sh`, `gather_results.sh`,
`final_repair.sh`, `two_bit_pipeline.sh`, `runners/`).

Two worth repeating here:
- behemoth needs **cu129 torch** (`sm_120`); the pinned cu126 build stops at
  `sm_90`. Never run `uv run`/`uv sync` there — use `.venv/bin/python`.
- An ImageNet QAT epoch took **24 h on the 4090** (32 cores, 16 workers) and
  **55 min on behemoth** with `TORCH_NUM_WORKERS=96`. Dataloader-bound, not
  GPU-bound.
