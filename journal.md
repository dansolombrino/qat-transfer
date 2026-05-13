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
