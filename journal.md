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

# RepQ-ViT Baseline

**Status:** Implemented for vision models (ilharco_timm_supervised, ilharco_open_clip). TBD for text (ilharco_automodelforsequenceclassification) — RepQ-ViT targets ViT architectures and would not be a 1:1 application to text models.

## Experiment progress

- [ ] timm — all 7 models x 22 datasets (w_bits=3, a_bits=3, seed=2038)
- [ ] open_clip — ViT-B-16 / laion2b_s34b_b88k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
- [ ] open_clip — ViT-L-14 / laion2b_s32b_b82k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
- [ ] open_clip — ViT-H-14 / laion2b_s32b_b79k x 22 datasets (w_bits=3, a_bits=3, seed=2038)
