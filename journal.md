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
