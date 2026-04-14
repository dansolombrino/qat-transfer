# qat-transfer

All commands below assume the repo root as the working directory and the project `.venv` managed by `uv`. Every Hydra script resolves config search paths from `${oc.env:PWD}`, so **you must launch from the repo root** — not from inside `code/` or `config/`.

Scripts come in two families:
- **Hydra scripts** — everything under `code/src/vision/` and `code/experiments/vision/`. They support three launch modes: single local run, local *sequential* sweep (Hydra's basic launcher), and Slurm *parallel* sweep (submitit launcher).
- **Argparse scripts** — the plotting utilities under `code/visualizations/`. Plain CLI, no Hydra.

The full list of supported datasets (used in every sweep example below) is defined in [code/src/vision/data/common.py](code/src/vision/data/common.py#L35):

```
Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet
```

---

## Launch modes for Hydra scripts

For **every** Hydra script listed below, three launch modes are available. They only differ in the wrapper around the command; the config overrides are identical.

### 1. Single local run (no `-m`)
Runs in the current process. Good for debugging on a workstation / login node.
```
uv run --active python <script.py> <overrides...>
```

### 2. Local sequential sweep (`-m`, no launcher override)
Hydra's default `basic` launcher. Every combination of swept values runs **sequentially in the same process**, one after the other. Use this when you want a sweep on a single local GPU without Slurm.
```
uv run --active python <script.py> -m <overrides...> key=a,b,c other=x,y
```

### 3. Submitit parallel sweep (`-m hydra/launcher=submitit_slurm`)
Every combination of swept values is dispatched as a **separate Slurm job** via `hydra-submitit-launcher`. Jobs run in parallel subject to Slurm scheduling. Also usable for a single run (no comma-separated values) to get one Slurm job instead of a local run.
```
uv run --active python <script.py> -m hydra/launcher=submitit_slurm <overrides...> key=a,b,c
```
The launcher config lives at [config/hydra/launcher/submitit_slurm.yaml](config/hydra/launcher/submitit_slurm.yaml) (partition, account, GPUs, CPUs, mem, walltime). Override any field on the CLI, e.g. `hydra.launcher.timeout_min=240 hydra.launcher.mem_gb=64`.

> Modes 2 and 3 both use `-m`. The **only** difference is whether you add `hydra/launcher=submitit_slurm`. Without it, Hydra's basic launcher runs the sweep sequentially in-process; with it, submitit fans the sweep out to Slurm.

---

# ilharco_hf_clip

## Finetuning

### FP finetuning — [code/src/vision/ilharco_hf_clip/finetune_fp.py](code/src/vision/ilharco_hf_clip/finetune_fp.py)
Config: [config/src/vision/ilharco_hf_clip/finetune_fp.yaml](config/src/vision/ilharco_hf_clip/finetune_fp.yaml)

Single local run:
```
uv run --active python code/src/vision/ilharco_hf_clip/finetune_fp.py model_name=openai/clip-vit-base-patch16 dataset_name=CIFAR10 seed=2038
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_hf_clip/finetune_fp.py -m model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_hf_clip/finetune_fp.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3
```

### QAT finetuning — [code/src/vision/ilharco_hf_clip/finetune_qat.py](code/src/vision/ilharco_hf_clip/finetune_qat.py)
Config: [config/src/vision/ilharco_hf_clip/finetune_qat.yaml](config/src/vision/ilharco_hf_clip/finetune_qat.yaml). Requires `qat.bits`, `qat.granularity`, `qat.skip_modules`.

Single local run:
```
uv run --active python code/src/vision/ilharco_hf_clip/finetune_qat.py model_name=openai/clip-vit-base-patch16 dataset_name=CIFAR10 seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_hf_clip/finetune_qat.py -m model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_hf_clip/finetune_qat.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

> The user's feedback memory notes: `finetune_*` scripts are GPU-intensive and long-running — do not launch them casually.

---

## 000_baselines — evaluation scripts

All scripts live in [code/experiments/vision/ilharco_hf_clip/000_baselines/](code/experiments/vision/ilharco_hf_clip/000_baselines/) with matching configs in [config/experiments/vision/ilharco_hf_clip/000_baselines/](config/experiments/vision/ilharco_hf_clip/000_baselines/).

### evaluate_pretrained — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.py](code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.yaml](config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.yaml)

Single local run:
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.py model_name=openai/clip-vit-base-patch16 dataset_name=CIFAR10 batch_size=128 seed=2038
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.py -m model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 seed=1,2,3
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 seed=1,2,3
```

### evaluate_fp — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.py](code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.yaml](config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.yaml)

Single local run:
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.py model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.py -m model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3
```

### evaluate_fp_ptq — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.py](code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.yaml](config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.yaml). Adds a `ptq` group.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.py model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.py -m model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

### evaluate_fp_ptq_bias_norm_emb_from_pt — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.py](code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.yaml](config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.yaml). Same parameters as `evaluate_fp_ptq`. Before applying PTQ, swaps biases, layer-norm weights, and embedding weights back to pretrained values.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.py model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.py -m model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

### evaluate_qat — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.py](code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.yaml](config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.yaml). The `qat.*` group must match the training config of the checkpoint being evaluated.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.py model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.py -m model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

### evaluate_qat_ptq — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.py](code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.yaml](config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.yaml). Has both a `qat` group (to locate the checkpoint) and a `ptq` group (for the PTQ applied at eval time).

Single local run:
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.py model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.py -m model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

---

## 001_qat_transfer — QV transfer experiment

### qv_transfer — [code/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.py](code/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.py)
Config: [config/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.yaml](config/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.yaml)

Requires three checkpoints to already exist on disk: `FP_source`, `QAT_source`, `FP_target`. Defines a QV = `QAT_source - FP_source`, patches it into `FP_target` at scale `qv.alpha`, then applies PTQ and evaluates.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.py model_name=openai/clip-vit-base-patch16 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 source.dataset_name=CIFAR10 source.seed=2038 target.dataset_name=CIFAR100 target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' qv.alpha=1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

Local sequential sweep (all source × all target datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.py -m model_name=openai/clip-vit-base-patch16 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 source.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet source.seed=2038 target.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' qv.alpha=0.25,0.5,0.75,1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

Submitit parallel sweep (all source × all target datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 source.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet source.seed=2038 target.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' qv.alpha=0.25,0.5,0.75,1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

---

## Visualizations (argparse, no Hydra, no Slurm)

These scripts read the JSON results produced by the experiments above and render plots. They use plain `argparse`, so Hydra sweeps and submitit do not apply — just call them one at a time (or wrap them in a shell `for` loop for a local sequential sweep).

### qv_transfer_heatmap — [code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap.py](code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap.py)
```
uv run --active python code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap.py --model-name openai/clip-vit-base-patch16 --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --qv-alpha 1.0
```

### qv_transfer_heatmap_best_sf — [code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap_best_sf.py](code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap_best_sf.py)
```
uv run --active python code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap_best_sf.py --model-name openai/clip-vit-base-patch16 --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --qv-alpha 1.0
```

Local sequential loop over `qv.alpha` values:
```
for alpha in 0.25 0.5 0.75 1.0; do uv run --active python code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap.py --model-name openai/clip-vit-base-patch16 --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --qv-alpha $alpha; done
```

### weights_candlestick_comparison notebook — [code/visualizations/vision/ilharco_hf_clip/weights_candlestick_comparison.ipynb](code/visualizations/vision/ilharco_hf_clip/weights_candlestick_comparison.ipynb)
Interactive notebook for side-by-side **layer weight distribution candlesticks** between two model sources (HF model IDs or local `.pt` checkpoints/checkpoint directories).  
Set `MODEL_A` and `MODEL_B` in the config cell, then run all cells. The notebook includes:
- layer-level view (all parameter blocks aggregated per layer index)
- granular view across all layers where each x-tick is a `(layer, component)` pair (e.g. `L0:attention`, `L0:mlp`, ..., `L11:attention`)

Candlesticks show:
- min/max wick excluding outliers
- mean line
- mean ± 1σ body
- outlier points beyond 2σ

Layer-level stats aggregate all parameter blocks that share the same layer index.

---

# ilharco_timm_supervised

## Finetuning

### FP finetuning — [code/src/vision/ilharco_timm_supervised/finetune_fp.py](code/src/vision/ilharco_timm_supervised/finetune_fp.py)
Config: [config/src/vision/ilharco_timm_supervised/finetune_fp.yaml](config/src/vision/ilharco_timm_supervised/finetune_fp.yaml)

Single local run:
```
uv run --active python code/src/vision/ilharco_timm_supervised/finetune_fp.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 seed=2038
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_timm_supervised/finetune_fp.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_timm_supervised/finetune_fp.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3
```

### QAT finetuning — [code/src/vision/ilharco_timm_supervised/finetune_qat.py](code/src/vision/ilharco_timm_supervised/finetune_qat.py)
Config: [config/src/vision/ilharco_timm_supervised/finetune_qat.yaml](config/src/vision/ilharco_timm_supervised/finetune_qat.yaml). Requires `qat.bits`, `qat.granularity`, `qat.skip_modules`.

Single local run:
```
uv run --active python code/src/vision/ilharco_timm_supervised/finetune_qat.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_timm_supervised/finetune_qat.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_timm_supervised/finetune_qat.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[head]'
```

---

## 000_baselines — evaluation scripts

All scripts live in [code/experiments/vision/ilharco_timm_supervised/000_baselines/](code/experiments/vision/ilharco_timm_supervised/000_baselines/) with matching configs in [config/experiments/vision/ilharco_timm_supervised/000_baselines/](config/experiments/vision/ilharco_timm_supervised/000_baselines/).

### evaluate_pretrained — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.py](code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.yaml](config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.yaml). Unlike the hf_clip variant, requires training hyperparameters (`lr`, `wd`, `ls`, `wl`, `max_grad_norm`) to locate the finetuned head checkpoint.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3
```

### evaluate_pretrained_ptq — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.py](code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.yaml](config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.yaml). Evaluates the pretrained backbone after applying PTQ. Has both `qat.*` (to locate the QAT-trained head checkpoint) and `ptq.*` groups.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[head]'
```

### evaluate_fp — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.py](code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.yaml](config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.yaml)

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3
```

### evaluate_fp_ptq — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.py](code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.yaml](config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.yaml). Adds a `ptq` group.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[head]'
```

### evaluate_qat — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.py](code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.yaml](config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.yaml). The `qat.*` group must match the training config of the checkpoint being evaluated.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[head]'
```

### evaluate_qat_ptq — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.py](code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.yaml](config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.yaml). Has both a `qat` group (to locate the checkpoint) and a `ptq` group (for the PTQ applied at eval time).

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[head]'
```

---

## 001_qat_transfer — QV transfer experiment

### qv_transfer — [code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.py](code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.py)
Config: [config/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.yaml](config/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.yaml)

Same concept as the hf_clip variant but uses `source.dataset_names` (plural — a list of source datasets iterated internally by the script) instead of `source.dataset_name`.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[CIFAR10]' source.seed=2038 target.dataset_name=CIFAR100 target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]'
```

Local sequential sweep (all source × all target datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 target.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=0.25,0.5,0.75,1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]'
```

Submitit parallel sweep (all source × all target datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 target.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=0.25,0.5,0.75,1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]'
```

---

## Visualizations (argparse, no Hydra, no Slurm)

### qv_transfer_heatmap — [code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap.py](code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap.py)
```
uv run --active python code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap.py --model-name vit_base_patch16_clip_224.openai_ft_in12k_in1k --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules head --qv-alpha 1.0
```

### qv_transfer_heatmap_best_sf — [code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap_best_sf.py](code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap_best_sf.py)
```
uv run --active python code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap_best_sf.py --model-name vit_base_patch16_clip_224.openai_ft_in12k_in1k --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules head --qv-alpha 1.0
```

Local sequential loop over `qv.alpha` values:
```
for alpha in 0.25 0.5 0.75 1.0; do uv run --active python code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap.py --model-name vit_base_patch16_clip_224.openai_ft_in12k_in1k --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules head --qv-alpha $alpha; done
```

---

# Notes

- **Always launch from the repo root.** Hydra config search paths use `${oc.env:PWD}` and will not resolve otherwise.
- **List-valued overrides** (e.g. `skip_modules`) must be quoted to protect brackets from the shell: `'qat.skip_modules=[classification_head]'`.
- **Tuning the Slurm job** without editing YAML: append `hydra.launcher.timeout_min=...`, `hydra.launcher.mem_gb=...`, `hydra.launcher.cpus_per_task=...`, etc.
- **Output layout**: single runs land under `hydra.run.dir`; sweeps (both local and submitit) land under `hydra.sweep.dir` with per-combo `subdir`s defined in each experiment's YAML. Submitit additionally writes per-job metadata under `${hydra.sweep.dir}/.submitit/%j`.
- **Environment variables** required at runtime (loaded via `.env`): `CHECKPOINT_BASE_PATH`, `HEAD_BASE_PATH`, `EVALUATION_BASE_PATH`, `TORCH_NUM_WORKERS`.

# REx baselines
 uv run --active python code/experiments/vision/rex/evaluate_rex.py \
   model_family=ilharco_hf_clip \
   model_name=openai/clip-vit-base-patch16 \
   dataset_name=CIFAR10 seed=2038 \
   'skip_modules=[classification_head]'

 uv run --active python code/experiments/vision/rex/evaluate_rex.py \
   model_family=ilharco_timm_supervised \
   model_name=vit_base_patch16_224.augreg2_in21k_ft_in1k \
   dataset_name=CIFAR10 seed=2038 \
   'skip_modules=[head]'

## Visualizations of REx
  uv run --active python code/visualizations/vision/rex/rex_heatmap.py --model-family ilharco_hf_clip --model-name openai/clip-vit-base-patch16 --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --order 2 --granularity channel --skip-modules classification_head --evaluation-root quantization/qat-transfer/evaluations/vision/rex

  uv run --active python code/visualizations/vision/rex/rex_heatmap.py --model-family ilharco_timm_supervised --model-name vit_base_patch16_224.augreg2_in21k_ft_in1k --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --order 2 --granularity channel --skip-modules head --evaluation-root quantization/qat-transfer/evaluations/vision/rex