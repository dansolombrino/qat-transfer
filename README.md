# qat-transfer

All commands below assume the repo root as the working directory and the project `.venv` managed by `uv`. Every Hydra script resolves config search paths from `${oc.env:PWD}`, so **you must launch from the repo root** — not from inside `code/` or `experiments/`.

Scripts come in two families:
- **Hydra scripts** — everything under `code/src/vision/` and `code/experiments/`. They support three launch modes: single local run, local *sequential* sweep (Hydra's basic launcher), and Slurm *parallel* sweep (submitit launcher).
- **Argparse scripts** — the plotting utilities under `visualizations/`. Plain CLI, no Hydra.

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
The launcher config lives at [config/src/hydra/launcher/submitit_slurm.yaml](config/src/hydra/launcher/submitit_slurm.yaml) (partition, account, GPUs, CPUs, mem, walltime). Override any field on the CLI, e.g. `hydra.launcher.timeout_min=240 hydra.launcher.mem_gb=64`.

> Modes 2 and 3 both use `-m`. The **only** difference is whether you add `hydra/launcher=submitit_slurm`. Without it, Hydra's basic launcher runs the sweep sequentially in-process; with it, submitit fans the sweep out to Slurm.

---

## Finetuning

### FP finetuning — [code/src/vision/finetune_fp.py](code/src/vision/finetune_fp.py)
Config: [config/src/finetune_fp.yaml](config/src/finetune_fp.yaml)

Single local run:
```
uv run --active python code/src/vision/finetune_fp.py model_name=openai/clip-vit-base-patch16 dataset_name=CIFAR10 seed=2038
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/vision/finetune_fp.py -m model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/vision/finetune_fp.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3
```

### QAT finetuning — [code/src/vision/finetune_qat.py](code/src/vision/finetune_qat.py)
Config: [config/src/finetune_qat.yaml](config/src/finetune_qat.yaml). Requires `qat.bits`, `qat.granularity`, `qat.skip_modules`.

Single local run:
```
uv run --active python code/src/vision/finetune_qat.py model_name=openai/clip-vit-base-patch16 dataset_name=CIFAR10 seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/vision/finetune_qat.py -m model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/vision/finetune_qat.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

> The user's feedback memory notes: `finetune_*` scripts are GPU-intensive and long-running — do not launch them casually.

---

## 000_baselines — evaluation scripts

All five scripts live in [code/experiments/000_baselines/](code/experiments/000_baselines/) with matching configs in [experiments/000_baselines/](experiments/000_baselines/).

### evaluate_pretrained — [code/experiments/000_baselines/evaluate_pretrained.py](code/experiments/000_baselines/evaluate_pretrained.py)
Config: [experiments/000_baselines/evaluate_pretrained.yaml](experiments/000_baselines/evaluate_pretrained.yaml)

Single local run:
```
uv run --active python code/experiments/000_baselines/evaluate_pretrained.py model_name=openai/clip-vit-base-patch16 dataset_name=CIFAR10 batch_size=128 seed=2038
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/000_baselines/evaluate_pretrained.py -m model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 seed=1,2,3
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/000_baselines/evaluate_pretrained.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 seed=1,2,3
```

### evaluate_fp — [code/experiments/000_baselines/evaluate_fp.py](code/experiments/000_baselines/evaluate_fp.py)
Config: [experiments/000_baselines/evaluate_fp.yaml](experiments/000_baselines/evaluate_fp.yaml)

Single local run:
```
uv run --active python code/experiments/000_baselines/evaluate_fp.py model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/000_baselines/evaluate_fp.py -m model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/000_baselines/evaluate_fp.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3
```

### evaluate_fp_ptq — [code/experiments/000_baselines/evaluate_fp_ptq.py](code/experiments/000_baselines/evaluate_fp_ptq.py)
Config: [experiments/000_baselines/evaluate_fp_ptq.yaml](experiments/000_baselines/evaluate_fp_ptq.yaml). Adds a `ptq` group.

Single local run:
```
uv run --active python code/experiments/000_baselines/evaluate_fp_ptq.py model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/000_baselines/evaluate_fp_ptq.py -m model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/000_baselines/evaluate_fp_ptq.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

### evaluate_qat — [code/experiments/000_baselines/evaluate_qat.py](code/experiments/000_baselines/evaluate_qat.py)
Config: [experiments/000_baselines/evaluate_qat.yaml](experiments/000_baselines/evaluate_qat.yaml). The `qat.*` group must match the training config of the checkpoint being evaluated.

Single local run:
```
uv run --active python code/experiments/000_baselines/evaluate_qat.py model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/000_baselines/evaluate_qat.py -m model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/000_baselines/evaluate_qat.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

### evaluate_qat_ptq — [code/experiments/000_baselines/evaluate_qat_ptq.py](code/experiments/000_baselines/evaluate_qat_ptq.py)
Config: [experiments/000_baselines/evaluate_qat_ptq.yaml](experiments/000_baselines/evaluate_qat_ptq.yaml). Has both a `qat` group (to locate the checkpoint) and a `ptq` group (for the PTQ applied at eval time).

Single local run:
```
uv run --active python code/experiments/000_baselines/evaluate_qat_ptq.py model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/000_baselines/evaluate_qat_ptq.py -m model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/000_baselines/evaluate_qat_ptq.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

---

## 001_qat_transfer — QV transfer experiment

### qv_transfer — [code/experiments/001_qat_transfer/qv_transfer.py](code/experiments/001_qat_transfer/qv_transfer.py)
Config: [experiments/001_qat_transfer/qv_transfer.yaml](experiments/001_qat_transfer/qv_transfer.yaml)

Requires three checkpoints to already exist on disk: `FP_source`, `QAT_source`, `FP_target`. Defines a QV = `QAT_source - FP_source`, patches it into `FP_target` at scale `qv.alpha`, then applies PTQ and evaluates.

Single local run:
```
uv run --active python code/experiments/001_qat_transfer/qv_transfer.py model_name=openai/clip-vit-base-patch16 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 source.dataset_name=CIFAR10 source.seed=2038 target.dataset_name=CIFAR100 target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' qv.alpha=1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

Local sequential sweep (all source × all target datasets):
```
uv run --active python code/experiments/001_qat_transfer/qv_transfer.py -m model_name=openai/clip-vit-base-patch16 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 source.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet source.seed=2038 target.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' qv.alpha=0.25,0.5,0.75,1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

Submitit parallel sweep (all source × all target datasets):
```
uv run --active python code/experiments/001_qat_transfer/qv_transfer.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 source.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet source.seed=2038 target.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' qv.alpha=0.25,0.5,0.75,1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

---

## Visualizations (argparse, no Hydra, no Slurm)

These scripts read the JSON results produced by the experiments above and render plots. They use plain `argparse`, so Hydra sweeps and submitit do not apply — just call them one at a time (or wrap them in a shell `for` loop for a local sequential sweep).

### qv_transfer_heatmap — [visualizations/001_qat_transfer/qv_transfer_heatmap.py](visualizations/001_qat_transfer/qv_transfer_heatmap.py)
```
uv run --active python visualizations/001_qat_transfer/qv_transfer_heatmap.py --model-name openai/clip-vit-base-patch16 --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --qv-alpha 1.0
```

### qv_transfer_heatmap_best_sf — [visualizations/001_qat_transfer/qv_transfer_heatmap_best_sf.py](visualizations/001_qat_transfer/qv_transfer_heatmap_best_sf.py)
```
uv run --active python visualizations/001_qat_transfer/qv_transfer_heatmap_best_sf.py --model-name openai/clip-vit-base-patch16 --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --qv-alpha 1.0
```

Local sequential loop over `qv.alpha` values:
```
for alpha in 0.25 0.5 0.75 1.0; do uv run --active python visualizations/001_qat_transfer/qv_transfer_heatmap.py --model-name openai/clip-vit-base-patch16 --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --qv-alpha $alpha; done
```

---

## Notes

- **Always launch from the repo root.** Hydra config search paths use `${oc.env:PWD}` and will not resolve otherwise.
- **List-valued overrides** (e.g. `skip_modules`) must be quoted to protect brackets from the shell: `'qat.skip_modules=[classification_head]'`.
- **Tuning the Slurm job** without editing YAML: append `hydra.launcher.timeout_min=...`, `hydra.launcher.mem_gb=...`, `hydra.launcher.cpus_per_task=...`, etc.
- **Output layout**: single runs land under `hydra.run.dir`; sweeps (both local and submitit) land under `hydra.sweep.dir` with per-combo `subdir`s defined in each experiment's YAML. Submitit additionally writes per-job metadata under `${hydra.sweep.dir}/.submitit/%j`.
- **Environment variables** required at runtime (loaded via `.env`): `CHECKPOINT_BASE_PATH`, `HEAD_BASE_PATH`, `EVALUATION_BASE_PATH`, `TORCH_NUM_WORKERS`.
