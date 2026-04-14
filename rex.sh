#!/bin/bash

# Define the array of datasets
DATASETS=(
    "Cars"
    "DTD"
    "EuroSAT"
    "GTSRB"
    "MNIST"
    "RESISC45"
    "SUN397"
    "SVHN"
    "CIFAR100"
    "STL10"
    "Food101"
    "Flowers102"
    "FER2013"
    "PCAM"
    "OxfordIIITPet"
    "RenderedSST2"
    "EMNIST"
    "FashionMNIST"
    "KMNIST"
    "TinyImageNet"
    "ImageNet"
)

# Loop over each dataset in the array
for DATASET in "${DATASETS[@]}"; do
    echo "============================================================"
    echo "Starting evaluation for dataset: $DATASET"
    echo "============================================================"

    # Run the CLIP model evaluation
    echo "-> Running ilharco_hf_clip for $DATASET..."
    uv run --active python code/experiments/vision/rex/evaluate_rex.py \
       model_family=ilharco_hf_clip \
       model_name=openai/clip-vit-base-patch16 \
       dataset_name="$DATASET" seed=2038 \
       "skip_modules=['classification_head']"

    # Run the TIMM supervised model evaluation
    echo "-> Running ilharco_timm_supervised for $DATASET..."
    uv run --active python code/experiments/vision/rex/evaluate_rex.py \
       model_family=ilharco_timm_supervised \
       model_name=vit_base_patch16_224.augreg2_in21k_ft_in1k \
       dataset_name="$DATASET" seed=2038 \
       "skip_modules=['head']"
       
    echo "Finished evaluating $DATASET."
    echo ""
done

echo "All dataset evaluations are complete!"