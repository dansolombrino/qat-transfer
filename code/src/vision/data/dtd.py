import torch
from datasets import load_dataset
from .common import HFVisionDataset, HF_TOKEN, HF_DATASETS_CACHE, make_val_split


class DTD:
    def __init__(
        self,
        preprocess_train,
        preprocess_inference,
        batch_size,
        num_workers
    ):

        hf_train = load_dataset("tanganke/dtd", split="train", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)

        self.train_dataset, self.val_dataset = make_val_split(
            train_dataset=HFVisionDataset(
                hf_dataset=hf_train,
                transform=None
            ),
            train_transform=preprocess_train,
            val_transform=preprocess_inference,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        hf_test = load_dataset("tanganke/dtd", split="test", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)
        self.test_dataset = HFVisionDataset(hf_test, transform=preprocess_inference)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        classnames = hf_train.features['label'].names
        self.class_names = [c.replace('_', ' ') for c in classnames]
