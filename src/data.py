"""CIFAR-100 loaders via timm, using the transforms the pretrained model expects."""
import torch
from timm.data import create_dataset, create_transform, resolve_data_config
from torch.utils.data import DataLoader, Subset


def make_loaders(model, root="data", batch_size=128, num_workers=4, subset=0, seed=0):
    # resolve_data_config reads input size (224) + mean/std from the pretrained model.
    cfg = resolve_data_config({}, model=model)
    loaders = {}
    for split, training in (("train", True), ("validation", False)):
        ds = create_dataset("torch/cifar100", root=root, split=split, download=True)
        ds.transform = create_transform(**cfg, is_training=training)
        if subset:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(ds), generator=g)[:subset].tolist()
            ds = Subset(ds, idx)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=training, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), drop_last=training,
            persistent_workers=num_workers > 0,
        )
    return loaders["train"], loaders["validation"]
