"""T5 image groups: one dataset per class plus a mixed-class control, for CIFAR-100 or ImageNet.

ImageNet is read in place from the lab copy (train split only; there is no val split on the server). Files are listed
per class folder instead of scanning all 1.28M images, and labels come from timm's ImageNetInfo, so they match the
pretrained model's 1000-class head.
"""
from pathlib import Path

import torch
from PIL import Image
from timm.data import ImageNetInfo
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR100

IMAGENET_ROOT = "/storage/INSLAB/Data/ILSVRC2012_img_train"
CONTROL = "control (all classes)"
DEFAULT_CIFAR_CLASSES = ("apple", "bicycle", "fox", "cloud")  # one each from 4 different CIFAR-100 superclasses
# 4 matched to the CIFAR set (no "cloud" in ImageNet, so a landscape class), then two look-alike pairs
IMAGENET_MATCHED = ("n07742313", "n03792782", "n02119022", "n09193705")  # Granny Smith, mountain bike, red fox, alp
IMAGENET_PAIRS = (("n02110063", "n02110185"),  # malamute, Siberian husky
                  ("n02123045", "n02123159"))  # tabby, tiger cat
DEFAULT_IMAGENET_CLASSES = IMAGENET_MATCHED + tuple(wnid for pair in IMAGENET_PAIRS for wnid in pair)


class FileListDataset(Dataset):
    """(image, label) pairs from an explicit list of image files."""

    def __init__(self, files: list[Path], labels: list[int], transform) -> None:
        if len(files) != len(labels):
            raise ValueError(f"{len(files)} files but {len(labels)} labels")
        self.files, self.labels, self.transform = files, labels, transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB")), self.labels[index]


def resolve_class_indices(all_classes: list[str], names: list[str]) -> list[int]:
    """Map CIFAR-100 class names to label indices. Raises ValueError on an unknown name."""
    unknown = [name for name in names if name not in all_classes]
    if unknown:
        raise ValueError(f"unknown CIFAR-100 class(es) {unknown}; valid names: {sorted(all_classes)}")
    return [all_classes.index(name) for name in names]


def resolve_imagenet_classes(info: ImageNetInfo, classes: list[str]) -> list[int]:
    """Map ImageNet wnids ("n02119022") or class indices ("277") to indices. Raises ValueError on unknown ones."""
    wnids = info.label_names()
    indices, unknown = [], []
    for item in classes:
        if item.isdigit() and int(item) < len(wnids):
            indices.append(int(item))
        elif item in wnids:
            indices.append(wnids.index(item))
        else:
            unknown.append(item)
    if unknown:
        raise ValueError(f"unknown ImageNet class(es) {unknown}; use a wnid like n02119022 or an index 0-999")
    return indices


def imagenet_short_name(info: ImageNetInfo, index: int) -> str:
    """First synonym of the class description, e.g. 'red fox' for 'red fox, Vulpes vulpes'."""
    return info.index_to_description(index).split(",")[0]


def class_files(root: Path, wnid: str) -> list[Path]:
    """Sorted image files of one ImageNet class folder. Raises FileNotFoundError if there are none."""
    files = sorted(path for path in (root / wnid).glob("*") if path.suffix.lower() in {".jpeg", ".jpg", ".png"})
    if not files:
        raise FileNotFoundError(f"no images for {wnid} under {root}")
    return files


def cifar_subsets(
    transform, root: str, class_names: list[str], per_class: int, control_size: int, seed: int, split: str
) -> dict[str, Dataset]:
    """One Subset of a CIFAR-100 split per class (first `per_class` images), plus a seeded mixed-class control.

    torchvision's CIFAR100 is the same data timm's "torch/cifar100" wraps, and it exposes `.classes`/`.targets`.
    """
    dataset = CIFAR100(root, train=split == "train", download=True, transform=transform)
    targets = torch.tensor(dataset.targets)
    groups: dict[str, Dataset] = {}
    for name, label in zip(class_names, resolve_class_indices(dataset.classes, class_names)):
        groups[name] = Subset(dataset, (targets == label).nonzero().flatten()[:per_class].tolist())
    if control_size:
        generator = torch.Generator().manual_seed(seed)
        groups[CONTROL] = Subset(dataset, torch.randperm(len(dataset), generator=generator)[:control_size].tolist())
    return groups


def imagenet_subsets(
    transform, root: str, classes: list[str], per_class: int, control_size: int, seed: int
) -> dict[str, Dataset]:
    """One FileListDataset per ImageNet class (first `per_class` files), plus a control of `control_size` images,
    each from a different randomly chosen class (seeded) among the class folders present under `root`."""
    info, root_path = ImageNetInfo(), Path(root)
    groups: dict[str, Dataset] = {}
    for index in resolve_imagenet_classes(info, classes):
        files = class_files(root_path, info.index_to_label_name(index))[:per_class]
        groups[imagenet_short_name(info, index)] = FileListDataset(files, [index] * len(files), transform)
    if control_size:
        present = [i for i, wnid in enumerate(info.label_names()) if (root_path / wnid).is_dir()]
        if control_size > len(present):
            raise ValueError(f"control of {control_size} needs as many classes; only {len(present)} found")
        generator = torch.Generator().manual_seed(seed)
        chosen = [present[i] for i in torch.randperm(len(present), generator=generator)[:control_size].tolist()]
        files, labels = [], []
        for index in chosen:
            candidates = class_files(root_path, info.index_to_label_name(index))
            files.append(candidates[torch.randint(len(candidates), (1,), generator=generator).item()])
            labels.append(index)
        groups[CONTROL] = FileListDataset(files, labels, transform)
    return groups


def imagenet_separation_sets(classes: list[str]) -> dict[str, list[str]]:
    """Which class groups to measure separation over: all chosen classes, the 4 CIFAR-matched ones, and each
    look-alike pair, when all their members were chosen. Values are group names as `imagenet_subsets` makes them."""
    info = ImageNetInfo()
    indices = resolve_imagenet_classes(info, classes)
    name = {info.index_to_label_name(i): imagenet_short_name(info, i) for i in indices}
    sets = {"all classes": list(name.values())}
    if set(IMAGENET_MATCHED) <= name.keys():
        sets["matched 4"] = [name[wnid] for wnid in IMAGENET_MATCHED]
    for first, second in IMAGENET_PAIRS:
        if {first, second} <= name.keys():
            sets[f"{name[first]} vs {name[second]}"] = [name[first], name[second]]
    return sets
