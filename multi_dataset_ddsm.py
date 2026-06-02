"""DDSM double-view dataset utilities.

The model expects each sample to contain two mammography views with shape
``[num_views, channels, height, width]``. Images are loaded as grayscale so the
resulting tensor has one channel per view.
"""

from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from constants import VIEWS


REQUIRED_COLUMNS = ("patient_id", "image_id", "laterality", "view", "cancer")
IMAGE_EXTENSIONS = ("", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
DEFAULT_LABEL_MAP: Dict[str, int] = {
    "benign_without_callbacks": 0,
    "benigns": 1,
    "cancers": 2,
}


def get_transforms(
    aug: bool = False,
    image_size: Tuple[int, int] = (1024, 512),
) -> Callable[[Image.Image], torch.Tensor]:
    """Build image transforms for train or evaluation."""

    if aug:
        transforms = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=(-5, 5)),
            T.RandomResizedCrop(size=image_size, scale=(0.8, 1.0), ratio=(0.45, 0.55)),
        ]
    else:
        transforms = [T.Resize(image_size)]

    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def _as_path_token(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _normalize_view(laterality: object, view: object) -> str:
    view_name = str(view)
    if "-" in view_name:
        return view_name
    return f"{laterality}-{view_name}"


def _view_sort_key(row: pd.Series) -> Tuple[int, str]:
    view = str(row["view"]).upper()
    if "CC" in view:
        order = 0
    elif "MLO" in view:
        order = 1
    else:
        order = 2
    return order, _as_path_token(row["image_id"])


class DDSMDoubleViewDataset(Dataset):
    """Pairs CC/MLO views from the same patient and laterality."""

    def __init__(
        self,
        csv_dir: str,
        images_dir: str,
        _is_train: bool = True,
        transforms: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        horizontal_flip: str = "NO",
        label_map: Optional[Mapping[str, int]] = None,
        image_size: Tuple[int, int] = (1024, 512),
    ) -> None:
        self.csv_dir = Path(csv_dir)
        self.images_dir = Path(images_dir)
        self.is_train = bool(_is_train)
        self.horizontal_flip = horizontal_flip
        self.label_map = dict(label_map or DEFAULT_LABEL_MAP)
        self.transforms = transforms or get_transforms(self.is_train, image_size)

        if not self.csv_dir.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_dir}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")

        self.df = pd.read_csv(self.csv_dir)
        missing = set(REQUIRED_COLUMNS) - set(self.df.columns)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"{self.csv_dir} is missing columns: {missing_columns}")

        self.pairs = self._build_pairs()
        if not self.pairs:
            raise ValueError(f"No double-view pairs found in {self.csv_dir}")

    def _build_pairs(self) -> Sequence[Tuple[pd.Series, pd.Series]]:
        pairs = []
        grouped = self.df.groupby(["patient_id", "laterality"], sort=False)

        for (patient_id, laterality), group in grouped:
            if len(group) < 2:
                continue

            rows = sorted(
                (row for _, row in group.iterrows()),
                key=_view_sort_key,
            )
            first, second = rows[0], rows[1]

            if first["cancer"] != second["cancer"]:
                raise ValueError(
                    "Inconsistent labels for "
                    f"patient={patient_id}, laterality={laterality}"
                )

            pairs.append((first, second))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        first, second = self.pairs[index]
        image = torch.stack(
            [self._load_view(first), self._load_view(second)],
            dim=0,
        )
        label = torch.tensor(self._encode_label(first["cancer"]), dtype=torch.long)

        return {
            "image": image,
            "cancer": label,
        }

    def _encode_label(self, label: object) -> int:
        if isinstance(label, str):
            if label not in self.label_map:
                raise KeyError(f"Unknown label {label!r}; update label_map.")
            return self.label_map[label]
        return int(label)

    def _load_view(self, row: pd.Series) -> torch.Tensor:
        path = self._resolve_image_path(row)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")

        if self._should_flip(row):
            image = np.fliplr(image).copy()

        pil_image = Image.fromarray(image.astype(np.uint8), mode="L")
        return self.transforms(pil_image)

    def _resolve_image_path(self, row: pd.Series) -> Path:
        stem = "_".join(
            [
                _as_path_token(row["patient_id"]),
                _as_path_token(row["laterality"]),
                _as_path_token(row["view"]),
                _as_path_token(row["image_id"]),
            ]
        )

        for extension in IMAGE_EXTENSIONS:
            candidate = self.images_dir / f"{stem}{extension}"
            if candidate.exists():
                return candidate

        return self.images_dir / stem

    def _should_flip(self, row: pd.Series) -> bool:
        policy = str(self.horizontal_flip).lower()
        view = _normalize_view(row["laterality"], row["view"])

        if policy in {"right", "standardize_right"}:
            return VIEWS.is_right(view)
        if policy in {"left", "standardize_left"}:
            return VIEWS.is_left(view)
        return False


RsnaDataset = DDSMDoubleViewDataset


def create_dataloaders(
    train_csv: Optional[str],
    valid_csv: Optional[str],
    test_csv: Optional[str],
    images_dir: str,
    batch_size: int,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (1024, 512),
    pin_memory: Optional[bool] = None,
    label_map: Optional[Mapping[str, int]] = None,
) -> Dict[str, DataLoader]:
    """Create train, validation, and optional test dataloaders."""

    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

    loaders: Dict[str, DataLoader] = {}

    if train_csv:
        loaders["train"] = DataLoader(
            DDSMDoubleViewDataset(
                train_csv,
                images_dir,
                _is_train=True,
                transforms=get_transforms(aug=True, image_size=image_size),
                label_map=label_map,
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if valid_csv:
        loaders["valid"] = DataLoader(
            DDSMDoubleViewDataset(
                valid_csv,
                images_dir,
                _is_train=False,
                transforms=get_transforms(aug=False, image_size=image_size),
                label_map=label_map,
            ),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if test_csv:
        loaders["test"] = DataLoader(
            DDSMDoubleViewDataset(
                test_csv,
                images_dir,
                _is_train=False,
                transforms=get_transforms(aug=False, image_size=image_size),
                label_map=label_map,
            ),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loaders
