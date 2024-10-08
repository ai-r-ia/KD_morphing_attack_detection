import os
import glob
import cv2
import numpy as np
from typing import List, Any
import albumentations as A
from torch.utils.data import DataLoader


from datasets.dataset import DataItem, DatasetGenerator


class DatasetWrapper:
    CLASS_NAMES = ["bonafide", "morph"]

    def __init__(
        self,
        root_dir: str,
        morph_type: str,
        morph_dir=None,
        height: int = 224,
        width: int = 224,
        classes: int = 2,
    ) -> None:
        self.root_dir = root_dir
        self.morph_type = morph_type
        self.morph_dir = morph_dir
        self.height = height
        self.width = width
        self.classes = classes

    def loop_through_dir(
        self,
        dir: str,
        label: int,
        augment_times: int = 0,
    ) -> List[DataItem]:
        allowed_extensions = {".jpg", ".png", ".jpeg"}
        items: List[DataItem] = []

        for image_path in os.listdir(dir):
            if os.path.splitext(image_path)[1].lower() in allowed_extensions:
                image_path = os.path.join(dir, image_path)
                items.append(DataItem(image_path, False, label))
                items.extend(
                    DataItem(image_path, True, label) for _ in range(augment_times)
                )

        return items

    def transform(self, data: DataItem) -> Any:
        image = cv2.imread(data.path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load image: {data.path}")
            return None, None  # Or handle this case as needed

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height))
        image = (image - image.min()) / ((image.max() - image.min()) or 1.0)
        image = np.transpose(image.astype("float32"), axes=(2, 0, 1))

        label = np.zeros((self.classes), dtype=np.float32)
        label[data.label] = 1  # One-hot encoding

        return image, label

    def augment(self, image: np.ndarray, label: np.ndarray) -> Any:
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.25),
                A.VerticalFlip(p=0.25),
                A.RandomBrightnessContrast(p=0.2),
                A.InvertImg(p=0.05),
                A.PixelDropout(p=0.02),
            ]
        )

        transformed = transform(image=image)
        transformed_image = transformed["image"]

        return transformed_image, label

    def get_dataset(
        self,
        split_type,
        augment_times: int,
        batch_size: int,
        morph_type: str,
        shuffle: bool = True,
        num_workers: int = 8,
    ):
        data: List[DataItem] = []
        for label, cid in enumerate(self.CLASS_NAMES):
            augment_count = augment_times * 2 if cid == "bonafide" else augment_times
            root_dir = self.root_dir
            if cid == "morph":
                cid = f"morph/{morph_type}"
                if self.morph_dir != self.root_dir:
                    root_dir = self.morph_dir
                    cid = ""
                    if split_type == "train":
                        split_type = "Train"
                    if split_type == "test":
                        split_type = "Test"

            data.extend(
                self.loop_through_dir(
                    os.path.join(root_dir, cid, split_type),
                    label,
                    augment_count,
                )
            )
        
        return DataLoader(
            DatasetGenerator(data, self.transform, self.augment),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def get_train_dataset(
        self,
        augment_times: int,
        batch_size: int,
        morph_type: str,
        shuffle: bool = True,
        num_workers: int = 8,
    ):
        return self.get_dataset(
            "train", augment_times, batch_size, morph_type, shuffle, num_workers
        )

    def get_test_dataset(
        self,
        augment_times: int,
        batch_size: int,
        morph_type: str,
        shuffle: bool = True,
        num_workers: int = 8,
    ):
        return self.get_dataset(
            "test", augment_times, batch_size, morph_type, shuffle, num_workers
        )
