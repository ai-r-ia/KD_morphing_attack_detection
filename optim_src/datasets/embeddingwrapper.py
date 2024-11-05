import os
import glob
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import albumentations as A
from torch.utils.data import DataLoader

from datasets.embedding import EmbeddingDataItem, EmbeddingDatasetGenerator


class EmbeddingDatasetWrapper:
    CLASS_NAMES = ["bonafide", "morph"]

    def __init__(
        self,
        root_dir: str,
        morph_type: str,
        height: int = 224,
        width: int = 224,
        classes: int = 2,
    ) -> None:
        self.root_dir = root_dir
        self.morph_type = morph_type
        self.height = height
        self.width = width
        self.classes = classes

    def loop_through_dir(
        self,
        dir: str,
        label: int,
        augment_times: int = 0,
    ) -> List[EmbeddingDataItem]:
        allowed_extensions = {".jpg", ".png", ".jpeg"}
        items: List[EmbeddingDataItem] = []

        for image_path in os.listdir(dir):
            image_path = os.path.join(dir, image_path)
            embedding_path = os.path.splitext(image_path)[0] + ".npy"

            if os.path.exists(embedding_path):
                embedding = np.load(embedding_path)
                items.append(EmbeddingDataItem(image_path, embedding, label))
                items.extend(
                    EmbeddingDataItem(image_path, embedding, label)
                    for _ in range(augment_times)
                )

        return items

    def transform(self, data: EmbeddingDataItem) -> Any:
        embedding = data.embedding
        label = np.zeros((self.classes), dtype=np.float32)
        label[data.label] = 1  # One-hot encoding

        return embedding, label

    # def augment(self, image: np.ndarray, label: np.ndarray) -> Any:
    #     transform = A.Compose(
    #         [
    #             A.HorizontalFlip(p=0.25),
    #             A.VerticalFlip(p=0.25),
    #             A.RandomBrightnessContrast(p=0.2),
    #             A.InvertImg(p=0.05),
    #             A.PixelDropout(p=0.02),
    #         ]
    #     )

    #     transformed = transform(image=image)
    #     transformed_image = transformed["image"]

    #     return transformed_image, label

    def get_embeddings(
        self,
        split_type,
        augment_times: int,
        batch_size: int,
        morph_type: str,
        student_morph: str,
        shuffle: bool = True,
        num_workers: int = 8,
    ):
        data: List[EmbeddingDataItem] = []
        for label, cid in enumerate(self.CLASS_NAMES):
            augment_count = augment_times * 2 if cid == "bonafide" else augment_times

            if cid == "morph":
                cid = f"{morph_type}/morph/{student_morph}"
            data.extend(
                self.loop_through_dir(
                    os.path.join(self.root_dir, cid, split_type),
                    label,
                    augment_count,
                )
            )
        return DataLoader(
            EmbeddingDatasetGenerator(data, self.transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True
        )

    def get_train_embeddings(
        self,
        augment_times: int,
        batch_size: int,
        morph_type: str,
        student_morph: str,
        shuffle: bool,
        num_workers: int = 8,
    ):
        return self.get_embeddings(
            "train", augment_times, batch_size, morph_type,student_morph, shuffle, num_workers
        )

    def get_test_embeddings(
        self,
        augment_times: int,
        batch_size: int,
        morph_type: str,
        student_morph: str,
        shuffle: bool,
        num_workers: int = 8,
    ):
        return self.get_embeddings(
            "test", augment_times, batch_size, morph_type, student_morph,shuffle, num_workers
        )

    def get_multi_dataloaders(
        self,
        split_type: str,
        augment_times: int,
        batch_size: int,
        morph_types: List[str],
        student_morph: str,
        shuffle: bool = True,
        num_workers: int = 8,
    ) -> Dict[str, DataLoader]:
        dataloaders = {}
        for morph_type in morph_types:
            dataloaders[morph_type] = self.get_embeddings(
                split_type, augment_times, batch_size, morph_type,student_morph, shuffle, num_workers
            )
        return dataloaders
