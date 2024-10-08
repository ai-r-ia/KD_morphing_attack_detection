from numpy import ndarray
from torch.utils.data import Dataset
from typing import List, Callable, Tuple


class EmbeddingDataItem:
    def __init__(self, image_path: str, embedding: ndarray, label: int):
        self.image_path = image_path
        self.embedding = embedding
        self.label = label


class EmbeddingDatasetGenerator(Dataset):
    def __init__(
        self,
        data: List[EmbeddingDataItem],
        transform: Callable[[EmbeddingDataItem], Tuple[ndarray, ndarray]],
    ) -> None:
        super(EmbeddingDatasetGenerator, self).__init__()

        self.data: List[EmbeddingDataItem] = data
        self.transform: Callable[[EmbeddingDataItem], Tuple[ndarray, ndarray]] = (
            transform
        )
        # self.augment: Callable[
        #     [ndarray, ndarray], Tuple[ndarray, ndarray]
        # ] = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[ndarray, ndarray]:
        embedding, label = self.transform(self.data[idx])
        return embedding, label
