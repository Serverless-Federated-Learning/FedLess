import io
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from pydantic import BaseModel, Field

from fedless.common.cache import cache
from fedless.datasets.dataset_loaders import DatasetLoader, DatasetNotLoadedError


class CharacterPredictionConfig(BaseModel):

    type: str = Field("char_prediction", const=True)
    dataset_name: str
    location: str
    http_params: Dict = None
    sequence_length: int = 80
    steps: int = 3
    split: str = "train"
    user_indices: Optional[List[int]] = None
    max_sequences: Optional[int] = 100000


class CharacterPrediction(DatasetLoader):
    def __init__(
        self,
        location: str,
        dataset_name: str,
        sequence_length: int = 80,
        steps: int = 3,
        split: str = "train",
        max_sequences: int = 100000,
        http_params: Dict = None,
        user_indices: Optional[List[int]] = None,
    ):

        self.location = location
        self.sequence_length = sequence_length
        self.http_params = http_params
        self.steps = steps
        self.dataset_name = dataset_name
        self.user_indices = user_indices
        self.max_sequences = max_sequences
        self.split = split

    @cache
    def load(self) -> tf.data.Dataset:
        """
        Load dataset
        :raise DatasetNotLoadedError when an error occurred in the process
        """
        text_embeddings, next_character_embedding = self._vectorize_samples()

        return tf.data.Dataset.from_tensor_slices((text_embeddings, next_character_embedding))

    def get_total_data_indices(self):
        sentences, _ = self._create_samples()
        return range(len(sentences))

    def _fetch_data(self):
        try:
            path = tf.keras.utils.get_file(self.dataset_name, origin=self.location)
            with io.open(path, encoding="utf-8") as f:
                text = f.read()
            chars_to_remove = ["=", "_", ">", "&", "}", "ä", "æ", "ë", "é", "Æ"]
            for char in chars_to_remove:
                text = text.replace(char, "")
            return text
        except:
            raise DatasetNotLoadedError

    def _create_samples(self) -> Tuple[list, list]:
        file_content = self._fetch_data()

        sentences = []
        next_chars = []

        for i in range(0, len(file_content) - self.sequence_length, self.steps):
            sentences.append(file_content[i : i + self.sequence_length])
            next_chars.append(file_content[i + self.sequence_length])

        num_test = max(int(len(next_chars) * 0.2), 1)

        if self.split == "test":
            sentences = sentences[-num_test:]
            next_chars = next_chars[-num_test:]
        else:
            sentences = sentences[:-num_test]
            next_chars = next_chars[:-num_test]

        if len(sentences) > self.max_sequences:
            sentences = sentences[: self.max_sequences]
            next_chars = next_chars[: self.max_sequences]

        if self.user_indices is not None:
            sentences = [sentences[i] for i in self.user_indices]
            next_chars = [next_chars[i] for i in self.user_indices]

        return sentences, next_chars

    def _vectorize_samples(self) -> Tuple[tf.Tensor, tf.Tensor]:

        sentences, next_chars = self._create_samples()

        vocabulary = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"

        vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            standardize=None,
            split=tf.strings.bytes_split,
            vocabulary=[c for c in vocabulary],
        )

        return (
            vectorizer(tf.convert_to_tensor(sentences)),
            vectorizer(tf.convert_to_tensor(next_chars)),
        )
