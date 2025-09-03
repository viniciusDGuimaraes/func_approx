import tensorflow as tf
import random
from typing import Tuple, Dict, List


class DatasetGenerator:
    def __new__(cls):
        raise TypeError("Cannot instantiate DatasetGenerator")


    @staticmethod
    def generate_dataset(inputs: List[Tuple[float, float]], train_split_pct: float, val_split_pct: float, seed: int=42) -> Dict[str, tf.data.Dataset]:
        def _sanitize_parameters(inputs: Tuple[float, float], train_split_pct: float, val_split_pct: float):
            if type(inputs) != list:
                raise TypeError(f"inputs argument is of type {type(inputs)}. Expected tuple")

            if type(train_split_pct) != float:
                raise TypeError(f"train_split_pct argument is of type {type(train_split_pct)}. Expected float")

            if train_split_pct <= 0:
                raise ValueError("train_split_pct argument should have value greater than 0.")

            if type(val_split_pct) != float:
                raise TypeError(f"val_split_pct argument is of type {type(val_split_pct)}. Expected float")

            if val_split_pct <= 0:
                raise ValueError("val_split_pct argument should have value greater than 0.")

            if train_split_pct + val_split_pct > 1:
                raise ValueError("Arguments train_split_size and val_split_size should sum up to 1.")

        
        def reshape_features(x, y):
            x = tf.expand_dims(x, axis=-1)
            y = tf.expand_dims(y, axis=-1)
            return x, y


        _sanitize_parameters(inputs, train_split_pct, val_split_pct)

        dataset_size = len(inputs)
        train_size = int(dataset_size * train_split_pct)
        val_size = int(dataset_size * val_split_pct)
        test_size = dataset_size - train_size - val_size

        random.seed(42)
        random.shuffle(inputs)

        xs, ys = zip(*inputs)

        dataset = tf.data.Dataset.from_tensor_slices((list(xs), list(ys)))
        dataset = dataset.map(reshape_features)

        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        
        test_dataset = None
        if test_size > 0:
            test_dataset = dataset.skip(train_size + val_size).take(test_size)

        dataset_dict = {
            "train": train_dataset,
            "validation": val_dataset
        }

        if test_dataset is not None:
            dataset_dict["test"] = test_dataset

        for split in dataset_dict:
            dataset_dict[split].shuffle(buffer_size=len(dataset_dict[split]))

        return dataset_dict
