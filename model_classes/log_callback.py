import tensorflow as tf
import numpy as np
import json
from typing import Dict


class LogCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset: Dict[str, tf.data.Dataset], output_filepath: str, epochs: int, dataset_size: int):
        super().__init__()
        self.dataset = dataset
        self.output_filepath = output_filepath
        self.preds = np.empty((epochs, dataset_size, 4), dtype=np.float32)

    
    def on_epoch_end(self, epoch, logs):
        count = 0
        for current_split in self.dataset:
            for x, y in self.dataset[current_split]:
                y_pred_raw = self.model(x)

                x = x.numpy()[0]
                y_true = y.numpy()[0]
                y_pred = y_pred_raw.numpy()[0][0]

                self.preds[epoch, count, 0] = epoch
                self.preds[epoch, count, 1] = x
                self.preds[epoch, count, 2] = y_true
                self.preds[epoch, count, 3] = y_pred

                count += 1


    def on_train_end(self, logs):
        np.save(self.output_filepath, self.preds)
