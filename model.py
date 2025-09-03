import tensorflow as tf
import numpy as np
from log_callback import LogCallback


class Model():
    def __init__(self, hidden_layer: int) -> tf.keras.Model:
        inputs = tf.keras.Input((1,))
        dense_1 = tf.keras.layers.Dense(hidden_layer, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(1)(dense_1)

        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=[tf.keras.metrics.MeanSquaredError()])

        self.model.summary()
            

    def train(self, dataset: tf.data.Dataset, epochs: int, output_filepath: str, dataset_size: int) -> None:
        log_callback = LogCallback(dataset, output_filepath, epochs, dataset_size)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
        
        self.model.fit(x=dataset["train"], validation_data=dataset["validation"], epochs=epochs, verbose=2, callbacks=[log_callback, reduce_lr])
