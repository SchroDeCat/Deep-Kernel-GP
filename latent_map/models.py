import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

assert tf.__version__.startswith("2")

# from tensorflow.keras.datasets import mnist
# print('TensorFlow version:', tf.__version__)
# print('Is Executing Eagerly?', tf.executing_eagerly())


class Encoder(tf.keras.layers.Layer):
    """First encoder to encode input"""

    def __init__(self, intermediate_dim):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim, activation=tf.nn.relu
        )
        self.output_layer = tf.keras.layers.Dense(
            units=intermediate_dim, activation=tf.nn.relu
        )

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)


class ContEncoder(tf.keras.Model):
    """Used for encoding the content"""

    def __init__(self, dest_dim, original_dim):
        super(ContEncoder, self).__init__()
        self.loss = []
        self.original_dimm = original_dim
        self.encoder = Encoder(intermediate_dim=dest_dim)

    def call(self, input_features):
        code = self.encoder(input_features)
        return code
