import numpy as np
from typing import List
import tensorflow as tf


def norm_square(t1: np.ndarray, t2: np.ndarray) -> float:
    """Calculate distance of two input vectors"""
    return tf.reduce_sum(tf.square(tf.subtract(t1, t2)))


def loss_2d(
    latent_space: np.ndarray, target_space: np.ndarray, Lambda: int = 1
) -> float:
    """Calculate loss of the 2D input"""
    tf.math.multiply(Lambda, target_space)
    return tf.reduce_mean(tf.math.maximum(tf.subtract(target_space, latent_space), 0))
