import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

assert tf.__version__.startswith("2")


def norm_square(t1: np.ndarray, t2: np.ndarray) -> float:
    """Calculate distance of two input vectors"""
    return tf.reduce_sum(tf.square(tf.subtract(t1, t2)))


def loss_2d(
    latent_space: np.ndarray, target_space: np.ndarray, Lambda: int = 1
) -> float:
    """Calculate loss of the 2D input"""
    tf.math.multiply(Lambda, target_space)
    return tf.reduce_mean(tf.math.maximum(tf.subtract(target_space, latent_space), 0))


def train_2d(
    loss, model, opt, train_set: np.ndarray, original_dim: int, batch_size: int
):
    """
    Train set contains original vec & label
    """
    with tf.GradientTape() as tape:
        latent_space1 = model(
            tf.slice(train_set, [0, 0, 0], [batch_size, 1, original_dim])
        )
        latent_space2 = model(
            tf.slice(train_set, [0, 1, 0], [batch_size, 1, original_dim])
        )
        latent_space = norm_square(latent_space1, latent_space2)

        target_space1 = tf.slice(train_set, [0, 0, original_dim], [batch_size, 1, 1])
        target_space2 = tf.slice(train_set, [0, 1, original_dim], [batch_size, 1, 1])
        target_space = norm_square(target_space1, target_space2)

        latent_error = loss(latent_space, target_space)
    gradients = tape.gradient(latent_error, model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)

    return latent_error


def train_loop_2d(
    model,
    opt,
    loss,
    dataset: np.ndarray,
    epochs: int,
    original_dim: int,
    batch_size: int,
):
    """Training loop for 2d input"""
    for epoch in range(epochs):
        epoch_loss = 0
        for step, batch_features in enumerate(dataset):
            loss_values = train_2d(
                loss, model, opt, batch_features, original_dim, batch_size
            )
            epoch_loss += loss_values
        model.loss.append(epoch_loss)
        print("Epoch {}/{}. Loss: {}".format(epoch + 1, epochs, epoch_loss.numpy()))
