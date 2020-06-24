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


def train_gp(gp, optimizer, loss, model, train_set, original_dim, batch_size):
    """Input data points and GP(tfp)"""
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
        gp_loss1 = -gp.get_marginal_distribution(latent_space1).log_prob(target_space1)
        gp_loss2 = -gp.get_marginal_distribution(latent_space2).log_prob(target_space2)
        err = tf.math.add(
            tf.math.add(latent_error, tf.reduce_mean(gp_loss1)),
            tf.reduce_mean(gp_loss2),
        )
    vars = tuple([*gp.trainable_variables, *model.trainable_variables])
    grads = tape.gradient(err, vars)
    optimizer.apply_gradients(zip(grads, vars))
    return err


def train_gp_loop_2d(gp, model, opt, loss, dataset, epochs, original_dim, batch_size):
    """Train DKL with 2d regularizer"""
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_features in dataset:
            err = train_gp(
                gp, opt, loss, model, batch_features, original_dim, batch_size
            )
            epoch_loss += err
        model.loss.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}. Loss: {epoch_loss.numpy()}")
