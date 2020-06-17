import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import product
from data import Data_Factory
from models import ContEncoder
from train import train_loop_2d
from loss import loss_2d

assert tf.__version__.startswith("2")

# Data Factory
DF = Data_Factory()
dim = 3
batch_size = 25
train_data = DF.convex_1(dim=dim, num=3 * batch_size)
x_train = train_data[:, :-1].astype("float32")
y_train = train_data[:, -1]
test_data = DF.convex_1(dim=dim, num=1 * batch_size)
x_test = test_data[:, :-1].astype("float32")
y_test = test_data[:, -1]

training_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
training_labels = tf.data.Dataset.from_tensor_slices(y_train).batch(batch_size)

# dataset 2d
compond = list(product(train_data.astype("float32"), train_data.astype("float32")))
feed_data = tf.data.Dataset.from_tensor_slices(compond).batch(batch_size)

# define training process
model_2d = ContEncoder(dest_dim=dim - 1, original_dim=dim)
opt_2d = tf.keras.optimizers.Adam(learning_rate=1e-2)
train_loop_2d(
    model_2d,
    opt_2d,
    loss=loss_2d,
    dataset=feed_data,
    epochs=10,
    original_dim=train_data.shape[1] - 1,
    batch_size=batch_size,
)

# plot figure
plt.plot(range(10), model_2d.loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
