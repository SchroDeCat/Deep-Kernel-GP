import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import product
from data import Data_Factory_Base
from models import ContEncoder
from train import train_loop_2d
from loss import loss_2d
from utils import sample_data

assert tf.__version__.startswith("2")

# Data Factory
DF = Data_Factory_Base()
dim = 3
batch_size = 40
train_num = 10000
train_data = DF.convex_1(dim=dim, num= 60 * batch_size)

# dataset 2d
sampled_data = sample_data(train_data, sample_num = train_num)
feed_data = tf.data.Dataset.from_tensor_slices(sampled_data).batch(batch_size)

# define training process
model_2d = ContEncoder(dest_dim=dim - 1, original_dim=dim)
opt_2d = tf.keras.optimizers.Adam(learning_rate=2e-1)
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
