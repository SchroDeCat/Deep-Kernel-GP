import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

assert tf.__version__.startswith("2")

from itertools import product
from data import Data_Factory_Base
from models import ContEncoder
from train import train_loop_2d, train_gp_loop_2d
from loss import loss_2d
from utils import sample_data
from sklearn.metrics import r2_score

# Data Factory
DF = Data_Factory_Base()
dim = 3
batch_size = 40
train_num = 10000
train_data = DF.convex_1(dim=dim, num=3 * batch_size)
x_train = train_data[:,:-1].astype('float32') 
y_train = train_data[:,-1].astype('float32')
test_data = DF.convex_1(dim=dim, num=1 * batch_size)
x_test = test_data[:,:-1].astype('float32') 
y_test= test_data[:,-1].astype('float32') 

# dataset 2d
sampled_data = sample_data(train_data, sample_num=train_num)
feed_data = tf.data.Dataset.from_tensor_slices(sampled_data).batch(batch_size)

# GP
kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
    amplitude=tf.Variable(1.0, dtype=np.float32, name="amplitude"),
    length_scale=tf.Variable(1.0, dtype=np.float32, name="length_scale"),
)   # k(x, y) = amplitude**2 * exp(-||x - y||**2 / (2 * length_scale**2)) # equals to RBF

gp = tfp.distributions.GaussianProcess(kernel)

# define training process
model_2d = ContEncoder(dest_dim=dim - 1, original_dim=dim)
opt_2d = tf.keras.optimizers.Adam(learning_rate=2e-1)

train_gp_loop_2d(
    gp,
    model_2d,
    opt_2d,
    loss=loss_2d,
    dataset=feed_data,
    epochs=4,
    original_dim=train_data.shape[1] - 1,
    batch_size=batch_size,
)

# plot figure
plt.plot(range(10), model_2d.loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# check regression
gprm = tfp.distributions.GaussianProcessRegressionModel(
    kernel=kernel,
    index_points=x_test,
    observation_index_points=x_train,
    observations=y_train.astype('float32'),
    jitter=1e-3)

samples = gprm.sample(100).numpy()
y_pred = np.mean(samples, axis=0)
print(f"R score: {r2_score(y_pred, y_test)}")
