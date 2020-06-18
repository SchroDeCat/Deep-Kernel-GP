import numpy as np
import matplotlib.pyplot as plt
from latent_map.data import Data_Factory, Data_Factory_Base

from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,ConstantKernel

# get data
DFB = Data_Factory_Base()
dim = 2
batch_size = 40
obj_func = DFB.convex_1
DF = Data_Factory(obj_func=obj_func, dim=dim, batch_size=batch_size)
feed_data, x_train, y_train = DF.get_data()

# train a raw GPR

# train a GPR with latent map

# train a DKl

# comparison


