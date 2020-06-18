import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import sleep
from tqdm import tqdm

import sys
sys.path.append('../')

from dknet import NNRegressor
from dknet.layers import Dense,CovMat,Dropout,Parametrize,Scale
from dknet.optimizers import Adam,SciPyMin,SDProp
from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,ConstantKernel
from target_funcs import *
from utils import time_process

class Collision_Problem():
	def __init__(self, f, train_num=100, noise=0):
		'''Init (1)training and testing data (2) form the network'''
		np.random.seed(0)
		self.target_func = f
		self.noise = noise
		self.x_train=np.random.random(size=(train_num,1)) - .5
		self.y_train=f(self.x_train)+np.random.normal(0.0,noise,size=self.x_train.shape)
		self.x_test=np.linspace(-.7, .7, 10000).reshape(-1,1)
		self.x_plot=np.linspace(-.7, .7, 10000).reshape(-1,1)
		self.model()

	def model(self, latent_dim:int=1):
		'''Keras style layers + optimizers'''
		self.layers=[]
		# self.layers.append(Dense(32, activation='tanh'))
		# self.layers.append(Dropout(keep_prob=0.9))
		# self.layers.append(Dense(16, activation='tanh'))
		# self.layers.append(Dropout(keep_prob=0.9))
		# self.layers.append(Dense(8, activation='tanh'))
		# self.layers.append(Dropout(keep_prob=0.9))
		self.layers.append(Dense(16, activation='tanh'))
		self.layers.append(Dropout(keep_prob=0.9))
		self.layers.append(Dense(latent_dim))
		self.layers.append(Scale(fixed=True, init_vals=1.1))	# why need scaling
		# self.layers.append(CovMat(kernel='rbf', alpha_fixed=False))
		self.layers.append(CovMat(kernel='rbf', alpha_fixed=True, alpha=self.noise))	# noise free
		# optimizer
		self.opt=Adam(1e-3)
		#self.opt=SciPyMin('l-bfgs-b')

	@time_process
	def find_collision(self, iter_num:int=1000, tol:float=1e-3):
		''' 
		Test problem in Latent space
		Return: if find the problem
		'''
		self.gp = NNRegressor(self.layers,opt=self.opt,batch_size=self.x_train.shape[0],maxiter=iter_num, gp=True,verbose=False)
		self.gp.fit(self.x_train, self.y_train)
		self.y_pred, self.std = self.gp.predict(self.x_test)
		
		# find same latent
		xt = 0.2
		zt = self.gp.fast_forward(xt)	# latent space
		z_test = self.gp.fast_forward(self.x_test)

		nearest = np.argmin(np.abs(z_test - zt))
		collision =  np.abs(self.gp.fast_forward(self.x_test[nearest]) - zt) < tol
		if collision:
			print(f"Iter Num: {iter_num} \n{'*'*50}\
			\nCloset Latent: \t{self.gp.fast_forward(self.x_test[nearest])}\t/ {zt},\
			\nInput:\t\t{self.x_test[nearest]})\t/ {xt},\
			\nOutput:\t\t{self.target_func(self.x_test[nearest])}\t/ {self.target_func(np.ones(1) * xt)}")
			self.plot_collision()
		return collision
	
	def plot_collision(self):
		''' Plot the collision on latent space and target space '''
		fig = plt.figure(figsize=(5,3.2))
		ax1 = fig.add_subplot(121)
		ax1.plot(self.x_plot, self.gp.fast_forward(self.x_plot))
		ax1.set_xlabel('X')
		ax1.set_ylabel('Z')
		ax1.set_title("Latent Map")

		ax2 = fig.add_subplot(122)
		ax2.plot(self.x_train, self.y_train,'.')
		ax2.plot(self.x_plot, self.target_func(self.x_plot)[:,0])
		ax2.plot(self.x_plot, self.y_pred)
		ax2.set_xlabel('X')
		ax2.set_ylabel('Y')
		ax2.fill_between(self.x_test[:,0], self.y_pred[:,0] - self.std, self.y_pred[:,0] + self.std,alpha=0.5)
		ax2.set_title("Prediction")
		
		plt.legend(['Training samples', 'True function', 'Predicted function','Prediction stddev'])
		plt.show()

class Collision_Problem_2d(Collision_Problem):
	'''Noise free 2-dimensional version'''
	def __init__(self, f, train_num:int=100, dim:int=2, noise:float=0):
		np.random.seed(0)
		self.resolution = 100j
		self.resolution_num = 100
		self.noise = noise
		self.target_func = f
		self.x_train = np.random.uniform(low=np.ones(dim)*-1.0, high=np.ones(dim)*1.0, size=[train_num, dim])
		self.y_train = np.array([self.target_func(vec) for vec in self.x_train])
		self.y_train = self.y_train.reshape(-1,1)
		self.base_test = np.linspace(-1.0, 1.0, 10000)	# latent space is still one dimension
		self.x_test = np.mgrid[-1.0:1:(self.resolution),-1.0:1:(self.resolution)].reshape(dim, -1).T
		self.x_plot = np.mgrid[-1.0:1:(self.resolution),-1.0:1:(self.resolution)]
		self.y_plot = np.array([self.target_func(vec) for vec in self.x_test]).T.reshape([self.resolution_num, self.resolution_num])
		self.model()

	def plot_collision(self):
		''' Plot the collision on latent space '''
		# fig = plt.figure(figsize=(10,6.5))
		fig = plt.figure()
		ax1 = fig.add_subplot(221, projection='3d')
		self.z_plot = self.gp.fast_forward(self.x_test).reshape([self.resolution_num, self.resolution_num])
		# print(f"x_plot {self.x_plot[0].shape} {self.x_plot[1].shape}; z_plot {self.z_plot.shape}")
		ax1.plot_surface(self.x_plot[0], self.x_plot[1], self.z_plot)
		ax1.set_xlabel('X1')
		ax1.set_ylabel('X2')
		ax1.set_zlabel('Z')
		ax1.set_title("Latent Map")

		ax2 = fig.add_subplot(222, projection='3d')
		ax2.plot_surface(self.x_plot[0], self.x_plot[1], self.y_plot)
		ax2.set_title("Obj Func")

		ax3 = fig.add_subplot(223, projection='3d')
		ax3.plot_surface(self.x_plot[0], self.x_plot[1], self.y_pred.reshape([self.resolution_num, self.resolution_num]))
		ax3.set_title("Predicted Func")
		
		ax4 = fig.add_subplot(224, projection='3d')
		ax4.scatter(self.x_train.T[0], self.x_train.T[1], self.y_train)
		ax4.set_title("Training Data")
		
		plt.show()

	def model(self, latent_dim:int=1):
		'''Keras style layers + optimizers'''
		self.layers=[]
		self.layers.append(Dense(32, activation='tanh'))
		self.layers.append(Dropout(keep_prob=0.9))
		self.layers.append(Dense(16, activation='tanh'))
		self.layers.append(Dropout(keep_prob=0.9))
		self.layers.append(Dense(8, activation='tanh'))
		self.layers.append(Dropout(keep_prob=0.9))
		self.layers.append(Dense(latent_dim))
		self.layers.append(Scale(fixed=True, init_vals=10))	# why need scaling
		self.layers.append(CovMat(kernel='rbf', alpha_fixed=False))
		# self.layers.append(CovMat(kernel='rbf', alpha_fixed=False, alpha=self.noise))	# noise free
		# optimizer
		self.opt = Adam(1e-3)
		#self.opt=SciPyMin('l-bfgs-b')

	@time_process
	def find_collision(self, iter_num:int=1000, tol:float=1e-3):
		''' 
		Test problem in Latent space
		Return: if find the problem
		'''
		print("find latent collision start!")
		self.gp = NNRegressor(self.layers, opt=self.opt, batch_size=self.x_train.shape[0], maxiter=iter_num, gp=True, verbose=False)
		self.gp.fit(self.x_train, self.y_train)
		self.y_pred, self.std = self.gp.predict(self.x_test)
		
		# find same latent
		print("find latent collision")
		xt = [0.2, 0.2]
		zt = self.gp.fast_forward(xt)	# latent space
		yt = self.target_func(xt)
		z_test = self.gp.fast_forward(self.x_test)

		nearest = np.argmin(np.abs(z_test - zt))
		collision =  np.abs(self.gp.fast_forward(self.x_test[nearest]) - zt) < tol
		different = np.abs(yt - self.target_func(self.x_test[nearest])) > tol
		if collision and different:
			print(f"Iter Num: {iter_num} \n{'*'*50}\
			\nCloset Latent: \t{self.gp.fast_forward(self.x_test[nearest])}\t/ {zt},\
			\nInput:\t\t{self.x_test[nearest]})\t/ {xt},\
			\nOutput:\t\t{self.target_func(self.x_test[nearest])}\t/ {yt}")
			self.plot_collision()
		return collision

if __name__=='__main__':
	'''1d case'''
	# test = Collision_Problem(target_func1)
	# for i in tqdm(range(0,20,2),desc='Iter Num'):
	# 	if test.find_collision(iter_num=i):
	# 		tqdm.write("Break: ", i)
	# 		break
	# test.plot_collision() # plot result at the end if collision not found
	'''2d case'''
	test = Collision_Problem_2d(target_func3)
	test.find_collision(iter_num=5000)
	test.plot_collision() # plot result at the end if collision not found

