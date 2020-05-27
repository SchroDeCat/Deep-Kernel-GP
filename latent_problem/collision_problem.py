import numpy as np
import matplotlib.pyplot as plt
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

	def __init__(self, f, train_num=100):
		'''Init (1)training and testing data (2) form the network'''
		np.random.seed(0)
		self.target_func = f
		self.x_train=np.random.random(size=(train_num,1)) - .5
		self.y_train=f(self.x_train)+np.random.normal(0.0,0.01,size=self.x_train.shape)
		self.x_test=np.linspace(-.7, .0, 10000).reshape(-1,1)
		self.x_plot=np.linspace(-.7, .7, 10000).reshape(-1,1)
		self.model()

	def model(self,):
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
		self.layers.append(Dense(1))
		# self.layers.append(Scale(fixed=True, init_vals=64.0))
		self.layers.append(CovMat(kernel='rbf', alpha_fixed=False))
		# optimizer
		self.opt=Adam(1e-3)
		#self.opt=SciPyMin('l-bfgs-b')

	@time_process
	def find_collision(self, iter_num:int=10, tol:float=1e-3):
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

if __name__=='__main__':
	test = Collision_Problem(target_func1)
	for i in tqdm(range(0,20,2),desc='Iter Num'):
		if test.find_collision(iter_num=i):
			tqdm.write("Break: ", i)
			break
		test.plot_collision() # plot result at the end if collision not found


