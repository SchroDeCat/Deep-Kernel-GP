import numpy as np
import matplotlib.pyplot as plt
from time import sleep

from dknet import NNRegressor
from dknet.layers import Dense,CovMat,Dropout,Parametrize,Scale
from dknet.optimizers import Adam,SciPyMin,SDProp
from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,ConstantKernel
def f(x):
	return (x+0.5>=0)*np.sin(64*(x+0.5)**4)#-1.0*(x>0)+numpy.
	# return np.tan(0.9 * np.pi*x)#-1.0*(x>0)+numpy.

np.random.seed(0)
x_train=np.random.random(size=(70,1)) - .5
y_train=f(x_train)+np.random.normal(0.0,0.01,size=x_train.shape)

layers=[]
layers.append(Dense(6,activation='tanh'))
layers.append(Dropout(0.99))
layers.append(Dense(1))
layers.append(Scale(fixed=True,init_vals=64.0))
layers.append(CovMat(kernel='rbf',alpha_fixed=False))

opt=Adam(1e-3)
x_test=np.linspace(-.7, .0, 10000).reshape(-1,1)
x_plot=np.linspace(-.7, .7, 10000).reshape(-1,1)


#opt=SciPyMin('l-bfgs-b')
def test_latent_space(iter_num = 10):
	gp=NNRegressor(layers,opt=opt,batch_size=x_train.shape[0],maxiter=iter_num,gp=True,verbose=False) # insufficient training
	gp.fit(x_train,y_train)
	
	y_pred, std = gp.predict(x_test)
	# find same latent
	test_x = 0.2
	test_tartget = gp.fast_forward(test_x)
	z_test = gp.fast_forward(x_test)

	nearest = np.argmin(np.abs(z_test - test_tartget))
	flag =  np.abs(gp.fast_forward(x_test[nearest]) - test_tartget) < 1e-3
	if flag:
		print(f"Iter Num: {iter_num} \n{'*'*50}\
		\nCloset Latent: \t{gp.fast_forward(x_test[nearest])}\t/ {test_tartget},\
		\nInput:\t\t{x_test[nearest]})\t/ {test_x},\
		\nOutput:\t\t{f(x_test[nearest])}\t/ {f(np.ones(1) * test_x)}")

		plt.plot(x_plot,gp.fast_forward(x_plot))
		plt.xlabel('X')
		plt.ylabel('Z')
		plt.title("Latent Map")
		plt.figure()

		plt.plot(x_train,y_train,'.')
		plt.plot(x_plot,f(x_plot)[:,0])
		plt.plot(x_plot,y_pred)
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.fill_between(x_test[:,0],y_pred[:,0]-std,y_pred[:,0]+std,alpha=0.5)
		plt.title("Prediction")
		plt.legend(['Training samples', 'True function', 'Predicted function','Prediction stddev'])
		plt.show()
	return flag

if __name__=='__main__':
	for i in range(0,20,2):
		if test_latent_space(i):
			print("Break: ", i)
			break

	# plt.plot(x_plot,gp.layers[-2].out)

