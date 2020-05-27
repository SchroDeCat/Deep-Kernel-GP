import numpy as np

def target_func1(x):
    '''Increesing frequency'''
    return (x+0.5>=0)*np.sin(64*(x+0.5)**4)#-1.0*(x>0)+numpy.
    # return np.tan(0.9 * np.pi*x)#-1.0*(x>0)+numpy.