import numpy as np
from itertools import product

def sample_data(data, sample_num = 10000):
    """clean and sample original product data"""
    train_data = data.astype("float32")
    compond_np = np.array(list(product(train_data, train_data)))
    filter = np.min(np.abs(compond_np[:,1,:]-compond_np[:,0,:]),axis=-1).astype(np.bool)
    compond_np_clear = compond_np[filter]
    try:
        random_filter = np.random.choice(compond_np_clear.shape[0], sample_num)  
        compond_np_sampled = compond_np_clear[random_filter]
    except:
        print("ERROR: not enough data for sampling!")
        compond_np_sampled = compond_np_clear
    return compond_np_sampled