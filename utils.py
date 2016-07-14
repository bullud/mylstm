import numpy as np
from theano import config

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)
