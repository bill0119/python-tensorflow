score = [4.0, 1.0, 2.0, 3.0]

import numpy as np

def softmax(x):
    y = np.array(x)
    return np.exp(y)/np.sum(np.exp(y), axis=0)

print(softmax(score))
