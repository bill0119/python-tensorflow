import tensorflow as tf
import numpy as np

print(tf.__version__)
print(np.__version__)
a = np.array([4, 3, 9])
b = np.array([-1, -2, -3])
c = np.add(a, b)
print(c)
ta = tf.constant([4, 3, 9])
tb = tf.constant([-1, -2, -3])
tc = tf.add(a, b)
print(ta,tb)
print(tc)