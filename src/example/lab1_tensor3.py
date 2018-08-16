import tensorflow as tf

tri1 = [5.0, 6.0, 7.0]
tri2 = [3.0, 4.0, 5.0]

def computeArea(sides):
    p = sides[:, 0]
    q = sides[:, 1]
    r = sides[:, 2]
    s = (p + q + r) / 2
    areasquare = s * (s - p) * (s - q) * (s - r)
    return areasquare**0.5

with tf.Session() as session1:
    area = computeArea(tf.constant([tri1, tri2]))
    result1 = session1.run(area)