import tensorflow as tf

# define a tensor for calculation
ta = tf.placeholder(dtype=tf.int32, shape=(None,))
tb = tf.placeholder(dtype=tf.int32, shape=(None,))
tc = tf.add(ta, tb)
print(type(ta), type(tb), type(tc))
# do real calculation
with tf.Session() as session1:
    result1 = session1.run(tc, feed_dict={
        ta: [4, 3, 9],
        tb: [-1, -2, -3]
    })
    print(result1)