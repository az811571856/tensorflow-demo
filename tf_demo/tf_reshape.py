import tensorflow as tf
import numpy as np

a = tf.constant(1)
b = tf.constant(2)
cs = tf.reshape([1, 2, 3, 4, 5, 6], (6, 1))
# mm = tf.matmul(c, cs)
print(cs)
with tf.Session():
    print(cs.eval())
    # print(mm.eval())

with tf.Session():
    print(tf.add(a, b))

a = tf.constant([1, 2])
b = tf.constant([2, 4])
sess = tf.Session()
print(sess.run(tf.add(a, b)))
sess.close()

#
# sess = tf.Session()
# print(sess.run(tf.sin(a)))
# sess.close()

sess = tf.Session()
print(sess.run(tf.sin(tf.cast(a, dtype=tf.float32))))
sess.close()

a_float = tf.cast(a, dtype=tf.float32)
sess = tf.Session()
print(sess.run(a_float))
print(sess.run(tf.exp(a_float)))
print(sess.run(tf.log(tf.exp(a_float))))
sess.close()


a = tf.constant(2, shape=[2, 2])
b = tf.constant(3, shape=[2, 2])
sess = tf.Session()
print(sess.run(a))
print(sess.run(b))
print(sess.run(tf.matmul(a, b)))
print(sess.run(a * b))
print(sess.run(tf.mul(a, b)))
sess.close()


rand_normal = tf.random_normal([100])
with tf.Session():
    # print(rand_normal.eval())
    print('----')

W = tf.Variable(10)
sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
init.run()
print(W.eval(sess))
sess.close()


#
W = tf.Variable(10)
sess = tf.InteractiveSession()
sess.run(W.initializer)
print(W.eval(sess))
sess.close()


#
W = tf.Variable(10)
sess = tf.InteractiveSession()
W.initializer.run()
print(W.eval(sess))
sess.close()


#
W = tf.Variable(10)
sess = tf.InteractiveSession()
W.initializer.run()
W.assign(100)
print(W.eval())
sess.close()

#
W = tf.Variable(10)
sess = tf.InteractiveSession()
W.initializer.run()
# after op
W.assign(100).eval()
print(W.eval())
sess.close()


# bb = tf.placeholder(tf.float32, shape=(2, 2))
# sess = tf.Session()
# bbb = np.array([[1, 23], [1, 23]])
# print(sess.run(bb, feed_dict={bb: bbb}))
# sess.close()

x = tf.placeholder(tf.float32, shape=(2, 2))
# y = tf.matmul(x, x)

with tf.Session() as sess:
    rand_array = np.random.rand(2, 2)
    print(sess.run(x, feed_dict={x: rand_array}))  # Will succeed.



