#  coding: utf-8

import tensorflow as tf
import numpy as np

# 变形
a = tf.constant([1, 2, 3, 4])
b = tf.constant([1, 2, 3, 4, 5, 6])
cs1 = tf.reshape(a, (2, 2))
cs2 = tf.reshape(b, (2, 3))
# 矩阵乘法
mm1 = tf.matmul(cs1, cs2)
# 点乘
mm2 = tf.mul(cs2, tf.constant([1, 2, 3]))
with tf.Session() as sess:
    # print(mm.eval())
    print(cs1.eval())
    print(cs2.eval())
    print(sess.run(mm1))
    print("----")
    print(sess.run(mm2))

# with tf.Session():
#     print(tf.add(a, b))

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


print("sess.run的类型是什么")
a = tf.constant(2, shape=[2, 2])
b = tf.constant(3, shape=[2, 2])
sess = tf.Session()
print(type(a))
py_a = sess.run(a)
print(type(py_a))
print('----+++++')
print(sess.run([a, b]))
print(sess.run(tf.matmul(a, b)))
print(sess.run(a * b))
print(sess.run(tf.mul(a, b)))
sess.close()


rand_normal = tf.random_normal([100])
with tf.Session():
    # print(rand_normal.eval())
    print('----')

#
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


# 变量
print('变量assgin后生成新的对象')
W = tf.Variable([10, 20])
sess = tf.InteractiveSession()
W.initializer.run()
W.assign([100, 200])
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
y = tf.matmul(x, x)

with tf.Session() as sess:
    rand_array = np.random.rand(2, 2)
    print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.


name = tf.constant("hello world", name="hello")
name1 = tf.constant("hello world1", name="hello1")
# logX =
ab = tf.matmul(a, b, name="ab")
with tf.Session() as sess:
    print("{} : {}".format("44", "xxxxx"))
    print("{} : {}".format(name1.name, "xxxxx"))
    print("{} - {} - {}".format(ab.name, type(ab), type(sess.run(ab))))
    print(sess.run(tf.exp(tf.div(1.0, 2.0))))
    print(sess.run(tf.log(tf.constant(100.0))))

# sigmoid 函数构造
e1 = tf.constant(1.0)
ex = tf.placeholder(dtype='float32')
with tf.Session() as sess:
    print(sess.run(tf.div(e1, tf.add(e1, tf.exp(-ex))), feed_dict={ex: 0.0}))

# 构造0 or 1矩阵
zeros = tf.ones((2, 3), dtype=tf.int32)
zeros_like = tf.zeros_like(zeros)
sess = tf.Session()
print(sess.run(zeros))
print(sess.run(zeros_like))
sess.close()

# 填充矩阵
print('# 填充矩阵')
fill = tf.fill((2, 4), 2)
sess = tf.Session()
print(sess.run(fill))
sess.close()

#把数字分成几份
print('# 把数字分成几份')
linespace = tf.linspace(1.0, 5.0, 7)
print("type is {}".format(type(linespace)))
sess = tf.Session()
print(linespace)
print(sess.run(linespace))
sess.close()

#获得tf的图
print('# 获得tf的图')
a = tf.constant(1, name="xxxx")
b = tf.constant(2)
c = tf.sub(a, b)
g = tf.get_default_graph()
print(g)
for op in g.get_operations():
    cc = 1
    # print(op.name)
print(a.name)

print('# 通过名字获得张量')
hello_tensor = g.get_tensor_by_name("xxxx:0")
with tf.Session() as sess:
    print(sess.run(hello_tensor))

print('# 新图')
g1 = tf.Graph()
print(g1)
with g1.as_default():
    a2 = tf.constant(23)
    print(tf.get_default_graph())
print(a.graph)
print(a2.graph)

with tf.Session() as sess:
    graph_writer = tf.tensorflow_dot_core_dot_framework_dot_summary__pb2.DESCRIPTOR



