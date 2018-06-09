# -*- coding: utf-8 -*
import tensorflow as tf
import np as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
random_r = np.random.rand(2, 100)
print random_r
print "\n"
x_data = np.float32(random_r) # 随机输入
print x_data

y_data = np.dot([0.100, 0.200], x_data) + 0.300
print "\n"
print y_data
print "\n"
# 构造一个线性模型
# 
b = tf.Variable(tf.zeros([1]))
print "b-------"
print "\n"
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
print W
print "\n"
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
tf.assign()
sess.run(init)

# 拟合平面
for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]