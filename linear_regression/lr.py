# coding: utf-8
# 线性回归 tensorflow 实现，有图形显示
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 展示训练数据与，预测数据
def show(e, x_train, y_train, y_pred_numpy, loss_numpy):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    fig.show()
    fig.canvas.draw()
    ax.clear()
    ax.plot(x_train, y_train, 'bo', label='real')
    ax.plot(x_train, y_pred_numpy, 'ro', label='estimated')
    ax.legend()
    fig.canvas.draw()
    plt.pause(0.5)
    print('epoch: {}, loss: {}'.format(e, loss_numpy))

# 生成原始数据
x_data = np.random.rand(20)
print("# xdata {}".format(x_data))
y_data = 10*x_data + np.random.randint(1, 10, 20)
print("# ydata {}".format(y_data))
# 图形展示原始数据
plt.figure()
plt.plot(x_data, y_data, 'bo')
# plt.show() # 显示默认关掉
# 原始数据转换成tensor
x = tf.cast(tf.constant(x_data, name='x'), dtype='float32')
y = tf.cast(tf.constant(y_data, name='y'), dtype='float32')
# 定义线性模型
w = tf.Variable(initial_value=tf.random_normal(shape=()), name='weight')
b = tf.Variable(0.0, name='bias')
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
print("# 初始参数 {}".format(sess.run((w, b))))
with tf.variable_scope('Linear_Model'):
    y_pred = w * x + b
    y_pred_numpy = sess.run(y_pred)

# 图形展示原始与模型初始数据
plt.plot(x_data, y_data, 'bo')
plt.plot(x_data, y_pred_numpy, 'ro')
plt.legend()
# plt.show() # 显示默认关掉
# 构造目标函数
loss = tf.reduce_mean(tf.square(y_pred - y))
print("# 初始误差 {}".format(sess.run(loss)))
# 目标函数的梯度(参数变化速率)
w_grad, b_grad = tf.gradients(loss, [w, b])
print("# w, b初始梯度 {}".format(sess.run((w_grad, b_grad))))
# 更新参数(w参数初始值 - 学习率*初始梯度)
lr = 1e-1  # 学习率
w_update = w.assign_sub(lr * w_grad)
b_update = b.assign_sub(lr * b_grad)
# 第一次更新参数值
print("#首次更新参数值w b{} ".format(sess.run((w_update, b_update))))
# 多次更新参数值
for i in range(50):
    print("# "+str(i)+"次更新参数值w b{} ".format(sess.run((w_update, b_update))))
    show(i, x_data, y_data, sess.run(y_pred), sess.run(loss))


