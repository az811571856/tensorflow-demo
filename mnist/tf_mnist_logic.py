# -*- coding: utf-8 -*

from mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#存None张图片，每张图片784个像素点
x = tf.placeholder(tf.float32, [None, 784])

#权重：每个像素对应每个数字都有一个权重。w[i,j] 第i个像素 第j个数字的权重
W = tf.Variable(tf.zeros([784, 10]))

#偏置量：对应每个数字
b = tf.Variable(tf.zeros([10]))

#假设模型
y = tf.nn.softmax(tf.matmul(x,W) + b)

#真实概率值。None张图片（与上面的None张图片对应），10个数字概率
y_ = tf.placeholder("float", [None,10])

#代价函数： -求和（实际的概率分布*log(预测的概率分布)）
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#y_是None*10 二维，y 是10*1 ?

#优化：自动选择反向传播优化？
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化我们创建的变量
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#模型循环训练1000次
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#评估模型：y_ 与 y进行比较，如果某个图片最大的下标索引（预测正确），则表示预测正确。累加求和可以求出正确的%比

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



