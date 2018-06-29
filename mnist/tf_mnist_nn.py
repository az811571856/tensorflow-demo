# coding:utf-8
# 神经网络识别手写数字
import input_data
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取数据，分为训练集，测试集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
train_set = mnist.train
test_set = mnist.test


# 创建网格子图： 6*2 个，用来放置手写图片
fig, axes = plt.subplots(ncols=6, nrows=2)
# 调整子图间隔
plt.tight_layout(w_pad=-2.0, h_pad=-8.0)

# 取一些进行图形化 images:12,784  labels:12,10
images, labels = train_set.next_batch(12)

# 逐个提取
# zip:[([784],[10]), ().... ()]   12个tuple
for ind, (image, label) in enumerate(zip(images, labels)):
    # image 1，784 -> 28,28
    image = image.reshape((28, 28))

    # 取出代表的数字
    label = label.argmax()
    row = ind // 6
    col = ind % 6
    axes[row][col].imshow(image, cmap='gray')
    axes[row][col].axis('on')
    axes[row][col].set_title('%d' % label)


def hidden_layer(layer_input, output_depth, scope='hidden_layer', reuse=None):
    """
    构造单层隐藏层（不包含激活函数）

    :param layer_input: 输入层数据
    :param output_depth:  隐藏层个数
    :param scope: 变量名称
    :param reuse:
    :return:
    """
    input_depth = layer_input.get_shape()[-1]  # 输入层个数
    with tf.variable_scope(scope, reuse=reuse):
        w = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                            shape=(input_depth, output_depth),
                            name='weights')
        b = tf.get_variable(initializer=tf.constant_initializer(0.1),
                            shape=(output_depth),
                            name='bias')
        net = tf.matmul(layer_input, w) + b
        return net


def DNN(x, output_depths, scope='DNN', reuse=None):
    """
    神经网络
    :param x:  原始数据
    :param output_depths:  隐藏层输出层个数向量
    :param scope:
    :param reuse:
    :return:
    """
    # 添加隐藏层
    net = x
    for e, output_depth in enumerate(output_depths):
        net = hidden_layer(net, output_depth, scope='hiddenLayer%d' % e, reuse=reuse)
        net = tf.nn.relu(net)
    # 添加输出层
    net = hidden_layer(net, 10, 'outLayer', reuse)

    return net


# 输入输出的占位符
input_ph = tf.placeholder(shape=(None, 784), dtype=tf.float32)
label_ph = tf.placeholder(shape=(None, 10), dtype=tf.int8)

# 输入数据带入神经网络。隐藏层为3层，个数分别为 400 200 100
dnn = DNN(input_ph, [400, 200, 100])

# 定义损失函数
loss = tf.losses.softmax_cross_entropy(label_ph, dnn)

# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss)

# 准确度
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(dnn, axis=-1),
                                      tf.argmax(label_ph, axis=-1)),
                             dtype=tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
# 循环训练
for i in range(20000):
    # 训练变量
    images, labels = train_set.next_batch(64)
    sess.run(train_op, feed_dict={input_ph: images, label_ph: labels})
    # 打印中间结果
    if i % 1000 == 0:
        # 验证在测试集上的效果
        # train_loss, train_acc = sess.run((loss, acc), feed_dict={input_ph: images, label_ph: labels})
        images_test, labels_test = test_set.next_batch(64)
        test_loss, test_acc = sess.run((loss, acc), feed_dict={input_ph: images_test, label_ph: labels_test})
        print("第 {} 次，损失值：{:.6f}，准确度 {:.3f}".format(i+1, test_loss, test_acc))


images_test, labels_test = test_set.next_batch(10000)
test_loss, test_acc = sess.run((loss, acc), feed_dict={input_ph: images_test, label_ph: labels_test})
print("损失值：{:.6f}，准确度 {:.3f}".format(test_loss, test_acc))