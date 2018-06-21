# -*- coding: utf-8 -*

import tensorflow as tf

# 新建 2X2 矩阵常量
tensor_1 = tf.constant([[1., 2.], [3.,4]])

tensor_2 = tf.constant([[5.,6.],[7.,8.]])

# 新建矩阵乘法操作
output_tensor = tf.matmul(tensor_1, tensor_2)

# 必须在会话 (Session) 中运行计算图
sess = tf.Session()

result = sess.run(output_tensor)
print(result)

sess.close()


