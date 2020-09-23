import tensorflow as tf
import scipy.misc
from tensorflow.examples.tutorials.mnist import input_data

import os


def full_connection():
    """用全连接来对手写数字进行识别"""
    # 1 准备数据
    # savedir = './mnist'
    mnist = input_data.read_data_sets("./mnist", one_hot=True)
    x = tf.placeholder(dtype=tf.float32, shape=(None, 784))
    y_true = tf.placeholder(dtype=tf.float32, shape=(None, 10))

    # 2 构建模型
    Weights = tf.Variable(initial_value=tf.random_normal(shape=[784, 10]))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[10]))
    y_predict = tf.matmul(x, Weights) + bias

    # 3构造损失函数
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4 优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(error)

    # 5准确率计算
    #     比较输出的结果最大值所在位置和真实值的最大值所在位置

    # 　　axis = 0, 时比较每一列的元素，将每一列最大元素所在的索引记录下来，最后输出每一列最大元素所在的索引数组。
    # 　axis = 1, 的时候，将每一行最大元素所在的索引记录下来，最后返回每一行最大元素所在的索引数组。先找到预测概率值最大值所在的位置，
    # 因为这里数据集采用one—hot编码来确定目标值
    equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
    # 求平均,这里通过equal输出的布尔类型（0或1），通过cast转化为浮点型的0或1
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 定义saver
    saver = tf.train.Saver()

    # 初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        # 每次重训练集中抽取100个样本
        image, label = mnist.train.next_batch(100)

        print("训练之前，损失为%f" % sess.run(error, feed_dict={x: image, y_true: label}))

        # 开始训练
        for i in range(500):
            _, loss, accuracy_value = sess.run([optimizer, error, accuracy], feed_dict={x: image, y_true: label})
            print("第%d次的训练，损失为%f,准确率为%f" % (i + 1, loss, accuracy_value))
        saver.save(sess, "./tmp/model/fully.ckpt")

        return None


if __name__ == '__main__':
    # print()
    full_connection()
