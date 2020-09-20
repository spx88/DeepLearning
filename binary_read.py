import tensorflow as tf
import os


class Cifar(object):

    def __init__(self):
        # 初始化操作
        self.height = 32
        self.width = 32
        self.channels = 3

        # 字节数
        self.image_bytes = self.height * self.width * self.channels
        self.label_bytes = 1
        self.all_bytes = self.label_bytes + self.image_bytes

    def read_and_decode(self, file_list):
        # 1构造文件名队列
        file_queue = tf.train.string_input_producer(file_list)
        # 2读取与解码
        # 读取阶段
        reader = tf.FixedLengthRecordReader(self.all_bytes)
        # key文件名 value一个样本
        key, value = reader.read(file_queue)
        print("key:", key)
        print("value", value)

        # 解码阶段
        decoded = tf.decode_raw(value, tf.uint8)
        print("decoded:\n", decoded)

        # 将目标值和特征值切片切开
        label = tf.slice(decoded, [0], [self.label_bytes])
        image = tf.slice(decoded, [self.label_bytes], [self.image_bytes])
        print("label:\n", label)
        print("image\n", image)

        # 调整图片形状
        tf.reshape(image, shape=[self.channels, self.height, self.width])
        image_reshaped = tf.reshape(image, shape=[self.channels, self.height, self.width])
        print("image_reshaped:\n", image_reshaped)

        # 转置，将图片的顺序为height,weidth,channels
        image_transposed = tf.transpose(image_reshaped, [1, 2, 0])
        print("image_transposed:\n", image_transposed)

        # 调整图像类型
        image_cast = tf.cast(image_transposed, tf.float32)

        # 3批处理
        label_batch, image_batch = tf.train.batch([label, image_cast], batch_size=100, num_threads=1, capacity=100)
        print("label_batch:", label_batch)
        print("image_batch:", image_batch)

        # 开启会话
        with tf.Session() as sess:
            # 开启线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            key_new, value_new, decoded_new, label_new, image_new, image_reshaped_new, image_transposed_new, label_value, image_value = sess.run(
                [key, value, decoded, label, image, image_reshaped, image_transposed, label_batch, image_batch])
            print("key_new", key_new)
            print("value_new", value_new)
            print("decoded_new", decoded_new)
            print("label_new", label_new)
            print("image_new", image_new)
            print("image_reshaped_new", image_reshaped_new)
            print("image_transposed_new", image_transposed_new)
            print("label_value:", label_value)
            print("image_value:", image_value)
            # 回收线程
            coord.request_stop()
            coord.join(threads)

        return None


if __name__ == '__main__':
    file_name = os.listdir("./cifar-10-batches-py")
    print("file_name", file_name)
    # 构造文件名路径列表
    file_list = [os.path.join("./cifar-10-batches-py/", file) for file in file_name if file[-3:] == "bin"]
    print("file_list:\n", file_list)

    # 实例化Cifar
    cifar = Cifar()
    cifar.read_and_decode(file_list)
