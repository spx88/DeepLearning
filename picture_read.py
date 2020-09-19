import tensorflow as tf
import os


def picture_read(file_list):
    """狗图片读取案例"""
    # # 1.构造文件名队列
    file_queue = tf.train.string_input_producer(file_list)

    # # 2.读取与解码
    reader = tf.WholeFileReader()
    # key-文件名 value-一张图片的原始编码形式
    key, value = reader.read(file_queue)
    print("key:", key)
    print("value:", value)
    # 解码阶段
    image = tf.image.decode_jpeg(value)
    print("image:", image)

    # 图像的形状、类型修改
    image_resized = tf.image.resize_images(image, [200, 200])
    print("image_resized:", image_resized)

    # 静态形状改变
    image_resized.set_shape(shape=[200, 200, 3])
    print("形状确定后image_resized:", image_resized)

    # # 3.批处理
    img_batch = tf.train.batch([image_resized], batch_size=100, num_threads=1, capacity=100)
    print(img_batch)
    # 开启会话
    with tf.Session() as sess:
        # 开启线程
        # 线程协调员
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        key_new, value_new, image_new, image_resized_new, img_batch_new = sess.run(
            [key, value, image, image_resized, img_batch])
        print("key_new:", key_new)
        print("value_new", value_new)
        print("image_new", image_new)
        print("image_resized_new", image_resized_new)
        print("img_batch_new", img_batch_new)

        # 回收线程
        coord.request_stop()
        coord.join(threads)
    return None


if __name__ == '__main__':
    # 构造路径+文件名列表
    filename = os.listdir("./dog")
    print(filename)
    # 拼接路径+文件名
    file_list = [os.path.join("./dog/", file) for file in filename]
    print(file_list)
    picture_read(file_list)
