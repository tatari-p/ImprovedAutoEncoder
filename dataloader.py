import glob
import tensorflow as tf
from PIL import Image


def loader(dir_path, batch_size, img_size, min_after_dequeue=5000, gray_scale=False):
    file_paths = glob.glob("{}/*.{}".format(dir_path, "jpg"))
    filename_queue = tf.train.string_input_producer(list(file_paths), shuffle=True)
    reader = tf.WholeFileReader()
    _, data = reader.read(filename_queue)
    img = tf.image.decode_jpeg(data, channels=3)

    if gray_scale:
        img = tf.image.rgb_to_grayscale(img)

    with Image.open(file_paths[0]) as sample:
        shape = (*sample.size, 3)

    img.set_shape(shape)

    capacity = min_after_dequeue+batch_size*3
    queue = tf.train.shuffle_batch([img], batch_size=batch_size, num_threads=16, capacity=capacity, min_after_dequeue=min_after_dequeue)
    queue = tf.image.resize_nearest_neighbor(queue, img_size)

    return norm_img(tf.cast(queue, tf.float32))


def norm_img(int_img):
    float_img = int_img/255-0.5

    return float_img


def denorm_img(float_img):
    int_img = (float_img+0.5)*255
    int_img[int_img < 0] = 0
    int_img[int_img > 255] = 255

    return int_img