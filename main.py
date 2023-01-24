import tensorflow as tf
import matplotlib.pyplot as plt
from models import NeuralStyleTransfer

def load_image(image_path):
    dimension = 512
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image,
    tf.float32)
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    longest_dimension = max(shape)
    scale = dimension / longest_dimension
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    return image[tf.newaxis, :]

style_img = load_image("Data/van-gogh.jpg")
content_img = load_image("Data/me.jpg")


nst = NeuralStyleTransfer()

gen_img = nst.transfer(style_img, content_img)

plt.imshow(gen_img)
plt.show()


