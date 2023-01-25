import tensorflow as tf
import matplotlib.pyplot as plt
from models import NeuralStyleTransfer
import numpy as np
import tensorflow_hub as hub

def load_image(image_path):

    # Read image from path
    image = tf.io.read_file(image_path)

    # Decode the jpeg to tensor
    image = tf.image.decode_jpeg(image, channels=3)

    # Cast the dtypes of image to float32
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image
    image = tf.image.resize(image, (224,224))

    # Add a dimension to make it look like a batch
    return image[tf.newaxis, :]


style_img = load_image("Data/van-gogh.jpg")
content_img = load_image("Data/me.jpg")

nst = NeuralStyleTransfer()

gen_img = nst.transfer(tf.constant(style_img), tf.constant(content_img), epochs=100)


def tensor_to_image(tensor):
    tensor = tensor * 255

    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor) > 3:
        tensor = tensor[0]

    return tensor


def show_image(content_img, style_img, img):

    fig, ax = plt.subplots(1,3)

    ax[0].imshow(style_img)
    ax[1].imshow(content_img)

    if len(img.shape) > 3:
        img = img[0]

    ax[2].imshow(img)
    plt.show()


generated_img = tensor_to_image(gen_img)
show_image(generated_img)

module_url = ('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
hub_module = hub.load(module_url)

results = hub_module(tf.constant(content_img), tf.constant(style_img))
hub_generated_img = tensor_to_image(results[0])

show_image(hub_generated_img)

