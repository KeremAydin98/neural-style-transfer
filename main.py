import tensorflow as tf
import matplotlib.pyplot as plt
from models import NeuralStyleTransfer

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

gen_img = nst.transfer(style_img, content_img)

plt.imshow(gen_img)
plt.show()


