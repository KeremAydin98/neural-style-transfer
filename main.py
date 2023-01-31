import tensorflow as tf
import matplotlib.pyplot as plt
from models import NeuralStyleTransfer
import numpy as np
import tensorflow_hub as hub


# Loading the images
style_img = load_image("Data/starry_night.jpg")
content_img = load_image("Data/night_sky.jpg")

# Transferring the style and content attributes to a noise image
nst = NeuralStyleTransfer()
gen_img = nst.transfer(tf.constant(style_img), tf.constant(content_img), epochs=100, image_size=448)

