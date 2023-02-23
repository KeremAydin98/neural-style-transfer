import tensorflow as tf
import matplotlib.pyplot as plt
from models import NeuralStyleTransfer
import cv2
import os
from helpful_functions import *

# Loading the images
style_img = load_image("Data/starry_night.jpg")
content_img = load_image("Data/night_sky.jpg")

fourcc = cv2.VideoWriter_fourcc(*'MP4V')

# Transferring the style and content attributes to a noise image
nst = NeuralStyleTransfer(style_weight=1e-1, content_weight=1e3, tv_weight=1e-6)
images = nst.content_transfer_only(tf.constant(content_img), epochs=20000, image_size=448)


video = cv2.VideoWriter("cTransfer.mp4", fourcc, 20, (448,448))

for image in images:
    video.write(image)


