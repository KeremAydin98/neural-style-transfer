import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import numpy as np

class NeuralStyleTransfer:

    def __init__(self):

        super().__init__()

        vgg = VGG19(weights="imagenet", include_top=False)
        vgg.trainable = False

        style_layer_names = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1"
        ]
        content_layer_names = [
            "block5_conv2"
        ]

        outputs = [vgg.get_layer(layer_name).output for layer_name in (style_layer_names + content_layer_names)]

        self.model = tf.keras.models.Model([vgg.input], outputs)
    def calc_content_loss(self, content_img, gen_img):

        losses = []
        for layer_name in [self.model.layers[-1]]:

            content_layer = self.model.get_layer(layer_name)
            con = content_layer(content_img)
            gen = content_layer(gen_img)

            losses.append(np.square(con-gen))

        return tf.reduce_mean(losses)

    def calc_style_loss(self, style_img, gen_img):










