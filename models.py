import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input


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

        content_output = [vgg.get_layer(layer_name).output for layer_name in content_layer_names]
        style_output = [vgg.get_layer(layer_name).output for layer_name in style_layer_names]

        self.model = tf.keras.models.Model(vgg.input, [content_output, style_output])
