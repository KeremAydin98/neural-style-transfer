from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import cv2

class NeuralStyleTransfer:

    def __init__(self, alpha=0.9, beta=0.1):

        super().__init__()

        self.alpha = alpha
        self.beta = beta

        self.base_model = VGG19(weights="imagenet", include_top=False)
        self.base_model.trainable = False

        self.style_layer_names = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1"
        ]
        self.content_layer_names = [
            "block5_conv2"
        ]

        all_layer_names = self.style_layer_names + self.content_layer_names

        outputs = [self.base_model.get_layer(layer_name).output for layer_name in all_layer_names]

        self.model = tf.keras.Model([self.base_model.input], outputs)












