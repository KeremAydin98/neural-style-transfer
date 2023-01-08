from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import *
import numpy as np
import cv2

class NeuralStyleTransfer:

    def __init__(self, alpha=0.9, beta=0.1):

        super().__init__()

        self.alpha = alpha
        self.beta = beta

        self.base_model = VGG19(weights="imagenet", include_top=False)
        self.base_model.trainable = False

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

        all_layer_names = style_layer_names + content_layer_names

        outputs = [self.base_model.get_layer(layer_name).output for layer_name in all_layer_names]

        self.model = tf.keras.models.Model([self.base_model.input], outputs)

    def calc_gram_matrix(self, feature_maps):

        gram = feature_maps[0].reshape(-1,1)
        for feature_map in feature_maps[1:]:

            feature_map = feature_map.reshape(-1,1)
            gram = np.concatenate((gram,feature_map))

        return gram

    def calc_content_loss(self, content_img, gen_img):

        losses = []
        for layer_name in [self.model.layers[-1]]:

            content_layer = self.model.get_layer(layer_name.name)
            con = content_layer(content_img)
            gen = content_layer(gen_img)

            losses.append(tf.pow((con-gen),2))

        return tf.reduce_mean(losses)

    def calc_style_loss(self, style_img, gen_img):

        losses = []
        for layer_name in [self.model.layers[:-1]]:
            style_layer = self.model.get_layer(layer_name.name)

            styl = style_layer(style_img)
            gen = style_layer(gen_img)

            _,h,w,c = gen.get_shape().as_list()

            styl_gram = self.calc_gram_matrix(styl)
            gen_gram = self.calc_gram_matrix(gen)

            styl_gram = tf.linalg.matmul(styl_gram, styl_gram, transpose_a=True)
            gen_gram = tf.linalg.matmul(gen_gram, gen_gram, transpose_a=True)

            losses.append(tf.pow((styl_gram - gen_gram),2))

        sum_losses = tf.reduce_sum(losses)

        return sum_losses * (1 / (4 * (c * h * w) ** 2))

    def calc_total_loss(self, content_img, style_img, gen_img):

        content_loss = self.calc_content_loss(content_img, gen_img)
        style_loss = self.calc_style_loss(style_img, gen_img)

        return self.alpha * content_loss + self.beta * style_loss

    def style_transfer(self, content_img, style_img):

        step_size = 0.01
        steps = 100

        # Generates a noisy image
        gen_img = np.random.uniform(0, 255, (1, 224, 224, 3)).astype("float32")

        content_img = preprocess_input(content_img)
        style_img = preprocess_input(style_img)

        content_img = content_img * 255
        style_img = style_img * 255

        for _ in tqdm(range(steps)):

            with tf.GradientTape() as tape:
                total_loss = self.calc_total_loss(content_img, style_img, gen_img)

            gradient = tape.gradient(total_loss, gen_img)

            gen_img = gen_img + step_size * gradient

        return gen_img













