import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input


class NeuralStyleTransfer:

    def __init__(self,
                 content_layers=None,
                 style_layers=None):

        # Initialize the pretrained model of VGG19
        self.base_model = VGG19(weights='imagenet', include_top=False)
        self.base_model.trainable = False

        # Weights of style and content
        self.style_weight = 1e-2
        self.content_weight = 1e3
        self.tv_loss = 1e-2

        # Layer names for content and style
        if content_layers is None:
            self.content_layers = ["block2_conv2"]
        else:
            self.content_layers = content_layers
        if style_layers is None:
            self.style_layers = ['block1_conv1',
                                 'block2_conv1',
                                 'block3_conv1',
                                 'block4_conv1',
                                 'block5_conv1']
        else:
            self.style_layers = style_layers

        # Outputs of given style and content layers
        outputs = [self.base_model.get_layer(name).output for name in (self.style_layers + self.content_layers)]

        # Whole model with inputs and outputs
        self.whole_model = tf.keras.models.Model([self.base_model.input], outputs)

    def gram_matrix(self, input_tensor):

        result = tf.linalg.matmul(input_tensor, input_tensor, transpose_a=True)
        # result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)

        input_shape = tf.shape(input_tensor)

        num_locations = float(np.prod(np.array(input_shape[1:3])))

        result = result / num_locations

        return result

    def calc_outputs(self, inputs):

        inputs = 255 * inputs

        preprocessed_input = preprocess_input(inputs)

        outputs = self.whole_model(preprocessed_input)

        style_outputs = outputs[:len(self.style_layers)]
        content_outputs = outputs[len(self.style_layers):]

        style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]

        return content_outputs, style_outputs

    @staticmethod
    def compute_loss(outputs, targets):

        # tf.math.add_n: returns the element-wise sum of a list of tensors
        return tf.add_n([tf.reduce_mean((outputs[key] - targets[key]) ** 2) for key in range(len(outputs))])

    @staticmethod
    def compute_tv_loss(output):

        return tf.reduce_sum(tf.image.total_variation(output))

    def calc_total_loss(self, output, content_outputs, style_outputs, style_targets, content_targets):

        style_loss = self.compute_loss(style_outputs, style_targets)
        style_loss *= (self.style_weight / len(self.style_layers))

        content_loss = self.compute_loss(content_outputs, content_targets)
        content_loss *= (self.content_weight / len(self.content_layers))

        tv_loss = self.compute_tv_loss(output)
        tv_loss *= tv_loss * self.tv_loss

        return style_loss + content_loss + tv_loss, style_loss, content_loss, tv_loss

    def train(self, image, style_targets, content_targets, epochs):

        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-2, beta_1=0.99, epsilon=0.1)

        for epoch in range(epochs):

            with tf.GradientTape(persistent=True) as tape:
                content_outputs, style_outputs = self.calc_outputs(image)
                loss, style_loss, content_loss, tv_loss = self.calc_total_loss(image, content_outputs, style_outputs,
                                                                               style_targets, content_targets)

            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch: {epoch + 1}/{epochs}, Style Loss: {style_loss}, Content Loss: {content_loss}, Tv Loss: {tv_loss}")

            img_gradient = tape.gradient(loss, image)
            optimizer.apply_gradients([(img_gradient, image)])

            image.assign(tf.clip_by_value(image, 0.0, 1.0))

        return image

    def transfer(self, style_image, content_image, epochs=1000):

        _, style_targets = self.calc_outputs(style_image)
        content_targets, _ = self.calc_outputs(content_image)

        image = tf.random.uniform((1, 224, 224, 3))
        image = tf.Variable(image)

        image = self.train(image, style_targets, content_targets, epochs)

        return image
