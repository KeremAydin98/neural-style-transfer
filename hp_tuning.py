import tensorflow as tf
import matplotlib.pyplot as plt
from models import NeuralStyleTransfer
from helpful_functions import *
import numpy as np
import tensorflow_hub as hub


# Loading the images
style_img = load_image("Data/starry_night.jpg")
content_img = load_image("Data/night_sky.jpg")

style_weights = [1e-4, 1e-3, 1e-2, 1e-1]
content_weights = [1e4, 1e3, 1e2, 1e1]
tv_weights = [1e-7, 1e-6, 1e-5]

for k, tv_weight in enumerate(tv_weights):
  fig, ax = plt.subplots(4,4,figsize=(12,12))
  fig.tight_layout(rect=[0, 0.03, 1, 0.95])

  for i, content_weight in enumerate(content_weights):
    for j, style_weight in enumerate(style_weights):

      nst = NeuralStyleTransfer(style_weight, content_weight, tv_weight)

      gen_img = nst.transfer(tf.constant(style_img), tf.constant(content_img), epochs=1000)

      stylized_img = tensor_to_image(gen_img[0])

      ax[i,j].imshow(stylized_img)
      ax[i,j].axis('off')
      ax[i,j].set_title(f"Content Weight: {content_weight}\nStyle Weight: {style_weight}")

  plt.suptitle(f"Total variation loss: {tv_weight}")
  plt.savefig(f'/content/drive/MyDrive/Colab Notebooks/Projects/neuralStyleTransfer/{k}_image.png')
  plt.show()

# Loading the images
style_img = load_image("Data/starry_night.jpg", 224)
content_img = load_image("Data/night_sky.jpg", 224)

# Load the pretrained neural style transfer model from tf_hub to compare
module_url = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(module_url)

# Start the pretrained neural style transfer
results = hub_module(tf.constant(content_img), tf.constant(style_img))
hub_generated_img = tensor_to_image(results[0])

# Display the generated image by pretrained model
show_image(hub_generated_img)

