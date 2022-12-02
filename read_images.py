import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image


def load_img(path_to_img):


  img = Image.open(path_to_img)
  img = kp_image.img_to_array(img)

  # We need to broadcast the image array such that it has a batch dimension 
  img = np.expand_dims(img, axis=0)
  

  return img

def imshow(img, title=None):
  # Remove the batch dimension
  out = np.squeeze(img, axis=0)
  # Normalize for display 
  out = out.astype('uint8')
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out)

#convert images from RGB to BGR with zero centered(preprocess_for_VGG)"
def load_and_process_img(path_to_img):
  img = Image.open(path_to_img)
  img = np.expand_dims(img, axis=0)
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img

#deprocess the final image from BGR to RGB with no-zero-centered(inverse of the preprocessing step)
def deprocess_img(processed_img):

  x = processed_img.copy()
  x = np.squeeze(x, 0)
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  # BGR To RGB [1,2,3] --> [3,2,1]
  x = x[:, :, ::-1]


  x = np.clip(x, 0, 255).astype('uint8') #Normalizes the image to 0-255
  return x

content_layers = ['block4_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
