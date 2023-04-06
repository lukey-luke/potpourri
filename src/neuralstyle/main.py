# import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
# import IPython.display as display

import numpy as np
import PIL.Image
import time
import functools

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def show_img(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    print('calling plt.imshow()... ')
    plt.imshow(image)
    if title:
        plt.title(title)

def main():
    style_path = '../../twister_toque/PXL_20211013_055448184.jpg'
    content_path = '../../twister_toque/rapids.jpeg'
    print('loading imgs')
    style_img = load_img(style_path)
    content_img = load_img(content_path)
    print('plotting content')
    plt.subplot(1, 2, 1)
    show_img(content_img, 'Content Image')
    print('plotting style')
    plt.subplot(1, 2, 2)
    show_img(style_img, 'Style Image')
    print('pausing pyplot')
    plt.pause(10)


if __name__ == '__main__':
    main()

