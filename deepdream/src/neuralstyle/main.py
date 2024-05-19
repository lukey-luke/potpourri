# import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.figsize"] = (12, 12)
mpl.rcParams["axes.grid"] = False
# import IPython.display as display

import numpy as np
import PIL.Image
import time
import functools
from model import StyleContentModel


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
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

    print("calling plt.imshow()... ")
    plt.imshow(image)
    if title:
        plt.title(title)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def main():
    def style_content_loss(outputs):
        style_outputs = outputs["style"]
        content_outputs = outputs["content"]
        style_loss = tf.add_n(
            [
                tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                for name in style_outputs.keys()
            ]
        )
        # style_loss *= style_weight / num_style_layers
        style_loss *= style_weight / len(style_layers)

        content_loss = tf.add_n(
            [
                tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                for name in content_outputs.keys()
            ]
        )
        # content_loss *= content_weight / num_content_layers
        content_loss *= content_weight / len(content_layers)
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    content_path = "../../twister_toque/toque.png"
    style_path = "../../twister_toque/rapids.jpeg"
    print("loading imgs")
    style_img = load_img(style_path)
    content_img = load_img(content_path)
    # print('plotting content')
    # plt.subplot(1, 2, 1)
    # show_img(content_img, 'Content Image')
    # print('plotting style')
    # plt.subplot(1, 2, 2)
    # show_img(style_img, 'Style Image')
    # print('pausing pyplot')
    # plt.pause(10)
    x = tf.keras.applications.vgg19.preprocess_input(content_img * 255)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=True, weights="imagenet")
    prediction_probabilities = vgg(x)
    prediction_probabilities.shape
    predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(
        prediction_probabilities.numpy()
    )[0]
    top_5 = [(class_name, prob) for (number, class_name, prob) in predicted_top_5]
    print(f"top_5: {top_5}")
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")

    # print()
    # for layer in vgg.layers:
    #     print(layer.name)
    content_layers = ["block5_conv2"]
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]

    extractor = StyleContentModel(style_layers, content_layers)
    # results = extractor(tf.constant(content_image))

    style_targets = extractor(style_img)["style"]
    content_targets = extractor(content_img)["content"]

    image = tf.Variable(content_img)
    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    style_weight = 1e-2
    content_weight = 1e4

    for num in range(69):
        train_step(image)
    # train_step(image)
    # train_step(image)
    tensor_img = tensor_to_image(image)
    print(f"tensor_img: {tensor_img}")
    tensor_img.show()
    # PIL.Image.open(tensor_img)


if __name__ == "__main__":
    main()
