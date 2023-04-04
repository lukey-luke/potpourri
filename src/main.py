import tensorflow as tf
import numpy as np
import matplotlib as mpl
import PIL.Image


def get_image(img_path):
    # image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(img_path)
    return np.array(img_path)

def get_models():
    applications = dir(tf.keras.applications)
    print(type(applications))
    for application in applications:
        # print(f'{application.__name__}')
        print(application)
        print(type(application))
        print(isinstance(application, type))



def main():
    # glorious_image_path = '../twister_toque/PXL_20211013_055448184.jpg' 
    # img = PIL.Image.open(glorious_image_path)
    # img.show()
    get_models()


if __name__ == '__main__':
    main()

