from __future__ import division

import numpy as np
import os
from keras import backend as K
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def current_fold(results_dir, prefix):
    results_csv = list()
    filenames = sorted(next(os.walk(results_dir))[2])
    for fname in filenames:
        if not fname.startswith(prefix):
            continue
        if not fname.endswith('.csv'):
            continue
        results_csv.append(fname)

    if results_csv:
        return int(results_csv[-1].split('.')[-3].split('_')[1]) + 1
    return 1


def load_image(image_data_generator, img_path, target_size=(224, 224)):
    image_shape = target_size + (3,)
    batch_x = np.zeros((1,) + image_shape, dtype=K.floatx())
    img = load_img(img_path, target_size=target_size, grayscale=False)
    x = img_to_array(img)
    x = image_data_generator.random_transform(x)
    x = image_data_generator.standardize(x)
    batch_x[0] = x
    return batch_x


def read_fold_dir(fold_dir):
    images = list()
    for dirpath, _, filenames in os.walk(fold_dir):
        if dirpath == fold_dir:
            continue
        label = int(os.path.split(dirpath)[-1])
        for filename in filenames:
            img_path = os.path.join(dirpath, filename)
            images.append((label, img_path))
    return images
