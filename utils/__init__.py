from __future__ import division

import errno
import os

import numpy as np


def makedirs(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(dir_path):
            pass
        else:
            raise


def num_digits(number):
    return int(np.floor(np.log10(np.abs(number))) + 1)


def ext(path):
    return os.path.splitext(path)[-1]


def link_images(num_categories, split_dir, padding_zeros, targets, img_paths, indices=None):
    counter_inst = np.ones(num_categories, np.int)
    if indices is None:
        indices = xrange(len(targets))

    for j in indices:
        category_ind = targets[j]

        cat_dir = str(category_ind).zfill(num_digits(num_categories))
        img_dir = os.path.join(split_dir, cat_dir)

        num_img = counter_inst[category_ind]
        dst_basename = str(num_img).zfill(padding_zeros)
        dst_basename += ext(img_paths[j])
        dst_filepath = os.path.join(img_dir, dst_basename)

        os.symlink(img_paths[j], dst_filepath)
        counter_inst[category_ind] += 1
