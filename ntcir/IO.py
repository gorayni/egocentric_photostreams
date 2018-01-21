from __future__ import division
import re
import os
import numpy as np
import utils
from . import Image
from . import Day
from . import User

from collections import defaultdict


class Filepaths(object):
    def __init__(self, ntcir_dir):
        ntcir_dir = os.path.realpath(ntcir_dir)
        self.annotations = os.path.join(ntcir_dir, 'annotations.txt')
        self.categories = os.path.join(ntcir_dir, 'categories.txt')
        self.images_dir = os.path.join(ntcir_dir, 'images')


def load_categories(filepaths):
    return list(np.loadtxt(filepaths.categories, str, delimiter='\n'))


def load_annotations(filepaths):
    images = defaultdict(lambda: defaultdict(list))
    lines = np.loadtxt(filepaths.annotations, str, delimiter='\n')
    for i, line in enumerate(lines):
        path, label = line.rsplit(' ', 1)
        id_, date, time = path.split(os.path.sep)
        label = int(label)
        path = os.path.join(filepaths.images_dir, path)
        time = re.sub("[^0-9]", "", time.split('_')[-1])

        image = Image(path, date, time, label)
        images[id_][date].append(image)

    users = defaultdict(lambda: defaultdict(list))
    for user_id, days in images.items():
        user = User(user_id, list())
        for date, images in days.items():
            images.sort(key=lambda img: img.time)
            day = Day(date, images, user)
            user.days.append(day)
            users[user_id][date] = day
        user.days.sort(key=lambda day: day.date)
    return utils.default_to_regular(users)
