from __future__ import division
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight

import os
import pickle
import numpy as np
import multiprocessing


def compute_class_weights(train_dir, ext='.jpg'):
    categories = sorted(next(os.walk(train_dir))[1])

    targets = list()
    for i, category in enumerate(categories):
        cat_dir = os.path.join(train_dir, category)
        targets.extend([i for name in os.listdir(cat_dir) if name.endswith(ext)])

    return class_weight.compute_class_weight('balanced', np.arange(len(categories)), targets)


def features_from_cnn(model, generator, progress_percent=0.05):
    num_features = np.sum([l.output_shape[-1] for l in model.output_layers])
    num_images = generator.n

    # Extract features of the images
    features = np.zeros((num_images, num_features))
    targets = np.zeros(num_images)

    if progress_percent:
        training_progress_percent = int(num_images * progress_percent)

    for i, (img, target) in enumerate(generator):
        if i == generator.n:
            break
        predictions = model.predict(img)
        if len(model.output_layers) == 1:
            predictions = [predictions]

        # Concatenating features
        feature = np.array([])
        for prediction in predictions:
            feature = np.append(feature, prediction[0].copy())
        features[i] = feature
        targets[i] = np.argmax(target)

        if progress_percent and (i + 1) % training_progress_percent == 0:
            print("Progress %3.2f%% (%d/%d)" % ((i + 1) / num_images * 100, i + 1, num_images))

    return features, targets


def images_from_fold_dir(fold_dir):
    for dirpath, _, filenames in os.walk(fold_dir):
        if dirpath == fold_dir:
            continue

        for filename in filenames:
            img_path = os.path.join(dirpath, filename)
            real_img_path = os.path.realpath(img_path)

            user_id, date, filename = real_img_path.split(os.path.sep)[-3:]
            relative_path = '/'.join([user_id,date,filename])
            yield user_id, date, relative_path


def get_images_indices(users):
    ind_by_img_path = dict()
    for user_id, days in users.iteritems():
        for date, day in days.iteritems():
            for ind, image in enumerate(day.images):
                relative_path =  '/'.join(image.path.split('/')[-3:])
                ind_by_img_path[relative_path] = ind
    return ind_by_img_path


def features_from_users(users, fold_dir):
    ind_by_img_path = get_images_indices(users)
    features, targets = list(), list()
    for i, (user_id, date, relative_path) in enumerate(images_from_fold_dir(fold_dir)):
        ind = ind_by_img_path[relative_path]
        features.append(users[user_id][date].images[ind].features)
        targets.append(users[user_id][date].images[ind].label)
    return np.asarray(features), np.asarray(targets)


def train_random_forest(n_estimators, max_depth, dst_filepath=None, model=None, generator=None, users=None, fold_dir=None, cores=None, progress_percent=0.05):

    if model and generator:
        features, targets = features_from_cnn(model, generator, progress_percent)
    elif users and fold_dir:
        features, targets = features_from_users(users, fold_dir)

    if not cores:
        cores = multiprocessing.cpu_count()
    random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=cores)
    features = np.squeeze(features)
    random_forest.fit(features, targets)

    if dst_filepath:
        with open(dst_filepath, 'w') as f:
            pickle.dump(random_forest, f, pickle.HIGHEST_PROTOCOL)
    return random_forest
