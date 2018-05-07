from __future__ import division

import os
from easydict import EasyDict as edict

import argparse
import pickle
import numpy as np
import utils

def convert_cnn_features(cnn_model, layers, features_dir, start_fold=1, end_fold=10):
    folds = [str(fold).zfill(2) for fold in range(start_fold, end_fold + 1)]
    for fold in folds:
        features_filename = cnn_model.features_filename.format(fold)
        features_filepath = os.path.join(features_dir, features_filename)

        with open(features_filepath.format(fold), 'r') as f:
            users = pickle.load(f)

        num_features = 0
        for user_id, days in users.iteritems():
            for date, day in days.iteritems():
                for image in day.images:
                    if not hasattr(image, 'features'):
                        continue

                    if num_features == 0:
                        for l in layers:
                            num_features += image.features[l].size

                    start_ind = 0
                    features = np.zeros((1, num_features))
                    for l in layers:
                        end_ind = start_ind + image.features[l].size
                        features[0, start_ind:end_ind] = image.features[l]
                        start_ind = end_ind
                    image.features = features

        output_filename = "features.{}.layers_{}.fold_{}.pkl".format(cnn_model.name, "_".join(layers), fold)
        output_filepath = os.path.join(features_dir, output_filename)

        with open(output_filepath, 'w') as f:
            pickle.dump(users, f, pickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser(description='Train temporal RF based on CNN networks')
    parser.add_argument('--features_dir', dest='features_dir',
                        help='Directory where the pre-computed features are stored',
                        default='features', type=str)
    parser.add_argument('--network', dest='network',
                        help='CNN to be trained',
                        default='resNet50', type=str)
    parser.add_argument('-l', '--layer', dest='layers',
                        help='Layers to be extracted',
                        required=True, nargs='+', type=str)
    parser.add_argument('--start_fold', dest='start_fold',
                        help='Start fold number to train',
                        default=1, type=int)
    parser.add_argument('--end_fold', dest='end_fold',
                        help='End fold number to train',
                        default=10, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.network == 'vgg-16':
        cnn_model = edict({'name': 'vgg-16',
                           'features_filename': "features.vgg-16.fold_{}.pkl"})

    elif args.network == 'resNet50':
        cnn_model = edict(
            {'name': 'resNet50',
             'features_filename': "features.resNet50.fold_{}.pkl"})

    elif args.network == 'inceptionV3':
        cnn_model = edict(
            {'name': 'inceptionV3',
             'features_filename': "features.inceptionV3.fold_{}.pkl"})

    features_dir = os.path.join(args.features_dir, args.network)
    utils.makedirs(features_dir)

    convert_cnn_features(cnn_model, args.layers, features_dir, args.start_fold, args.end_fold)
