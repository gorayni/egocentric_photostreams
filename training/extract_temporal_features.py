from __future__ import division

import ntcir
import ntcir.IO as IO

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from easydict import EasyDict as edict

import experiments as exp
import utils.gpu
import argparse
import pickle
from testing import load_image
from testing.test_rf import load_random_forest
from testing import read_fold_dir
import numpy as np
from skimage import io
from skimage import img_as_ubyte


def extract_cnn_features(cnn_model, layer, features_dir):
    # prepare data augmentation configuration
    datagen = ImageDataGenerator(rescale=1. / 255)

    base_model = cnn_model.load(weights=cnn_model.weights)

    target_size = (cnn_model.img_height, cnn_model.img_width)

    layers_by_name = {l.name: l for l in base_model.layers}
    outputs = layers_by_name[layer].output
    model = Model(inputs=base_model.input, outputs=outputs)

    users = IO.load_annotations(ntcir.filepaths)

    for user_id, user in users.iteritems():
        for date, day in user.iteritems():
            for image in day.images:
                img = load_image(datagen, image.path, target_size)
                image.features = model.predict(img).copy()

    features_filepath = os.path.join(features_dir, "features." + cnn_model.name + ".pkl")
    with open(features_filepath, 'w') as f:
        pickle.dump(users, f, pickle.HIGHEST_PROTOCOL)

    del model
    if K.backend() == 'tensorflow':
        K.clear_session()


def get_histogram(img_path, num_bins):
    img = img_as_ubyte(io.imread(img_path))

    histogram = np.zeros((3, num_bins))
    for channel in range(3):
        histogram[channel,:], _ = np.histogram(img[:,:,channel],num_bins,[0,256])
    histogram = histogram.reshape(3*num_bins)

    return histogram


def extract_castro_features(cnn_model, data_dir, features_dir, start_fold = 1, end_fold = 5, num_categories=21, num_bins=10, progress_percent=.05):

    backend = 'tf' if K.backend() == 'tensorflow' else 'th'
    target_size = (cnn_model.img_height, cnn_model.img_width)
    datagen = ImageDataGenerator(rescale=1. / 255)

    folds = [str(fold).zfill(2) for fold in range(start_fold, end_fold + 1)]
    for fold in folds:
        weights = cnn_model.weights.format(fold)
        model = cnn_model.load(weights=weights)

        users = IO.load_annotations(ntcir.filepaths)

        ind_by_img_path = dict()
        for user_id, days in users.iteritems():
            for date, day in days.iteritems():
                for ind, image in enumerate(day.images):
                    relative_path =  '/'.join(image.path.split('/')[-3:])
                    ind_by_img_path[relative_path] = ind

        test_dir = os.path.join(data_dir, fold, 'test')
        train_dir = os.path.join(data_dir, fold, 'train')
        validation_dir = os.path.join(data_dir, fold, 'validation')

        if os.path.isdir(validation_dir):
            images = read_fold_dir(train_dir) + read_fold_dir(test_dir) + read_fold_dir(validation_dir)
        else:
            images = read_fold_dir(train_dir) + read_fold_dir(test_dir)

        num_images = len(images)
        images_progress_percent = int(num_images * progress_percent)

        print 'Extracting temporal features on fold {} for {}'.format(fold, cnn_model.name)

        for i, (label, img_path) in enumerate(images):

            img = load_image(datagen, img_path, target_size)

            features = np.zeros((num_categories+3*num_bins+3))
            features[:num_categories] = model.predict(img)
            features[num_categories] = image.hour
            features[num_categories+1] = image.minute
            features[num_categories+2] = image.weekday
            features[num_categories+3:] = get_histogram(image.path, num_bins)

            rpath = os.path.realpath(img_path)
            user_id, date, filename = rpath.split('/')[-3:]

            relative_path = '/'.join([user_id, date, filename])

            img_ind = ind_by_img_path[relative_path]
            image = users[user_id][date].images[img_ind]
            image.features = features

            if progress_percent and (i + 1) % images_progress_percent == 0:
                print("Progress %3.2f%% (%d/%d)" % ((i + 1) / num_images * 100, i + 1, num_images))

        features_filepath = "features.{}.fold_{}.{}.pkl".format(rf_model.name, fold, backend)
        features_filepath = os.path.join(features_dir, features_filepath)
        with open(features_filepath, 'w') as f:
            pickle.dump(users, f, pickle.HIGHEST_PROTOCOL)

        del model
        if K.backend() == 'tensorflow':
            K.clear_session()


def extract_rf_features(data_dir, features_dir, cnn_model, rf_model, start_fold = 1, end_fold = 5, progress_percent=.1):

    backend = 'tf' if K.backend() == 'tensorflow' else 'th'
    target_size = (cnn_model.img_height, cnn_model.img_width)
    datagen = ImageDataGenerator(rescale=1. / 255)

    folds = [str(fold).zfill(2) for fold in range(start_fold, end_fold + 1)]
    for fold in folds:
        weights = cnn_model.weights.format(fold)
        base_model = cnn_model.load(weights=weights)

        layers_by_name = {l.name: l for l in base_model.layers}
        outputs = [layers_by_name[rf_model.layer].output]
        model = Model(inputs=base_model.input, outputs=outputs)

        weights = rf_model.weights.format(fold)
        rf = load_random_forest(weights)

        users = IO.load_annotations(ntcir.filepaths)

        ind_by_img_path = dict()
        for user_id, days in users.iteritems():
            for date, day in days.iteritems():
                for ind, image in enumerate(day.images):
                    relative_path =  '/'.join(image.path.split('/')[-3:])
                    ind_by_img_path[relative_path] = ind

        test_dir = os.path.join(data_dir, fold, 'test')
        train_dir = os.path.join(data_dir, fold, 'train')
        validation_dir = os.path.join(data_dir, fold, 'validation')

        if os.path.isdir(validation_dir):
            images = read_fold_dir(train_dir) + read_fold_dir(test_dir) + read_fold_dir(validation_dir)
        else:
            images = read_fold_dir(train_dir) + read_fold_dir(test_dir)

        num_images = len(images)
        images_progress_percent = int(num_images * progress_percent)

        print 'Extracting temporal features on fold {} for {} + RF on layer {}'.format(fold, cnn_model.name, rf_model.layer)

        for i, (label, img_path) in enumerate(images):

            img = load_image(datagen, img_path, target_size)

            predictions = model.predict(img)

            # Concatenating features
            features = predictions[0].copy()
            probability = rf.predict_proba([features])[0]

            rpath = os.path.realpath(img_path)
            user_id, date, filename = rpath.split('/')[-3:]

            relative_path = '/'.join([user_id, date, filename])

            img_ind = ind_by_img_path[relative_path]
            image = users[user_id][date].images[img_ind]
            image.features = probability.copy()

            if progress_percent and (i + 1) % images_progress_percent == 0:
                print("Progress %3.2f%% (%d/%d)" % ((i + 1) / num_images * 100, i + 1, num_images))

        features_filepath = "features.{}.fold_{}.{}.pkl".format(rf_model.name, fold, backend)
        features_filepath = os.path.join(features_dir, features_filepath)
        with open(features_filepath, 'w') as f:
            pickle.dump(users, f, pickle.HIGHEST_PROTOCOL)

        del model
        if K.backend() == 'tensorflow':
            K.clear_session()


def parse_args():
    parser = argparse.ArgumentParser(description='Train temporal RF based on CNN networks')
    parser.add_argument('--gpu_fraction', dest='gpu_fraction',
                        help='GPU fraction usage',
                        default=0.8, type=float)
    parser.add_argument('--features_dir', dest='features_dir',
                        help='Directory where the pre-computed features are stored',
                        default='features', type=str)
    parser.add_argument('--network', dest='network',
                        help='CNN to be trained',
                        default='resNet50', type=str)
    parser.add_argument('-l', '--layer', dest='layer',
                        help='Layer for feature extraction',
                        required=False, type=str)
    parser.add_argument("--features_type", dest='features_type',
                        help="Features type",
                        default="cnn", type=str)
    parser.add_argument('--num_estimators', dest='num_estimators',
                        help='Number of estimators for Random Forest',
                        default=500, type=int)
    parser.add_argument('--data_dir', dest='data_dir',
                        help='Data directory where the folds are located',
                        default='data', type=str)
    parser.add_argument('--weights_dir', dest='weights_dir',
                        help='Directory where the weights are stored',
                        default='weights', type=str)
    parser.add_argument('--weights_filename', dest='weights_filename',
                        help='CNN weights',
                        default=None, type=str)
    parser.add_argument('--start_fold', dest='start_fold',
                        help='Start fold number to train',
                        default=1, type=int)
    parser.add_argument('--end_fold', dest='end_fold',
                        help='End fold number to train',
                        default=5, type=int)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if K.backend() == 'tensorflow':
        import keras.backend.tensorflow_backend as KTF

        KTF.set_session(utils.gpu.get_session(args.gpu_fraction))
        backend = 'tf'
    else:
        backend = 'th'

    weights_dir = os.path.join(args.weights_dir, args.network)

    if args.network == 'vgg-16':
        if args.weights_filename:
            weights = os.path.join(weights_dir, args.weights_filename)
        else:
            weights = weights_dir + '/weights.vgg-16.phase_2.fold_{}.best.tf.hdf5'

        cnn_model = edict({'weights': weights,
                       'name': 'vgg-16',
                       'img_width': 224,
                       'img_height': 224,
                       'load': exp.vgg16_second_phase_model})

    elif args.network == 'resNet50':
        if args.weights_filename:
            weights = os.path.join(weights_dir, args.weights_filename)
        else:
            weights = weights_dir + '/weights.resNet50.best.tf.hdf5'

        cnn_model = edict(
            {'weights': weights,
             'name': 'resNet50',
             'img_width': 224,
             'img_height': 224,
             'load': exp.resNet50_second_phase})

    elif args.network == 'inceptionV3':
        if args.weights_filename:
            weights = os.path.join(weights_dir, args.weights_filename)
        else:
            weights = weights_dir + '/weights.inceptionV3.best.tf.hdf5'

        cnn_model = edict(
            {'weights': weights,
             'name': 'inceptionV3',
             'img_width': 299,
             'img_height': 299,
             'load': exp.inceptionV3_second_phase_model})

    features_dir = os.path.join(args.features_dir, args.network)
    utils.makedirs(features_dir)

    if args.features_type == 'cnn':
        extract_cnn_features(cnn_model, args.layer, features_dir)
    elif args.features_type == 'rf':
        rf_model = edict({'num_estimators': args.num_estimators,
                          'weights': weights_dir + "/weights." + cnn_model.name + '.RF.layers_' + args.layer + '.fold_{}.pkl',
                          'name': cnn_model.name + '.RF.layers_' + args.layer,
                          'layer': args.layer})
        extract_rf_features(args.data_dir, features_dir, cnn_model, rf_model, args.start_fold, args.end_fold)
    else:
        rf_model = edict({'num_estimators': args.num_estimators,
                          'weights': weights_dir + "/weights.castro." + cnn_model.name + '.fold_{}.pkl',
                          'name': 'castro.' + cnn_model.name})
        extract_castro_features(cnn_model, args.data_dir, features_dir, args.start_fold, args.end_fold, num_categories=21, num_bins=10)
