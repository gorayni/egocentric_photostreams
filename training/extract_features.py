from __future__ import division

import ntcir
import ntcir.IO as IO

import os
import numpy as np
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
from training import images_from_fold_dir
from training import get_images_indices

def extract_cnn_features(cnn_model, layers, features_dir, start_fold=1, end_fold=10):
    folds = [str(fold).zfill(2) for fold in range(start_fold, end_fold + 1)]
    for fold in folds:

        # prepare data augmentation configuration
        datagen = ImageDataGenerator(rescale=1. / 255)

        init_weights = cnn_model.weights.format(fold)
        base_model = cnn_model.load(weights=init_weights)

        target_size = (cnn_model.img_height, cnn_model.img_width)

        layers_by_name = {l.name: l for l in base_model.layers}
        outputs = [layers_by_name[l].output for l in layers]
        model = Model(inputs=base_model.input, outputs=outputs)

        users = IO.load_annotations(ntcir.filepaths)
        for user_id, user in users.iteritems():
            for date, day in user.iteritems():
                for image in day.images:
                    img = load_image(datagen, image.path, target_size)

                    predictions = model.predict(img)
                    if len(model.output_layers) == 1:
                        predictions = [predictions]

                    image.features = {l: predictions[i].copy() for i, l in enumerate(layers)}

        features_filepath = os.path.join(features_dir, "features." + cnn_model.name + ".fold_" + fold + ".pkl")
        with open(features_filepath, 'w') as f:
            pickle.dump(users, f, pickle.HIGHEST_PROTOCOL)

        del model
        if K.backend() == 'tensorflow':
            K.clear_session()


def num_images_from(users):
    num_images = 0
    for user_id, days in users.iteritems():
        for date, day in days.iteritems():
            num_images += day.user.num_images
            break
    return num_images


def extract_rf_on_features(rf_model, features_filepath, features_dir, start_fold = 1, end_fold = 10, progress_percent=.1):

    backend = 'tf' if K.backend() == 'tensorflow' else 'th'
    if not start_fold:
        start_fold = current_fold(results_dir, rf_model.name + '.fold')

    folds = [str(fold).zfill(2) for fold in range(start_fold, end_fold + 1)]
    for fold in folds:

        print "Extracting fold {} on layers {}".format(fold, ", ".join(rf_model.layers))
        weights = rf_model.weights.format(fold)
        rf = load_random_forest(weights)

        with open(features_filepath.format(fold), 'r') as f:
            users = pickle.load(f)

        images = list()
        for user_id, days in users.iteritems():
            for date, day in days.iteritems():
                for image in day.images:
                    if not hasattr(image, 'features'):
                        continue
                    images.append(image)

        num_images = len(images)

        # Determining the number of features
        num_features = images[0].features.size

        features = np.zeros((num_images, num_features))
        for i, image in enumerate(images):
            features[i,:] = image.features
        probabilities = rf.predict_proba(features)

        for i, image in enumerate(images):
            image.features = probabilities[i,:]

        dest_features_filepath = "features.{}.fold_{}.{}.pkl".format(rf_model.name, fold, backend)
        dest_features_filepath = os.path.join(features_dir, dest_features_filepath)
        with open(dest_features_filepath, 'w') as f:
            pickle.dump(users, f, pickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser(description='Train temporal RF based on CNN networks')
    parser.add_argument('--gpu_fraction', dest='gpu_fraction',
                        help='GPU fraction usage',
                        default=0.8, type=float)
    parser.add_argument('--features_dir', dest='features_dir',
                        help='Directory where the pre-computed features are stored',
                        default='features', type=str)
    parser.add_argument('--features_filepath', dest='features_filepath',
                        help='Features filepath template',
                        default=None, type=str)
    parser.add_argument('--network', dest='network',
                        help='CNN to be trained',
                        default='resNet50', type=str)
    parser.add_argument('-l', '--layer', dest='layers',
                        help='Layers to be extracted',
                        required=True, nargs='+', type=str)
    parser.add_argument("--features_type", dest='features_type',
                        help="Features type",
                        default="cnn", type=str)
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
                        default=10, type=int)
    parser.add_argument('--num_estimators', dest='num_estimators',
                        help='Number of estimators for Random Forest',
                        default=500, type=int)
    parser.add_argument('--progress_percent', dest='progress_percent',
                        help='Progress percent to display',
                        default=1, type=float)
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
                           'features_filename': "features.vgg-16.fold_{}.pkl",
                           'load': exp.vgg16_second_phase_model})

    elif args.network == 'resNet50':
        if args.weights_filename:
            weights = os.path.join(weights_dir, args.weights_filename)
        else:
            weights = weights_dir + '/weights.resNet50.phase_2.fold_{}.best.tf.hdf5'

        cnn_model = edict(
            {'weights': weights,
             'name': 'resNet50',
             'img_width': 224,
             'img_height': 224,
             'features_filename': "features.resNet50.fold_{}.pkl",
             'load': exp.resNet50_second_phase})

    elif args.network == 'inceptionV3':
        if args.weights_filename:
            weights = os.path.join(weights_dir, args.weights_filename)
        else:
            weights = weights_dir + '/weights.inceptionV3.phase_2.fold_{}.best.tf.hdf5'

        cnn_model = edict(
            {'weights': weights,
             'name': 'inceptionV3',
             'img_width': 299,
             'img_height': 299,
             'features_filename': "features.inceptionV3.fold_{}.pkl",
             'load': exp.inceptionV3_second_phase_model})

    features_dir = os.path.join(args.features_dir, args.network)
    utils.makedirs(features_dir)

    if args.features_type == 'cnn':
        extract_cnn_features(cnn_model, args.layers, features_dir, args.start_fold, args.end_fold)
    elif args.features_type == 'rf':
        rf_model = edict({'num_estimators': args.num_estimators,
                          'weights': weights_dir + "/weights." + cnn_model.name + '.RF.layers_' + '_'.join(args.layers) + ".num_estimators_" + str(args.num_estimators) + ".fold_{}.pkl",
                          'name': cnn_model.name + '.RF.layers_' + '_'.join(args.layers),
                          'layers': args.layers})
        if not args.features_filepath:
            args.features_filepath = features_dir + "/features.{}.layers_{}".format(cnn_model.name, "_".join(args.layers)) + ".fold_{}.pkl"
        extract_rf_on_features(rf_model, args.features_filepath, features_dir, args.start_fold, args.end_fold, args.progress_percent)
