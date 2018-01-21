from __future__ import division

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from easydict import EasyDict as edict
from training import train_random_forest

import os
import experiments as exp
import utils.gpu
import argparse
import pickle


def current_fold(weights_dir, prefix):
    rf_weights_files = list()
    filenames = sorted(next(os.walk(weights_dir))[2])
    for fname in filenames:
        if not fname.endswith('.pkl'):
            continue
        if not fname.startswith(prefix+'.fold_'):
            continue
        rf_weights_files.append(fname)
    if rf_weights_files:
        return int(rf_weights_files[-1].split('.')[-2].split('_')[1]) + 1
    return 1


def train_on_features(data_dir, weights_dir, features_filepath, rf_model, start_fold=None, end_fold=10):
    np.random.seed(42)

    if not start_fold:
        start_fold = current_fold(weights_dir, "weights." + rf_model.name)

    folds = [str(fold).zfill(2) for fold in range(start_fold, end_fold + 1)]
    for fold in folds:

        print "Training fold {} on layers {}".format(fold, ", ".join(rf_model.layers))
        train_dir = os.path.join(data_dir, fold, 'train')

        with open(features_filepath.format(fold), 'r') as f:
            features = pickle.load(f)

        weights_filepath = os.path.join(weights_dir, "weights." + rf_model.name + ".fold_" + fold + ".pkl")
        train_random_forest(rf_model.num_estimators, weights_filepath, users=features, fold_dir=train_dir)


def train_on_cnn(data_dir, weights_dir, cnn_model, rf_model, start_fold=None, end_fold=10):
    np.random.seed(42)

    if not start_fold:
        start_fold = current_fold(weights_dir, "weights." + rf_model.name)

    folds = [str(fold).zfill(2) for fold in range(start_fold, end_fold + 1)]
    for fold in folds:

        # prepare data augmentation configuration
        train_datagen = ImageDataGenerator(rescale=1. / 255)

        init_weights = cnn_model.weights.format(fold)
        base_model = cnn_model.load(weights=init_weights)

        layers_by_name = {l.name: l for l in base_model.layers}
        outputs = [layers_by_name[l].output for l in rf_model.layers]
        model = Model(inputs=base_model.input, outputs=outputs)

        print "Training fold {} on layers {}".format(fold, ", ".join(rf_model.layers))
        train_dir = os.path.join(data_dir, fold, 'train')
        train_generator = train_datagen.flow_from_directory(train_dir,
                                                            target_size=(cnn_model.img_height, cnn_model.img_width),
                                                            class_mode='categorical',
                                                            batch_size=1)

        weights_filepath = os.path.join(weights_dir, "weights." + rf_model.name + ".fold_" + fold + ".pkl")
        train_random_forest(rf_model.num_estimators, weights_filepath, model, train_generator)

        del model
        if K.backend() == 'tensorflow':
            K.clear_session()


def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN base networks')
    parser.add_argument('--gpu_fraction', dest='gpu_fraction',
                        help='GPU fraction usage',
                        default=0.8, type=float)
    parser.add_argument('--data_dir', dest='data_dir',
                        help='Data directory where the folds are located',
                        default='data', type=str)
    parser.add_argument('--weights_dir', dest='weights_dir',
                        help='Directory where the weights are stored',
                        default='weights', type=str)
    parser.add_argument('--network', dest='network',
                        help='CNN to be trained',
                        default='vgg-16', type=str)
    parser.add_argument('--start_fold', dest='start_fold',
                        help='Start fold number to train',
                        default=None, type=int)
    parser.add_argument('--end_fold', dest='end_fold',
                        help='End fold number to train',
                        default=10, type=int)
    parser.add_argument('--num_estimators', dest='num_estimators',
                        help='Number of estimators for Random Forest',
                        default=500, type=int)
    parser.add_argument('--features_filepath', dest='features_filepath',
                        help='Features filepath string',
                        default=None, type=str)
    parser.add_argument('-l', '--layer', dest='layers',
                        help='Layers to be extracted for the Random Forest training',
                        required=True, nargs='+', type=str)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if K.backend() == 'tensorflow':
        import keras.backend.tensorflow_backend as KTF
        KTF.set_session(utils.gpu.get_session(args.gpu_fraction))

    weights_dir = os.path.realpath(args.weights_dir)
    weights_dir = os.path.join(weights_dir, args.network)

    if args.network == 'vgg-16':
        cnn_model = edict({'weights': weights_dir + '/weights.vgg-16.phase_2.fold_{}.best.tf.hdf5',
                       'name': 'vgg-16',
                       'img_width': 224,
                       'img_height': 224,
                       'load': exp.vgg16_second_phase_model})
    elif args.network == 'inceptionV3':
        cnn_model = edict({'weights': weights_dir + '/weights.inceptionV3.phase_2.fold_{}.best.tf.hdf5',
                       'name': 'inceptionV3',
                       'img_width': 299,
                       'img_height': 299,
                       'load': exp.inceptionV3_second_phase_model})
    elif args.network == 'resNet50':
        cnn_model = edict({'weights': weights_dir + '/weights.resNet50.phase_2.fold_{}.best.tf.hdf5',
                       'name': 'resNet50',
                       'img_width': 224,
                       'img_height': 224,
                       'load': exp.resNet50_second_phase})

    rf_model = edict({'num_estimators': args.num_estimators,
                      'name': cnn_model.name + '.RF.layers_' + '_'.join(args.layers),
                      'layers': args.layers})

    utils.makedirs(args.weights_dir)

    if args.features_filepath:
        train_on_features(args.data_dir, weights_dir, args.features_filepath, rf_model, args.start_fold, args.end_fold)
    else:
        train_on_cnn(args.data_dir, weights_dir, cnn_model, rf_model, args.start_fold, args.end_fold)
