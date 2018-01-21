from __future__ import division

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from easydict import EasyDict as edict
from keras.models import Model

import os
import numpy as np
import experiments as exp
import utils.gpu
import pickle
import argparse
from testing import load_image
from testing import read_fold_dir
from testing import current_fold


def write_results(results, csv_filepath):
    with open(csv_filepath, 'w') as csv_file:
        for img_path, label, prediction in results:
            csv_file.write("{},{},{}".format(img_path, label, prediction))
            csv_file.write('\n')


def load_random_forest(random_forest_filepath):
    if os.path.isfile(random_forest_filepath):
        with open(random_forest_filepath, 'r') as f:
            return pickle.load(f)
    return None


def test(data_dir, results_dir, cnn_model, rf_model, start_fold=None, end_fold=10, progress_percent=1):
    if not start_fold:
        start_fold = current_fold(results_dir, rf_model.name + '.fold')

    backend = 'tf' if K.backend() == 'tensorflow' else 'th'
    target_size = (cnn_model.img_height, cnn_model.img_width)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    folds = [str(fold).zfill(2) for fold in range(start_fold, end_fold + 1)]
    for fold in folds:
        weights = cnn_model.best_weights.format(fold)
        base_model = cnn_model.load(weights=weights)

        layers_by_name = {l.name: l for l in base_model.layers}
        outputs = [layers_by_name[l].output for l in rf_model.layers]
        model = Model(inputs=base_model.input, outputs=outputs)

        weights = rf_model.weights.format(fold)
        rf = load_random_forest(weights)

        results = list()
        test_dir = os.path.join(data_dir, fold, 'test')
        test_images = read_fold_dir(test_dir)

        num_test_images = len(test_images)
        test_progress_percent = int(num_test_images * progress_percent)

        print 'Testing fold {} for {} + RF on layers {}'.format(fold, cnn_model.name, ', '.join(rf_model.layers))
        for i, (label, img_path) in enumerate(test_images):

            img = load_image(test_datagen, img_path, target_size)

            predictions = model.predict(img)
            if len(rf_model.layers) == 1:
                predictions = [predictions]

            # Concatenating features
            features = np.array([])
            for p in predictions:
                features = np.append(features, p[0].copy())
            prediction = rf.predict([features])[0].astype(np.int)

            results.append((img_path, label, prediction))

            if progress_percent and (i + 1) % test_progress_percent == 0:
                print("Progress %3.2f%% (%d/%d)" % ((i + 1) / num_test_images * 100, i + 1, num_test_images))

        results_fname = "{}.fold_{}.{}.csv".format(rf_model.name, fold, backend)
        results_filepath = os.path.join(results_dir, results_fname)
        write_results(results, results_filepath)

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
    parser.add_argument('--results_dir', dest='results_dir',
                        help='Directory where the CSV results will be stored',
                        default='results', type=str)
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
    parser.add_argument('-l', '--layer', dest='layers',
                        help='Layers to be extracted for the Random Forest training',
                        required=True, nargs='+', type=str)
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

    weights_dir = os.path.join(os.path.realpath(args.weights_dir), args.network)
    results_dir = os.path.join(args.results_dir, args.network)

    if args.network == 'vgg-16':
        cnn_model = edict({'best_weights': weights_dir + '/weights.vgg-16.phase_2.fold_{}.best.' + backend + '.hdf5',
                           'name': 'vgg-16',
                           'img_width': 224,
                           'img_height': 224,
                           'load': exp.vgg16_second_phase_model})
    elif args.network == 'resNet50':
        cnn_model = edict(
            {'best_weights': weights_dir + '/weights.resNet50.phase_2.fold_{}.best.' + backend + '.hdf5',
             'name': 'resNet50',
             'img_width': 224,
             'img_height': 224,
             'load': exp.resNet50_second_phase})

    elif args.network == 'inceptionV3':
        cnn_model = edict({'best_weights': weights_dir + '/weights.inceptionV3.phase_2.fold_{}.best.' + backend + '.hdf5',
             'name': 'inceptionV3',
             'img_width': 299,
             'img_height': 299,
             'load': exp.inceptionV3_second_phase_model})

    rf_model = edict({'name': cnn_model.name + '.RF.layers_' + '_'.join(args.layers),
                      'weights': weights_dir + "/weights." + cnn_model.name + '.RF.layers_' + '_'.join(
                          args.layers) + ".fold_{}.pkl",
                      'layers': args.layers})

    utils.makedirs(results_dir)
    test(args.data_dir, results_dir, cnn_model, rf_model, args.start_fold, args.end_fold, args.progress_percent)
