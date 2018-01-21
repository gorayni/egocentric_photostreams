from __future__ import division

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from easydict import EasyDict as edict

import os
import experiments as exp
import utils.gpu
import argparse
from testing import load_image
from testing import read_fold_dir
from testing import current_fold
from test_cnn import write_results


def test(data_dir, results_dir, base_model, start_fold=None, end_fold=5):
    if not start_fold:
        start_fold = current_fold(results_dir, base_model.name + '.fold')

    backend = 'tf' if K.backend() == 'tensorflow' else 'th'
    target_size = (base_model.img_height, base_model.img_width)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    folds = [str(fold).zfill(2) for fold in range(start_fold, end_fold + 1)]
    for fold in folds:

        weights = base_model.best_weights.format(fold)
        model = base_model.load(weights=weights)

        results = list()
        test_dir = os.path.join(data_dir, fold, 'test')
        test_images = read_fold_dir(test_dir)
        for label, img_path in test_images:
            img = load_image(test_datagen, img_path, target_size)
            predictions = model.predict(img)
            results.append((img_path, label, predictions))

        results_fname = "{}.fold_{}.{}.csv".format(base_model.name, fold, backend)
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

    weights_dir = os.path.realpath(os.path.join(args.weights_dir, args.network))

    if args.network == 'vgg-16':
        model = edict(
            {'best_weights': weights_dir + '/weights.vgg-16.best.tf.hdf5',
             'name': 'vgg-16',
             'img_width': 224,
             'img_height': 224,
             'load': exp.vgg16_second_phase_model})

    elif args.network == 'resNet50':
        model = edict(
            {'best_weights': weights_dir + '/weights.resNet50.phase_2.fold_{}.best.' + backend + '.hdf5',
             'name': 'resNet50',
             'img_width': 224,
             'img_height': 224,
             'load': exp.resNet50_second_phase})

    elif args.network == 'inceptionV3':
        model = edict(
            {'best_weights': weights_dir + '/weights.inceptionV3.phase_2.fold_{}.best.' + backend + '.hdf5',
             'name': 'inceptionV3',
             'img_width': 299,
             'img_height': 299,
             'load': exp.inceptionV3_second_phase_model})

    results_dir = os.path.join(args.results_dir, args.network)
    utils.makedirs(results_dir)
    test(args.data_dir, results_dir, model, args.start_fold, args.end_fold)
