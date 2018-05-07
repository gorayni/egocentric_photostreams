from __future__ import division

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import backend as K
from easydict import EasyDict as edict

import os
import experiments as exp
import utils.gpu
import argparse
from test_rf import test_on_cnn


def parse_args():
    parser = argparse.ArgumentParser(description='Test CNN base networks')
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

    weights_dir = os.path.realpath(os.path.join(args.weights_dir, args.network))

    if args.network == 'vgg-16':
        cnn_model = edict(
            {'best_weights': weights_dir + '/weights.vgg-16.best.tf.hdf5',
             'name': 'vgg-16',
             'img_width': 224,
             'img_height': 224,
             'load': exp.vgg16_second_phase_model})
    elif args.network == 'resNet50':
        cnn_model = edict(
            {'best_weights': 'imagenet',
             'name': 'resNet50',
             'img_width': 224,
             'img_height': 224,
             'load': exp.resNet50})
    elif args.network == 'inceptionV3':
        cnn_model = edict(
            {'best_weights': 'imagenet',
             'name': 'inceptionV3',
             'img_width': 299,
             'img_height': 299,
             'load': exp.inceptionV3_first_phase_model})

    rf_model = edict({'name': cnn_model.name + '.RF.layers_' + '_'.join(args.layers),
                      'weights': weights_dir + "/weights." + args.network + '.RF.layers_' + '_'.join(
                          args.layers) + ".fold_{}.pkl",
                      'layers': args.layers})

    results_dir = os.path.join(args.results_dir, cnn_model.name)
    utils.makedirs(results_dir)

    test_on_cnn(args.data_dir, results_dir, cnn_model, rf_model, args.start_fold, args.end_fold, args.progress_percent)
