from __future__ import division

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import backend as K
from easydict import EasyDict as edict

import experiments as exp
import utils.gpu
import argparse
from train_rf import train_on_cnn


def parse_args():
    parser = argparse.ArgumentParser(description='Train temporal RF based on CNN networks')
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
                        default='resNet50', type=str)
    parser.add_argument('--start_fold', dest='start_fold',
                        help='Start fold number to train',
                        default=None, type=int)
    parser.add_argument('--end_fold', dest='end_fold',
                        help='End fold number to train',
                        default=5, type=int)
    parser.add_argument('--num_estimators', dest='num_estimators',
                        help='Number of estimators for Random Forest',
                        default=500, type=int)
    parser.add_argument('-l', '--layer', dest='layers',
                        help='Layers to be extracted for the Random Forest training',
                        required=True, nargs='+', type=str)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if K.backend() == 'tensorflow':
        import keras.backend.tensorflow_backend as KTF

        KTF.set_session(utils.gpu.get_session(args.gpu_fraction))
        backend = 'tf'
    else:
        backend = 'th'

    weights_dir = os.path.realpath(args.weights_dir)
    weights_dir = os.path.join(weights_dir, args.network)

    if args.network == 'vgg-16':
        cnn_model = edict({'weights': weights_dir + '/weights.vgg-16.best.tf.hdf5',
                       'name': 'vgg-16',
                       'img_width': 224,
                       'img_height': 224,
                       'load': exp.vgg16_second_phase_model})
    elif args.network == 'resNet50':
        cnn_model = edict(
            {'weights': 'imagenet',
             'name': 'resNet50',
             'img_width': 224,
             'img_height': 224,
             'load': exp.resNet50})
    elif args.network == 'inceptionV3':
        cnn_model = edict(
            {'weights': 'imagenet',
             'name': 'inceptionV3',
             'img_width': 299,
             'img_height': 299,
             'load': exp.inceptionV3_first_phase_model})

    rf_model = edict({'num_estimators': args.num_estimators,
                      'name': cnn_model.name + '.RF.layers_' + '_'.join(args.layers),
                      'layers': args.layers})

    utils.makedirs(weights_dir)
    train_on_cnn(args.data_dir, weights_dir, cnn_model, rf_model, args.start_fold, args.end_fold)
