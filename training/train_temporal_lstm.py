from __future__ import division

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from experiments.utils import HistoryLog
from datetime import datetime
from easydict import EasyDict as edict
from time import time
import pickle
import ntcir
import ntcir.IO as IO
from experiments.utils import generate_batch

import os
import experiments as exp
import utils.gpu
import argparse


def num_images_fits_batch(split_path, batch_size):
    num_images = sum([len(files) for _, _, files in os.walk(split_path)])
    return np.floor_divide(num_images, batch_size) * batch_size


def current_fold(weights_dir, prefix):
    epoch_logs = list()
    filenames = sorted(next(os.walk(weights_dir))[2])
    for fname in filenames:
        if not fname.startswith(prefix):
            continue
        if not 'epoch' in fname:
            continue
        epoch_logs.append(fname)
    if epoch_logs:
        return int(epoch_logs[-1].split('.')[2].split('_')[1]) + 1
    return 1


def train(features_filepath, weights_dir, sgd_params, base_model, start_fold=None, end_fold=5, timestep=10,
          batch_size=1, iccv_epic=False):
    np.random.seed(42)

    users = IO.load_annotations(ntcir.filepaths)
    sorted_users = ntcir.utils.sort(users)

    num_frames_per_day = 2880
    sequences = ntcir.get_sequences(sorted_users, num_frames_per_day)

    backend = 'tf' if K.backend() == 'tensorflow' else 'th'

    if not start_fold:
        start_fold = current_fold(weights_dir, base_model.name)

    folds = [str(fold).zfill(2) for fold in range(start_fold, end_fold + 1)]
    for fold in folds:
        with open(features_filepath.format(fold), 'r') as f:
            features = pickle.load(f)

        if iccv_epic:
            train_split = ntcir.read_split('datasets/ntcir/training_split.txt')
            test_split = ntcir.read_split('datasets/ntcir/validation_split.txt')
        else:
            train_split = ntcir.get_split_fold(sorted_users, int(fold) - 1)
            test_split = ntcir.get_split_fold(sorted_users, int(fold) - 1, False)

        training_batches = ntcir.get_training_batches(train_split, sequences, timestep=timestep)
        test_batches = ntcir.get_batches(test_split, sequences, timestep=timestep)

        K.set_learning_phase(1)

        model = base_model.load(feature_vector_length=base_model.feature_vector_length, timestep=timestep)
        sgd = SGD(lr=sgd_params.lr, decay=sgd_params.decay, momentum=sgd_params.momentum, nesterov=sgd_params.nesterov)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        steps_per_epoch = int(len(training_batches) / batch_size)
        train_generator = generate_batch(features, training_batches, base_model.feature_vector_length, batch_size,
                                         timestep, steps_per_epoch)
        validation_steps = int(len(test_batches) / batch_size)
        validation_generator = generate_batch(features, test_batches, base_model.feature_vector_length, batch_size,
                                              timestep, validation_steps)

        # checkpoint
        base_model_weights = "weights." + base_model.name + ".fold_" + fold + ".epoch_{epoch:02d}." + backend + ".hdf5"
        weights_filepath = os.path.join(weights_dir, base_model_weights)
        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_acc', verbose=1, save_best_only=False)
        history = HistoryLog()

        # fine-tune the model
        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=10,
            callbacks=[checkpoint, history],
            validation_data=validation_generator,
            validation_steps=validation_steps)

        ts = time()
        timestamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

        loss_filepath = os.path.join(weights_dir,
                                     "{}.fold_{}.loss.{}.log".format(base_model.name, fold, timestamp))
        history.log_training_loss(loss_filepath)

        epoch_filepath = os.path.join(weights_dir,
                                      "{}.fold_{}.epoch.{}.log".format(base_model.name, fold, timestamp))
        history.log_epoch(epoch_filepath)

        del model
        if K.backend() == 'tensorflow':
            K.clear_session()


def parse_args():
    parser = argparse.ArgumentParser(description='Train features+LSTMs')
    parser.add_argument('--gpu_fraction', dest='gpu_fraction',
                        help='GPU fraction usage',
                        default=0.8, type=float)
    parser.add_argument('--features_dir', dest='features_dir',
                        help='Directory where the pre-computed features are stored',
                        default='features', type=str)
    parser.add_argument('--weights_dir', dest='weights_dir',
                        help='Directory where the weights are stored',
                        default='weights', type=str)
    parser.add_argument('--model', dest='model',
                        help='Model where the features were taken',
                        default='resNet50', type=str)
    parser.add_argument('--start_fold', dest='start_fold',
                        help='Start fold number to train',
                        default=None, type=int)
    parser.add_argument('--end_fold', dest='end_fold',
                        help='End fold number to train',
                        default=5, type=int)
    parser.add_argument('--learning_rate', dest='learning_rate',
                        help='Learning rate for SGD',
                        default=None, type=float)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='Batch size',
                        default=1, type=int)
    parser.add_argument('--timestep', dest='timestep',
                        help='timestep',
                        default=10, type=int)
    parser.add_argument('--iccv_epic', dest='iccv_epic',
                        help='ICCV Epic split',
                        default=False, type=bool)
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

    weights_dir = os.path.join(args.weights_dir, args.model)

    if args.model == 'vgg-16':
        lr = args.learning_rate if args.learning_rate else 0.001
        sgd_params = edict({'lr': lr, 'decay': 0.000005, 'momentum': 0.9, 'nesterov': True})

        model = edict({'base_model': 'vgg-16',
                       'name': 'vgg-16+LSTM.RF.layers_' + '_'.join(args.layers) +'.lr_' + str(
                           lr) + '.batch_size_' + str(args.batch_size) + '.timestep_' + str(args.timestep),
                       'feature_vector_length': 21,
                       'load': exp.probabilities_plus_lstm})
    elif args.model == 'resNet50':
        lr = args.learning_rate if args.learning_rate else 0.0001
        sgd_params = edict({'lr': lr, 'decay': 0.000005, 'momentum': 0.9, 'nesterov': True})

        model = edict({'base_model': 'resNet50',
                       'name': 'resNet50+LSTM.RF.layers_' + '_'.join(args.layers) +'.lr_' + str(
                           lr) + '.batch_size_' + str(args.batch_size) + '.timestep_' + str(args.timestep),
                       'feature_vector_length': 21,
                       'load': exp.probabilities_plus_lstm})
    elif args.model == 'inceptionV3':
        lr = args.learning_rate if args.learning_rate else 0.00001
        sgd_params = edict({'lr': lr, 'decay': 0.000005, 'momentum': 0.9, 'nesterov': True})
        model = edict({'base_model': 'inceptionV3',
                       'name': 'inceptionV3+LSTM.RF.layers_' + '_'.join(args.layers) + '.lr_' + str(
                           lr) + '.batch_size_' + str(args.batch_size) + '.timestep_' + str(args.timestep),
                       'feature_vector_length': 21,
                       'load': exp.probabilities_plus_lstm})

    features_fname = 'features.' + model.base_model + '.RF.layers_' + '_'.join(args.layers) + '.fold_{}.' + backend + '.pkl'
    features_filepath = os.path.join(args.features_dir, model.base_model, features_fname)

    utils.makedirs(weights_dir)
    train(features_filepath, weights_dir, sgd_params, model, args.start_fold, args.end_fold, args.timestep, args.batch_size, args.iccv_epic)
