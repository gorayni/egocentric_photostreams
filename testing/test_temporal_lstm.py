from __future__ import division

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from keras import backend as K
from easydict import EasyDict as edict
import pickle
import ntcir
import ntcir.IO as IO
from experiments.utils import load_batch
from testing import current_fold

import os
import experiments as exp
import utils.gpu
import argparse


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


def write_results(img_paths, groundtruth, predictions, csv_filepath):
    with open(csv_filepath, 'w') as csv_file:
        for i, img_path in enumerate(img_paths):
            csv_file.write("{},{},{}".format(img_path, groundtruth[i], predictions[i]))
            csv_file.write('\n')


def test(features_filepath, results_dir, base_model, start_fold, end_fold, timestep, iccv_epic=False):
    users = IO.load_annotations(ntcir.filepaths)
    sorted_users = ntcir.utils.sort(users)

    num_frames_per_day = 2880
    sequences = ntcir.get_sequences(sorted_users, num_frames_per_day)

    backend = 'tf' if K.backend() == 'tensorflow' else 'th'

    if not start_fold:
        start_fold = current_fold(results_dir, base_model.name + '.fold')

    folds = [str(fold).zfill(2) for fold in range(start_fold, end_fold + 1)]
    for fold in folds:
        with open(features_filepath.format(fold), 'r') as f:
            features = pickle.load(f)

        if iccv_epic:
            test_split = ntcir.read_split('datasets/ntcir/test_split.txt')
        else:
            test_split = ntcir.get_split_fold(sorted_users, int(fold) - 1, False)
        test_batches = ntcir.get_batches(test_split, sequences, timestep=timestep, include_last=True)

        K.set_learning_phase(False)

        weights = base_model.best_weights.format(fold)
        model = base_model.load(feature_vector_length=base_model.feature_vector_length,
                           weights=weights,
                           timestep=timestep)

        frames = list()
        groundtruth = list()
        predictions = list()
        for i, batch in enumerate(test_batches):
            x, y = load_batch(features, batch, feature_vector_length=base_model.feature_vector_length,
                              batch_size=1, timestep=timestep)

            prediction = model.predict_on_batch(x)
            prediction = np.argmax(prediction, axis=2).squeeze()[0:batch.size]

            predictions.extend(prediction)
            groundtruth.extend(np.argmax(y, axis=2).squeeze()[0:batch.size])

            for j, ind in enumerate(batch.indices):
                image = features[batch.user_id][batch.date].images[ind]
                frames.append(image.path)

        results_fname = "{}.fold_{}.{}.csv".format(base_model.name, fold, backend)
        results_filepath = os.path.join(results_dir, results_fname)
        write_results(frames, groundtruth, predictions, results_filepath)

        del model
        if K.backend() == 'tensorflow':
            K.clear_session()


def parse_args():
    parser = argparse.ArgumentParser(description='Test features+LSTMs')
    parser.add_argument('--gpu_fraction', dest='gpu_fraction',
                        help='GPU fraction usage',
                        default=0.8, type=float)
    parser.add_argument('--features_dir', dest='features_dir',
                        help='Directory where the pre-computed features are stored',
                        default='features', type=str)
    parser.add_argument('--results_dir', dest='results_dir',
                        help='Directory where the CSV results will be stored',
                        default='results', type=str)
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
    parser.add_argument('--timestep', dest='timestep',
                        help='timestep',
                        default=10, type=int)
    parser.add_argument('--iccv_epic', dest='iccv_epic',
                        help='ICCV Epic split',
                        default=False, type=bool)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if K.backend() == 'tensorflow':
        import keras.backend.tensorflow_backend as KTF

        KTF.set_session(utils.gpu.get_session(args.gpu_fraction))
        backend = 'tf'
    else:
        backend = 'th'

    weights_dir = os.path.realpath(os.path.join(args.weights_dir, args.model))

    if args.model == 'vgg-16':
        model = edict({'base_model': 'vgg-16',
                       'name': 'vgg-16+LSTM.timestep_' + str(args.timestep),
                       'best_weights': os.path.join(weights_dir, 'weights.vgg-16+LSTM.timestep_' + str(
                           args.timestep) + '.fold_{}.best.' + backend + ".hdf5"),
                       'feature_vector_length': 21,
                       'load': exp.probabilities_plus_lstm})
        features_fname = 'features.vgg-16.RF.layers_fc1.fold_{}.tf.pkl'

    elif args.model == 'resNet50':
        model = edict({'base_model': 'resNet50',
                       'name': 'resNet50+LSTM.timestep_' + str(args.timestep),
                       'best_weights': os.path.join(weights_dir, 'weights.resNet50+LSTM.timestep_' + str(
                           args.timestep) + '.fold_{}.best.' + backend + ".hdf5"),
                       'feature_vector_length': 2048,
                       'load': exp.resNet50features_plus_lstm})
        features_fname = 'features.' + model.base_model + '.fold_{}.pkl'

    elif args.network == 'inceptionV3':
        model = edict({'base_model': 'inceptionV3',
                       'name': 'inceptionV3+LSTM.timestep_' + str(args.timestep),
                       'best_weights': os.path.join(weights_dir, 'weights.inceptionV3+LSTM.timestep_' + str(
                           args.timestep) + '.fold_{}.best.' + backend + ".hdf5"),
                       'feature_vector_length': 2048,
                       'load': exp.inceptionV3features_plus_lstm})
        features_fname = 'features.' + model.base_model + '.fold_{}.pkl'

    results_dir = os.path.join(args.results_dir, model.base_model)
    utils.makedirs(results_dir)

    features_filepath = os.path.join(args.features_dir, model.base_model, features_fname)

    test(features_filepath, results_dir, model, args.start_fold, args.end_fold, args.timestep)
