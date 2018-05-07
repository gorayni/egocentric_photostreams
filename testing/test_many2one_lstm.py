from __future__ import division

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import backend as K
from easydict import EasyDict as edict

import experiments as exp
import utils.gpu
import argparse

import numpy as np
import pickle
import ntcir
import ntcir.IO as IO

from experiments.utils import load_batch
from testing import current_fold


def write_results(results, csv_filepath):
    with open(csv_filepath, 'w') as csv_file:
        for img_path, label, prediction in results:
            csv_file.write("{},{},{}".format(img_path, label, prediction))
            csv_file.write('\n')


def test(features_filepath, results_dir, base_model, start_fold, end_fold, timestep, iccv_epic=False,
         progress_percent=0.05):
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
        test_batches = ntcir.get_training_batches(test_split, sequences, timestep=timestep)

        K.set_learning_phase(False)

        weights = base_model.best_weights.format(fold)
        model = base_model.load(feature_vector_length=base_model.feature_vector_length,
                                weights=weights,
                                timestep=timestep)

        num_test_batches = len(test_batches)

        if progress_percent:
            test_progress_percent = int(num_test_batches * progress_percent)
            print "Testing fold {}".format(fold)

        results = list()
        for i, batch in enumerate(test_batches):
            x, y = load_batch(features, batch, feature_vector_length=base_model.feature_vector_length,
                              batch_size=1, timestep=timestep)

            prediction = model.predict_on_batch(x)
            prediction = np.argmax(prediction, axis=2).squeeze()[-1]

            ind = batch.indices[-1]
            image = features[batch.user_id][batch.date].images[ind]

            results.append((image.path, image.label, prediction))
            if progress_percent and (i + 1) % test_progress_percent == 0:
                print("Progress %3.2f%% (%d/%d)" % ((i + 1) / num_test_batches * 100, i + 1, num_test_batches))

        results_fname = "{}.many2one.fold_{}.{}.csv".format(base_model.name, fold, backend)
        results_filepath = os.path.join(results_dir, results_fname)
        write_results(results, results_filepath)

        del model
        if K.backend() == 'tensorflow':
            K.clear_session()


def parse_args():
    parser = argparse.ArgumentParser(description='Test temporal RF based on CNN networks')
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
                        default='vgg-16', type=str)
    parser.add_argument('--start_fold', dest='start_fold',
                        help='Start fold number to train',
                        default=1, type=int)
    parser.add_argument('--end_fold', dest='end_fold',
                        help='End fold number to train',
                        default=1, type=int)
    parser.add_argument('--timestep', dest='timestep',
                        help='timestep',
                        default=10, type=int)
    parser.add_argument('--iccv_epic', dest='iccv_epic',
                        help='ICCV Epic split',
                        default=True, type=bool)
    parser.add_argument('--progress_percent', dest='progress_percent',
                        help='Progress percent to display',
                        default=0.05, type=float)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if K.backend() == 'tensorflow':
        import keras.backend.tensorflow_backend as KTF

        KTF.set_session(utils.gpu.get_session(args.gpu_fraction))
        backend = 'tf'
    else:
        backend = 'th'

    weights_dir = os.path.join(os.path.realpath(args.weights_dir), args.model)
    results_dir = os.path.join(args.results_dir, args.model)

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


    features_filepath = os.path.join(args.features_dir, model.base_model, features_fname)
    utils.makedirs(results_dir)
    test(features_filepath, results_dir, model, args.start_fold, args.end_fold, args.timestep, args.iccv_epic, args.progress_percent)
