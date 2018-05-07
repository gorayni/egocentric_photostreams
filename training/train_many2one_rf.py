from __future__ import division

import gc
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import backend as K
from easydict import EasyDict as edict

import experiments as exp
import utils.gpu
import argparse

from sklearn.ensemble import RandomForestClassifier

import numpy as np
import multiprocessing
import pickle
import ntcir
import ntcir.IO as IO


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


def train(features_filepath, weights_dir, rf_model, start_fold=1, end_fold=1, timestep=5, progress_percent=0.05, iccv_epic=True, features_size=4096, cores=None):
    np.random.seed(42)

    users = IO.load_annotations(ntcir.filepaths)
    sorted_users = ntcir.utils.sort(users)

    num_frames_per_day = 2880
    sequences = ntcir.get_sequences(sorted_users, num_frames_per_day)

    if not start_fold:
        start_fold = current_fold(weights_dir, rf_model.name)

    folds = [str(fold).zfill(2) for fold in range(start_fold, end_fold + 1)]
    for fold in folds:
        with open(features_filepath.format(fold), 'r') as f:
            user_features = pickle.load(f)

        if iccv_epic:
            train_split = ntcir.read_split('datasets/ntcir/training_split.txt')
        else:
            train_split = ntcir.get_split_fold(sorted_users, int(fold) - 1)

        training_batches = ntcir.get_training_batches(train_split, sequences, timestep=timestep)

        num_features = timestep*features_size
        num_training_batches = len(training_batches)

        # Extract features of the images
        features = np.zeros((num_training_batches, num_features))
        targets = np.zeros(num_training_batches)

        if progress_percent:
            training_progress_percent = int(num_training_batches * progress_percent)
            print "Creating training matrix for fold {}".format(fold)

        for i, batch in enumerate(training_batches):
            day = user_features[batch.user_id][batch.date]
            for j, ind in enumerate(batch.indices):
                image = day.images[ind]
                start_ind = j * features_size
                end_ind = (j+1) * features_size
                features[i, start_ind:end_ind] = image.features

            last_ind = batch.indices[-1]
            targets[i] = day.images[last_ind].label

            if progress_percent and (i + 1) % training_progress_percent == 0:
                print("Progress %3.2f%% (%d/%d)" % ((i + 1) / num_training_batches * 100, i + 1, num_training_batches))
        gc.collect()
        if not cores:
            cores = multiprocessing.cpu_count()
        random_forest = RandomForestClassifier(n_estimators=rf_model.num_estimators, n_jobs=cores)
        random_forest.fit(features, targets)

        weights_filepath = os.path.join(weights_dir, "weights." + rf_model.name + ".fold_" + fold + ".pkl")
        with open(weights_filepath, 'w') as f:
            pickle.dump(random_forest, f, pickle.HIGHEST_PROTOCOL)
    return random_forest


def parse_args():
    parser = argparse.ArgumentParser(description='Train temporal RF based on CNN networks')
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
                        default='vgg-16', type=str)
    parser.add_argument('--features_size', dest='features_size',
                        help='CNN output features size',
                        default=None, required=True, type=int)
    parser.add_argument('--start_fold', dest='start_fold',
                        help='Start fold number to train',
                        default=1, type=int)
    parser.add_argument('--end_fold', dest='end_fold',
                        help='End fold number to train',
                        default=1, type=int)
    parser.add_argument('--num_estimators', dest='num_estimators',
                        help='Number of estimators for Random Forest',
                        default=500, type=int)
    parser.add_argument('-l', '--layer', dest='layers',
                        help='Layers to be extracted for the Random Forest training',
                        required=True, nargs='+', type=str)
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

    weights_dir = os.path.join(args.weights_dir, args.model)

    if args.model == 'vgg-16':
        cnn_model = edict({'base_model': 'vgg-16',
                       'name': 'vgg-16',
                       'load': exp.probabilities_plus_lstm})
    elif args.model == 'resNet50':
        cnn_model = edict({'base_model': 'resNet50',
                       'name': 'resNet50',
                       'load': exp.probabilities_plus_lstm})
    elif args.model == 'inceptionV3':
        cnn_model = edict({'base_model': 'inceptionV3',
                       'name': 'inceptionV3',
                       'load': exp.probabilities_plus_lstm})

    rf_model = edict({'num_estimators': args.num_estimators,
                      'name': cnn_model.name + '.Many2One_RF.layers_' + '_'.join(args.layers) + '.timestep_' + str(args.timestep),
                      'layers': args.layers})

    features_fname = 'features.' + cnn_model.base_model + '.layers_' + '_'.join(args.layers) + '.fold_{}.' + backend + '.pkl'
    features_filepath = os.path.join(args.features_dir, cnn_model.base_model, features_fname)
    utils.makedirs(weights_dir)

    train(features_filepath, weights_dir, rf_model, args.start_fold, args.end_fold, args.timestep, features_size=args.features_size)
