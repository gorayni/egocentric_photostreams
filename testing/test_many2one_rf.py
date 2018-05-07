from __future__ import division

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
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

from test_rf import load_random_forest


def write_results(results, csv_filepath):
    with open(csv_filepath, 'w') as csv_file:
        for img_path, label, prediction in results:
            csv_file.write("{},{},{}".format(img_path, label, prediction))
            csv_file.write('\n')


def test(features_filepath, results_dir, rf_model, start_fold=1, end_fold=1, timestep=5, progress_percent=0.05,
         iccv_epic=True, features_size=4096):
    np.random.seed(42)

    users = IO.load_annotations(ntcir.filepaths)
    sorted_users = ntcir.utils.sort(users)

    num_frames_per_day = 2880
    sequences = ntcir.get_sequences(sorted_users, num_frames_per_day)

    folds = [str(fold).zfill(2) for fold in range(start_fold, end_fold + 1)]
    for fold in folds:
        with open(features_filepath.format(fold), 'r') as f:
            user_features = pickle.load(f)

        weights = rf_model.weights.format(fold)
        rf = load_random_forest(weights)

        if iccv_epic:
            test_split = ntcir.read_split('datasets/ntcir/test_split.txt')
        else:
            test_split = ntcir.get_split_fold(sorted_users, int(fold) - 1, False)
        test_batches = ntcir.get_training_batches(test_split, sequences, timestep=timestep)

        num_features = timestep * features_size
        num_test_batches = len(test_batches)

        features = np.zeros((num_test_batches, num_features))
        img_paths = list()
        labels = list()
        for i, batch in enumerate(test_batches):
            day = user_features[batch.user_id][batch.date]
            for j, ind in enumerate(batch.indices):
                image = day.images[ind]
                start_ind = j * features_size
                end_ind = (j + 1) * features_size
                features[i, start_ind:end_ind] = image.features

            last_ind = batch.indices[-1]
            img_paths.append(day.images[last_ind].path)
            labels.append(day.images[last_ind].label)

        predictions = rf.predict(features)

        results = list()
        for i in range(num_test_batches):
            results.append((img_paths[i], labels[i], predictions[i]))

        #ORIGINAL
        # num_features = timestep * features_size
        # num_test_batches = len(test_batches)
        #
        # if progress_percent:
        #     test_progress_percent = int(num_test_batches * progress_percent)
        #     print "Testing fold {}".format(fold)
        #
        # results = list()
        # features = np.zeros(num_features)
        # for i, batch in enumerate(test_batches):
        #     day = user_features[batch.user_id][batch.date]
        #     for j, ind in enumerate(batch.indices):
        #         image = day.images[ind]
        #         start_ind = j * features_size
        #         end_ind = (j + 1) * features_size
        #         features[start_ind:end_ind] = image.features
        #
        #     last_ind = batch.indices[-1]
        #     img_path = day.images[last_ind].path
        #     label = day.images[last_ind].label
        #     prediction = rf.predict([features])[0].astype(np.int)
        #
        #     results.append((img_path, label, prediction))
        #     if progress_percent and (i + 1) % test_progress_percent == 0:
        #         print("Progress %3.2f%% (%d/%d)" % ((i + 1) / num_test_batches * 100, i + 1, num_test_batches))

        results_fname = "{}.fold_{}.{}.csv".format(rf_model.name, fold, backend)
        results_filepath = os.path.join(results_dir, results_fname)
        write_results(results, results_filepath)



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
    parser.add_argument('--features_size', dest='features_size',
                    help='CNN output features size',
                    default=None, required=True, type=int)
    parser.add_argument('--model', dest='model',
                        help='Model where the features were taken',
                        default='vgg-16', type=str)
    parser.add_argument('--start_fold', dest='start_fold',
                        help='Start fold number to train',
                        default=1, type=int)
    parser.add_argument('--end_fold', dest='end_fold',
                        help='End fold number to train',
                        default=1, type=int)
    parser.add_argument('-l', '--layer', dest='layers',
                        help='Layers to be extracted for the Random Forest training',
                        required=True, nargs='+', type=str)
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
        cnn_model = edict({'base_model': 'vgg-16',
                           'name': 'vgg-16',
                           'load': exp.probabilities_plus_lstm})

    elif args.model == 'resNet50':
        cnn_model = edict({'base_model': 'resNet50',
                       'name': 'resNet50',
                       'load': exp.probabilities_plus_lstm})

    elif args.network == 'inceptionV3':
        cnn_model = edict({'base_model': 'inceptionV3',
                       'name': 'inceptionV3',
                       'load': exp.probabilities_plus_lstm})

    features_fname = 'features.' + cnn_model.base_model + '.layers_' + '_'.join(args.layers) + '.fold_{}.' + backend + '.pkl'

    rf_model = edict(
        {'name': cnn_model.name + '.RF.layers_' + '_'.join(args.layers) + '.timestep_' + str(args.timestep),
         'weights': weights_dir + '/weights.' + cnn_model.name + '.Many2One_RF.layers_' + '_'.join(
             args.layers) + '.timestep_' + str(args.timestep) + ".fold_{}.pkl",
         'layers': args.layers})

    features_filepath = os.path.join(args.features_dir, cnn_model.name, features_fname)
    utils.makedirs(results_dir)

    test(features_filepath, results_dir, rf_model, args.start_fold, args.end_fold, args.timestep, args.progress_percent,
         args.iccv_epic, features_size=args.features_size)
