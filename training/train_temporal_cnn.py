from __future__ import division

import numpy as np

from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from experiments.utils import HistoryLog
from datetime import datetime
from easydict import EasyDict as edict
from time import time
from training import compute_class_weights

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


def train(data_dir, weights_dir, sgd_params, base_model, start_fold=None, end_fold=5, batch_size=32, class_weights=None):
    np.random.seed(42)

    if not start_fold:
        start_fold = current_fold(weights_dir, base_model.name)

    folds = [str(fold).zfill(2) for fold in range(start_fold, end_fold + 1)]
    for fold in folds:

        # prepare data augmentation configuration
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        val_datagen = ImageDataGenerator(rescale=1. / 255)

        init_weights = base_model.init_weights.format(fold)
        model = base_model.load(weights=init_weights)

        sgd = SGD(lr=sgd_params.lr, decay=sgd_params.decay, momentum=sgd_params.momentum, nesterov=sgd_params.nesterov)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        train_dir = os.path.join(data_dir, fold, 'train')
        train_generator = train_datagen.flow_from_directory(train_dir,
                                                            target_size=(base_model.img_height, base_model.img_width),
                                                            class_mode='categorical',
                                                            batch_size=batch_size)
        val_dir = os.path.join(data_dir, fold, 'test')
        validation_generator = val_datagen.flow_from_directory(val_dir,
                                                               target_size=(
                                                                   base_model.img_height, base_model.img_width),
                                                               class_mode='categorical',
                                                               batch_size=batch_size)

        # checkpoint
        backend = 'tf' if K.backend() == 'tensorflow' else 'th'

        base_model_weights = "weights." + base_model.name + ".fold_" + fold + ".epoch_{epoch:02d}." + backend + ".hdf5"
        weights_filepath = os.path.join(weights_dir, base_model_weights)
        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_acc', verbose=1, save_best_only=False)
        history = HistoryLog()

        steps_per_epoch = num_images_fits_batch(train_dir, batch_size)
        validation_steps = num_images_fits_batch(val_dir, batch_size)

        if not class_weights:
            class_weights = compute_class_weights(train_dir)

        # fine-tune the model
        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=10,
            callbacks=[checkpoint, history],
            class_weight=class_weights,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            initial_epoch=0)

        ts = time()
        timestamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

        loss_filepath = os.path.join(weights_dir,
                                     "{}.fold_{}.loss.{}.log".format(base_model.name, fold, timestamp))
        history.log_training_loss(loss_filepath)

        epoch_filepath = os.path.join(weights_dir,
                                      "{}.fold_{}.epoch.{}.log".format(base_model.name, fold, timestamp))
        history.log_epoch(epoch_filepath)

        if K.backend() == 'tensorflow':
            K.clear_session()


def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN base networks')
    parser.add_argument('--gpu_fraction', dest='gpu_fraction',
                        help='GPU fraction usage',
                        default=0.8, type=float)
    parser.add_argument('--data_dir', dest='data_dir',
                        help='Data directory where the folds are located',
                        default='data/temporal', type=str)
    parser.add_argument('--weights_dir', dest='weights_dir',
                        help='Directory where the weights are stored',
                        default='weights/temporal/vgg-16', type=str)
    parser.add_argument('--network', dest='network',
                        help='CNN to be trained',
                        default='resNet50', type=str)
    parser.add_argument('--start_fold', dest='start_fold',
                        help='Start fold number to train',
                        default=None, type=int)
    parser.add_argument('--end_fold', dest='end_fold',
                        help='End fold number to train',
                        default=5, type=int)
    parser.add_argument('--phase', dest='phase',
                        help='Training phase',
                        default=1, type=int)
    parser.add_argument('--learning_rate', dest='learning_rate',
                        help='Learning rate for SGD',
                        default=None, type=float)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='Batch size',
                        default=32, type=int)
    parser.add_argument('--class_weights', dest='class_weights',
                        help='Class weighting scheme',
                        default=None, type=bool)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if K.backend() == 'tensorflow':
        import keras.backend.tensorflow_backend as KTF

        KTF.set_session(utils.gpu.get_session(args.gpu_fraction))

    weights_dir = os.path.join(args.weights_dir, args.network)
    if args.network == 'vgg-16':
        if args.phase == 1:
            lr = args.learning_rate if args.learning_rate else 0.00001
            sgd_params = edict({'lr': lr, 'decay': 0.000005, 'momentum': 0.9, 'nesterov': True, 'batch_size': args.batch_size})
            model = edict({'init_weights': 'imagenet',
                           'name': 'vgg-16',
                           'img_width': 224,
                           'img_height': 224,
                           'load': exp.vgg16_first_phase_model})
        else:
            lr = args.learning_rate if args.learning_rate else 0.00004
            sgd_params = edict({'lr': lr, 'decay': 0.000005, 'momentum': 0.9, 'nesterov': True, 'batch_size': args.batch_size})
            model = edict({'init_weights': weights_dir + '/weights.vgg-16.phase_1.fold_{}.best.tf.hdf5',
                           'name': 'vgg-16',
                           'img_width': 224,
                           'img_height': 224,
                           'load': exp.vgg16_second_phase_model})

    elif args.network == 'resNet50':
        if args.phase == 1:
            lr = args.learning_rate if args.learning_rate else 0.001
            sgd_params = edict({'lr': lr, 'decay': 0.000005, 'momentum': 0.9, 'nesterov': True})
            model = edict({'init_weights': 'imagenet',
                           'name': 'resNet50.phase_' + str(args.phase) + '.lr_' + str(lr) +  '.batch_size_' + str(args.batch_size),
                           'img_width': 224,
                           'img_height': 224,
                           'load': exp.resNet50})
        elif args.phase == 2:
            init_weights = weights_dir + '/weights.resNet50.phase_1.fold_{}.best.tf.hdf5'
            lr = args.learning_rate if args.learning_rate else 0.004
            sgd_params = edict({'lr': lr, 'decay': 0.000005, 'momentum': 0.9, 'nesterov': True, 'batch_size': args.batch_size})
            model = edict({'init_weights': init_weights,
                           'name': 'resNet50.phase_' + str(args.phase) + '.lr_' + str(lr) +  '.batch_size_' + str(args.batch_size),
                           'img_width': 224,
                           'img_height': 224,
                           'load': exp.resNet50_second_phase})

    elif args.network == 'inceptionV3':
        if args.phase == 1:
            lr = args.learning_rate if args.learning_rate else 0.00001
            sgd_params = edict({'lr': lr, 'decay': 0.000005, 'momentum': 0.9, 'nesterov': True})
            model = edict({'init_weights': 'imagenet',
                           'name': 'inceptionV3.phase_' + str(args.phase) + '.lr_' + str(lr) + '.batch_size_' + str(args.batch_size),
                           'img_width': 299,
                           'img_height': 299,
                           'load': exp.inceptionV3_first_phase_model})
        elif args.phase == 2:
            init_weights = weights_dir + '/weights.inceptionV3.phase_1.fold_{}.best.tf.hdf5'
            lr = args.learning_rate if args.learning_rate else 0.00001
            sgd_params = edict({'lr': lr, 'decay': 0.000005, 'momentum': 0.9, 'nesterov': True})
            model = edict({'init_weights': init_weights,
                           'name': 'inceptionV3.phase_' + str(args.phase) + '.lr_' + str(lr) + '.batch_size_' + str(args.batch_size),
                           'img_width': 299,
                           'img_height': 299,
                           'load': exp.inceptionV3_second_phase_model})

    utils.makedirs(weights_dir)
    train(args.data_dir, weights_dir, sgd_params, model, args.start_fold, args.end_fold, args.batch_size, args.class_weights)
