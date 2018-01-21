from __future__ import division
from keras import backend as K
import numpy as np

from keras.callbacks import Callback
from easydict import EasyDict as edict


class HistoryLog(Callback):
    def on_train_begin(self, logs={}):
        self.training = edict({'loss': []})
        self.epoch = edict({'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []})

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.acc.append(logs.get('acc'))
        self.epoch.loss.append(logs.get('loss'))
        self.epoch.val_acc.append(logs.get('val_acc'))
        self.epoch.val_loss.append(logs.get('val_loss'))

    def on_batch_end(self, batch, logs={}):
        self.training.loss.append(logs.get('loss'))

    def log_training_loss(self, fpath):
        training_loss = np.array(self.training.loss)
        np.savetxt(fpath, training_loss, delimiter=",")

    def log_epoch(self, fpath):
        epoch = np.asarray([self.epoch.loss, self.epoch.val_loss, self.epoch.acc, self.epoch.val_acc])
        np.savetxt(fpath, epoch.T, delimiter=",")


def load_batch(user_features, minibatch, feature_vector_length, batch_size=None, timestep=None, num_classes=21):

    if isinstance(minibatch, list):
        if not batch_size:
            batch_size = len(minibatch)
    else:
        minibatch = [minibatch]
        if not batch_size:
            batch_size = 1

    if not timestep:
        timestep = minibatch[0].size

    batch_x = np.zeros((batch_size, timestep, feature_vector_length, ), dtype=K.floatx())
    batch_y = np.zeros((batch_size, timestep, num_classes), dtype='float32')

    for i, batch in enumerate(minibatch):
        day = user_features[batch.user_id][batch.date]
        for j, ind in enumerate(batch.indices):
            image = day.images[ind]
            batch_x[i, j] = image.features
            batch_y[i, j, image.label] = 1.

    return batch_x, batch_y


def generate_batch(user_features, batches, feature_vector_length, batch_size, timestep, steps_per_epoch, num_classes=21):
    while True:
        np.random.shuffle(batches)
        i = 0
        while i < steps_per_epoch:
            minibatch = batches[i:i+batch_size]
            batch_x, batch_y = load_batch(user_features, batches[i], feature_vector_length, batch_size, timestep, num_classes)
            i += batch_size
            yield (batch_x, batch_y)
