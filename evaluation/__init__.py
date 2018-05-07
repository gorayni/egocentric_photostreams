import os
import numpy as np
import csv
import warnings

import sklearn.metrics as metrics


class Metrics(object):
    def __init__(self, accuracy=None, macro_precision=None, macro_recall=None, macro_f1_score=None, recall=None,
                 confusion_matrix=None):
        self.accuracy = accuracy
        self.macro_precision = macro_precision
        self.macro_recall = macro_recall
        self.macro_f1_score = macro_f1_score
        self.recall = recall
        self.confusion_matrix = confusion_matrix


class Fold(Metrics):
    def __init__(self, number, img_filepaths, groundtruth, predictions, probabilities, num_categories=21, num_estimators=None):
        super(Fold, self).__init__()
        self.number = number
        self.img_filepaths = img_filepaths
        self.groundtruth = groundtruth
        self.probabilities = probabilities
        self.predictions = predictions
        self.num_categories = num_categories
        self.num_estimators = num_estimators

    def evaluate(self):
        self.accuracy = metrics.accuracy_score(self.groundtruth, self.predictions)
        self.macro_precision = metrics.precision_score(self.groundtruth, self.predictions, average='macro')
        self.macro_recall = metrics.recall_score(self.groundtruth, self.predictions, average='macro')
        self.macro_f1_score = metrics.f1_score(self.groundtruth, self.predictions, average='macro')
        self.recall = metrics.recall_score(self.groundtruth, self.predictions, average=None)

        # normalize confusion matrix
        cm = metrics.confusion_matrix(self.groundtruth, self.predictions, labels=np.arange(21)).astype(np.float)
        num_instances_per_class = cm.sum(axis=1)
        zero_indices = num_instances_per_class == 0
        if any(zero_indices):
            num_instances_per_class[zero_indices] = 1
            warnings.warn('One or more classes does not have instances')
        self.confusion_matrix = cm / num_instances_per_class[:, np.newaxis]


class Model(Metrics):
    def __init__(self, name, folds=None, num_categories=21):
        super(Model, self).__init__()
        self.name = name
        self.num_categories = num_categories
        self.folds = list() if not folds else folds

    @property
    def num_folds(self):
        if self.folds:
            return len(self.folds)
        return 0

    def evaluate(self):
        num_folds = self.num_folds
        self.accuracy = np.zeros(num_folds)
        self.macro_precision = np.zeros(num_folds)
        self.macro_recall = np.zeros(num_folds)
        self.macro_f1_score = np.zeros(num_folds)
        self.recall = -np.ones((num_folds, self.num_categories))
        self.confusion_matrix = np.zeros((self.num_categories, self.num_categories))

        for i, fold in enumerate(self.folds):
            fold.evaluate()
            self.accuracy[i] = fold.accuracy
            self.macro_precision[i] = fold.macro_precision
            self.macro_recall[i] = fold.macro_recall
            self.macro_f1_score[i] = fold.macro_f1_score

            indices = np.union1d(np.unique(fold.groundtruth),np.unique(fold.predictions))
            self.recall[i, indices] = fold.recall
            self.confusion_matrix += fold.confusion_matrix / num_folds


def read_results(csv_filepath):
    img_filepaths = list()
    groundtruth = list()
    probabilities = list()
    predictions = list()

    with open(csv_filepath, 'rb') as f:
        reader = csv.reader(f, delimiter=" ")
        for line in reader:
            line = line[0].split(',')
            img_filepaths.append(line[0])
            groundtruth.append(int(line[1]))

            if len(line) > 3:
                probs = np.array(line[2:]).astype(np.float)
                probabilities.append(probs)
                predictions.append(probs.argmax())
            else:
                predictions.append(int(line[2]))

    return img_filepaths, groundtruth, predictions, probabilities


def get_fold_results(results_dir, model):
    csv_filepaths = list()
    prefix = model + '.fold_'
    filenames = sorted(next(os.walk(results_dir))[2])
    for fname in filenames:
        if not fname.endswith('.csv'):
            continue
        if not fname.startswith(prefix):
            continue
        csv_filepaths.append(os.path.join(results_dir, fname))
    return csv_filepaths
