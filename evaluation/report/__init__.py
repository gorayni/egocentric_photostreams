from __future__ import division

import os
import sys
import numpy as np
from easydict import EasyDict as edict
from jinja2 import Environment, FileSystemLoader
from skimage import io

MOD_CURRENT_PATH = sys.modules[__name__].__file__
PATH = os.path.dirname(os.path.realpath(MOD_CURRENT_PATH))
TEMPLATE_ENVIRONMENT = Environment(autoescape=False, loader=FileSystemLoader(PATH), trim_blocks=False)


def _to_dict(labels, models):
    models_dict = dict()
    for m in models:
        recall_mean = m.recall.mean(axis=0)
        recall_std = m.recall.std(axis=0)
        recall = dict()
        for i, label in enumerate(labels):
            recall[label] = {'mean': recall_mean[i], 'std': recall_std[i]}

        models_dict[m.name] = {'accuracy': {'mean': m.accuracy.mean(), 'std': m.accuracy.std()},
                               'macro_precision': {'mean': m.macro_precision.mean(),
                                                   'std': m.macro_precision.std()},
                               'macro_recall': {'mean': m.macro_recall.mean(), 'std': m.macro_recall.std()},
                               'macro_f1': {'mean': m.macro_f1_score.mean(), 'std': m.macro_f1_score.std()},
                               'recall': recall}
    return models_dict


class ClassificationComparisonTable():
    def __init__(self, labels, models):
        html_template = TEMPLATE_ENVIRONMENT.get_template('classification_comparison_table_template.html')
        latex_template = TEMPLATE_ENVIRONMENT.get_template('classification_comparison_table_template.tex')

        models = _to_dict(labels, models)
        model_names = sorted(models.keys())
        max_rows = dict()
        metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
        metric_formatted_names = {'macro_recall': 'Macro recall', 'macro_f1': 'Macro F1-score',
                                  'macro_precision': 'Macro precision', 'accuracy': 'Accuracy'}
        for metric in metrics:
            max_rows[metric] = np.max([models[model_name][metric]['mean'] for model_name in model_names])

        for label in labels:
            max_rows[label] = np.max([models[model_name]['recall'][label]['mean'] for model_name in model_names])

        self.html = html_template.render(labels=labels, model_names=model_names, models=models, max_rows=max_rows,
                                         metrics=metrics, metric_formatted_names=metric_formatted_names)
        self.latex = latex_template.render(labels=labels, model_names=model_names, models=models, max_rows=max_rows,
                                           metrics=metrics, metric_formatted_names=metric_formatted_names).replace(
            'textbf{ ', 'textbf{')

    def _repr_html_(self):
        return self.html

    def _repr_latex_(self):
        return self.latex


class TopClassificationTable():
    def __init__(self, labels, predictions_ind, scores, other_model_prediction, title=None, num_top_scores=5):

        html_template = TEMPLATE_ENVIRONMENT.get_template('top5_classification_results_template.html')
        latex_template = TEMPLATE_ENVIRONMENT.get_template('top5_classification_results_template.tex')

        classification_results = [{'number': i + 1, 'label': labels[predictions_ind[i]], 'score': scores[i]} for i in
                                  range(num_top_scores)]

        for i, prediction_ind in enumerate(predictions_ind):
            if prediction_ind == other_model_prediction:
                if i >= num_top_scores:
                    classification_results.append(
                        {'number': i + 1, 'label': labels[predictions_ind[i]], 'score': scores[i]})
                other_model_prediction_num = i + 1
                break
        if not title:
            title = 'Top {} scores'.format(num_top_scores)
        self.html = html_template.render(title=title, classification_results=classification_results,
                                         other_model_prediction_num=other_model_prediction_num)
        self.latex = latex_template.render(title=title, classification_results=classification_results,
                                           other_model_prediction_num=other_model_prediction_num).replace('textbf{ ',
                                                                                                          'textbf{')

    def _repr_html_(self):
        return self.html

    def _repr_latex_(self):
        return self.latex


def get_predictions_comparison(base_model, other_model):
    predictions = edict({'better': list(),
                         'worse': list(),
                         'both_failed': list()})

    for fold_ind, (base_fold, other_fold) in enumerate(zip(base_model.folds, other_model.folds)):
        for img_ind, (gt, base_pred, other_pred) in enumerate(
                zip(base_fold.groundtruth, base_fold.predictions, other_fold.predictions)):
            if gt != base_pred:
                if gt == other_pred:
                    predictions.better.append((fold_ind, img_ind))
                else:
                    predictions.both_failed.append((fold_ind, img_ind))
            elif gt != other_pred:
                predictions.worse.append((fold_ind, img_ind))
    return predictions


def union_predictions(predictions):
    union = edict({'better': list(predictions[0].better),
                   'worse': list(predictions[0].worse),
                   'both_failed': list(predictions[0].both_failed)})

    def union_lists(l1, l2):
        l = list()
        for t1 in l1:
            for t2 in l2:
                if t1[0] == t2[0] and t1[1] == t2[1]:
                    l.append(t1)
        return l

    num_prediction_sets = len(predictions)
    for i in range(1,num_prediction_sets):
        union.better = union_lists(union.better, predictions[i].better)
        union.worse = union_lists(union.worse, predictions[i].worse)
        union.both_failed = union_lists(union.both_failed, predictions[i].both_failed)
    return union


def show_top_scores(categories, base_models, ensemble_models, predictions, show_image=True):

    import matplotlib.pyplot as plt
    from IPython.display import display

    if not isinstance(base_models, list):
        base_models = [base_models]

    if not isinstance(ensemble_models, list):
        ensemble_models = [ensemble_models]

    if not isinstance(predictions, list):
        predictions = [predictions]

    for fold_ind, img_ind in predictions:

        if show_image:
            groundtruth = base_models[0].folds[fold_ind].groundtruth[img_ind]
            label = categories[groundtruth]
            img_path = base_models[0].folds[fold_ind].img_filepaths[img_ind]
            img = io.imread(img_path)

            fig = plt.figure(figsize=(3, 3))
            plt.clf()
            ax = fig.add_subplot(111)
            ax.set_aspect(1)
            plt.title(label)
            plt.imshow(np.asarray(img))
            plt.yticks([], [])
            plt.xticks([], [])
            plt.show()

        for base_model, ensemble_model in zip(base_models, ensemble_models):
            base_fold = base_model.folds[fold_ind]
            top_label_indices = base_fold.probabilities[img_ind].argsort()[::-1]
            scores = base_fold.probabilities[img_ind][top_label_indices]

            ensemble_prediction = ensemble_model.folds[fold_ind].predictions[img_ind]

            top_table = TopClassificationTable(categories, top_label_indices, scores, ensemble_prediction,
                                               base_model.name)
            display(top_table)
