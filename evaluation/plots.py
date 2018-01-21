from __future__ import division

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def cm(cm, labels, figsize=None, annot=False, cmap='jet',
       ticks_size=None, linewidths=0, show_yticks=True, show_xticks=False, cbar=True):
    import seaborn as sns

    if not figsize:
        figsize = (5, 5)

    fig = plt.figure(figsize=figsize)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    if show_yticks:
        yticklabels = labels
    else:
        yticklabels = []

    if show_xticks:
        xticklabels = labels
    else:
        xticklabels = []

    ax = sns.heatmap(cm, annot=annot, cmap=cmap, linewidths=linewidths, xticklabels=xticklabels,
                     yticklabels=yticklabels, vmax=1., cbar=cbar)

    if show_yticks:
        for ticklabel in ax.get_yaxis().get_ticklabels():
            ticklabel.set_rotation('horizontal')
            if ticks_size:
                ticklabel.set_fontsize(ticks_size)

    if show_xticks:
        ax.xaxis.tick_top()
        for ticklabel in ax.get_xaxis().get_ticklabels():
            ticklabel.set_rotation('vertical')
            if ticks_size:
                ticklabel.set_fontsize(ticks_size)

    return fig, ax


def show_cms(models, labels, num_rows=1, num_cm_row=None, figsize=None, cmap=None, ticks_size=None):
    import seaborn as sns

    if not figsize:
        figsize = (5, 5)

    if not cmap:
        cmap = plt.cm.RdYlBu_r

    num_confusion_matrices = len(models)

    if not num_cm_row:
        num_cm_row = num_confusion_matrices

    fig, axn = plt.subplots(num_rows, num_cm_row, sharex=True, sharey=True, figsize=figsize, facecolor='white')
    cbar_ax = fig.add_axes([1., 0.22, 0.025, 0.375])
    fig.subplots_adjust(wspace=0.05, hspace=0.4, right=0.8)

    for i, ax in enumerate(axn.flat):
        if i >= num_confusion_matrices:
            continue
        normalized_cm = models[i].confusion_matrix

        sns.heatmap(normalized_cm, ax=ax,
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cmap=cmap,
                    yticklabels=labels, xticklabels=[],
                    cbar_ax=None if i else cbar_ax)
        if i == 0:
            ax.collections[0].colorbar.solids.set_rasterized(False)

        if ticks_size:
            for ticklabel in ax.get_yaxis().get_ticklabels():
                ticklabel.set_fontsize(ticks_size)

        ax.set_title(models[i].name, weight='bold', size='small')
        ax.set_aspect('equal', 'box-forced')
    fig.tight_layout(rect=[0, 0, 1, 0.8])
    fig.canvas
    return fig, axn


def plot_accuracy(values, labels=None, iters=None, epochs=None, figsize=None, xlabel=None, subplot=None):
    if not isinstance(values, list):
        values = [values]

    if iters is not None:
        x_value = iters
        if not xlabel:
            xlabel = u'Iteration'
    elif epochs is not None:
        x_value = epochs
        if not xlabel:
            xlabel = u'Epoch'
    else:
        x_value = np.arange(1, len(values[0]) + 1)
        if not xlabel:
            xlabel = u'Time'

    if not subplot:
        if not figsize:
            figsize = (1, 1)
        fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)
    else:
        fig, ax = subplot

    sns.set_style("whitegrid", {'axes.edgecolor': '.1', 'axes.linewidth': 0.8})
    sns.set_palette("muted")
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})

    colors = sns.color_palette()

    ax.grid(b=True, which='major', color='#555555', linestyle=':', linewidth=0.5)
    for i, v in enumerate(values):
        mean = v.mean(axis=0)
        std = v.std(axis=0)
        ax.fill_between(x_value, mean - std, mean + std, alpha=0.2, color=colors[i])
        if labels:
            ax.plot(x_value, mean, linewidth=1, linestyle='-', marker='o', markersize=3, label=labels[i],
                    color=colors[i])
        else:
            ax.plot(x_value, mean, linewidth=1, linestyle='-', marker='o', markersize=3, color=colors[i])

    plt.xlim(x_value[0], x_value[-1])
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 11, endpoint=True))

    plt.ylabel(u'Accuracy', fontweight='bold', fontsize=12)
    plt.xlabel(xlabel, fontweight='bold', fontsize=12)
    ax.legend(loc=4, fontsize=11, frameon=True)

    axes = plt.gca()
    axes.set_aspect(x_value[-1])
    return fig, ax
