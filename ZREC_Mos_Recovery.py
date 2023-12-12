import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_subject_inconsistency_bias(subject_inconsistency, subject_bias, aspect_ratio=(10, 4)):
    cp = ['#00bfc7', '#e8871a', '#514bd3', '#cc2481']
    nb_subjects = subject_inconsistency.shape[0]
    fig, ax = plt.subplots(2, 1, figsize=aspect_ratio, dpi=200)
    ax[0].scatter(np.arange(0, nb_subjects, 1), subject_inconsistency, c=cp[0])
    ax[1].scatter(np.arange(0, nb_subjects, 1), subject_bias, c=cp[0])
    ax[1].hlines(0, -0.5, nb_subjects-0.5, color='k', linestyles='--', linewidth=2)

    ax[1].set_xlabel('Subject ID')
    ax[0].set_ylabel('Subject Inconsistency'), ax[1].set_ylabel('Subject Bias')
    ax[0].set_xlim(-0.5, nb_subjects-0.5), ax[1].set_xlim(-0.5, nb_subjects-0.5)
    ax[0].set_xticks(np.arange(0, nb_subjects, 1)), ax[1].set_xticks(np.arange(0, nb_subjects, 1))
    ax[1].set_ylim(-1.1*np.absolute(subject_bias).max(), 1.1*np.absolute(subject_bias).max())
    ax[0].grid(), ax[1].grid()

    for l, label in enumerate(ax[0].xaxis.get_ticklabels()):
        if l%5!=0:
            label.set_visible(False)
    for l, label in enumerate(ax[1].xaxis.get_ticklabels()):
        if l%5!=0:
            label.set_visible(False)
    return fig


def plot_mos_ci(zrecmos, zrecmos_95ci, aspect_ratio=(12, 4)):
    cp = ['#00bfc7', '#e8871a', '#514bd3', '#cc2481']
    nb_stimuli = zrecmos.shape[0]
    zrecmos_95ci = np.repeat(zrecmos_95ci[:, np.newaxis], 2, axis=1)
    fig, ax = plt.subplots(figsize=aspect_ratio, dpi=200)
    ax.scatter(np.arange(0, nb_stimuli, 1), zrecmos,
               c=cp[0], s=15, marker='_')
    ax.errorbar(np.arange(0, nb_stimuli, 1), zrecmos, yerr=zrecmos_95ci.T,
                c=cp[0], ls='', capsize=3)

    ax.set_xlabel('Stimulus ID')
    ax.set_ylabel('Opinion Scores')
    ax.set_xlim(-1, nb_stimuli)
    ax.set_xticks(np.arange(0, nb_stimuli, 1))
    ax.grid()
    for l, label in enumerate(ax.xaxis.get_ticklabels()):
        if l%5!=0:
            label.set_visible(False)
    return fig


def weighted_avg_std(values, weights):
    """
    Function to calculate weighted average and weighted std
    :param values: values is an opinion score array with the shape [nb_obs, stimuli]
    :param weights: weighting factor for the average and std.
                    It is expected to be the uncertainty of individual observers.
                    The expected shape is [nb_obs]
    :return: average: weighted average as the recovered mean opinion scores
             std: weighted std as the std of the recovered mean opinion scores
    """
    average = np.ma.average(values, weights=weights, axis=0)
    variance = np.ma.average((values - average) ** 2, weights=weights, axis=0)
    std = np.sqrt(variance)
    return average, std


def zrec_mos_recovery(opinion_scores, stimuli_src_indices=''):
    """
    Function to calculate recovered MOS, corresponding CIs, subject bias, subject inconsistency and content ambiguity.
    :param opinion_scores (numpy.array): opinion score array with the shape [nb_subjects x nb_stimuli]
           if certain opinion scores are missing, they are expected to be represented as np.nan. Any other value might lead undesired recovery.
    :param stimuli_src_indices (numpy.array): input source content indices for each stimulus with the shape [nb_stimuli]
           if no SRC indices is given, content ambiguity will reflect the stimuli ambiguity.
           This has no effect on MOS recovery. If you are not interested in getting content ambiguity statistics, you can skip this step.
    :return: mos_recovered (numpy.array): Recovered MOS as unbiased, inconsistency weighted average of the opinion scores (with the shape [nb_stimuli])
    :return: mos_recoverd_ci_95 (numpy.array): 95% one-sided confidence intervals of the recovered MOS (with the shape [nb_stimuli])
    :return: subject_inconsistency (numpy.array): Inconsistency of the subjects in the opinion_scores array (with the shape [nb_subjects])
    :return: subject_bias_factor (numpy.array): Bias of the subjects in the opinion_scores array (with the shape [nb_subjects])
    :return: content_ambiguity (numpy.array): content ambiguity (if SRC indices is given with the shape [nb_SRC],
                                                                if not with the shape [nb_stimuli])
    """

    # Get trivial information from input opinion_scores
    nb_subjects = opinion_scores.shape[0]
    nb_stimuli = opinion_scores.shape[1]
    nb_valid_subjects = np.sum(~np.isnan(opinion_scores), axis=0)

    # Convert opinion scores to float64
    opinion_scores = opinion_scores.astype(np.float64)

    # get initial mos and its std for further calculations
    mos = np.nanmean(opinion_scores, axis=0)
    mos_stds = np.nanstd(opinion_scores, axis=0)

    # calculate zscores
    zscores = np.zeros_like(opinion_scores)
    for o in range(nb_subjects):
        zscores[o, :] = (opinion_scores[o, :] - mos) / mos_stds

    # calculate subject bias and subject inconsistency
    subject_bias_factor = np.nanmean(zscores, axis=1)
    subject_inconsistency = np.nanstd(zscores, axis=1)

    # get stimulus ambiguity aware bias per subject per stimulus
    subject_bias = np.zeros_like(opinion_scores)
    for o in range(nb_subjects):
        subject_bias[o, :] = mos_stds * subject_bias_factor[o]

    # get unbiased opinion scores
    opinion_scores_unbiased = opinion_scores - subject_bias
    # create a np.nan mask for the unbiased opinion scores
    opinion_scores_unbiased_masked = np.ma.MaskedArray(opinion_scores_unbiased, mask=np.isnan(opinion_scores_unbiased))
    # call the provided weighted_avg_std function to get the recovered MOS and std
    mos_recovered, mos_recovered_std = weighted_avg_std(opinion_scores_unbiased_masked, subject_inconsistency**-2)

    # get the 95% CI from the std values
    mos_recoverd_ci_95 = 1.96 * mos_recovered_std / np.sqrt(nb_valid_subjects)

    # if no src_indices is provided, we calculate the content ambiguity as the stimuli ambiguity
    if stimuli_src_indices == '':
        content_ambiguity = mos_stds

    # if src_indices is provided, we calculate the content ambiguity as the mean stimuli ambiguity for each SRC
    else:
        content_ambiguity = []
        # loop over each SRC
        for c in np.unique(stimuli_src_indices):
            # get the stimuli indices for the current SRC
            index = np.argwhere(stimuli_src_indices == c)
            # calculate the content ambiguity as the mean of the corresponding stimuli ambiguity
            c_content_ambiguity = np.mean(mos_stds[index])
            content_ambiguity.append(c_content_ambiguity)
        # wrap to a numpy array
        content_ambiguity = np.array(content_ambiguity)

    return mos_recovered, mos_recoverd_ci_95, subject_bias_factor, subject_inconsistency, content_ambiguity


def zrec_percentile_recovery(opinion_scores, percentile=25):
    """
    :param opinion_scores (numpy.array): opinion score array with the shape [nb_subjects x nb_stimuli]
           if certain opinion scores are missing, they are expected to be represented as np.nan. Any other value might lead undesired recovery.
    :param percentile (int or float): Percentile value to recover (in range [0, 100]). e.g. for SUR prediction 25th percentile is the common value.

    :return: percentile_values(numpy.array): percentile-th opinion scores per stimuli. Calculated as a weighted percentile over unbiased opinions. (with the shape [nb_stimuli])
    :return: subject_inconsistency (numpy.array): Inconsistency of the subjects in the opinion_scores array (with the shape [nb_subjects])
    :return: subject_bias_factor (numpy.array): Bias of the subjects in the opinion_scores array (with the shape [nb_subjects])
    """

    # Get trivial information from input opinion_scores
    nb_subjects = opinion_scores.shape[0]
    nb_stimuli = opinion_scores.shape[1]
    # Convert opinion scores to float64
    opinion_scores = opinion_scores.astype(np.float64)

    # get initial mos and its std for further calculations
    mos = np.nanmean(opinion_scores, axis=0)
    mos_stds = np.nanstd(opinion_scores, axis=0)

    # calculate zscores
    zscores = np.zeros_like(opinion_scores)
    for o in range(nb_subjects):
        zscores[o, :] = (opinion_scores[o, :] - mos) / mos_stds

    # calculate subject bias and subject inconsistency
    subject_bias_factor = np.nanmean(zscores, axis=1)
    subject_inconsistency = np.nanstd(zscores, axis=1)

    # get stimulus ambiguity aware bias per subject per stimulus
    subject_bias = np.zeros_like(opinion_scores)
    for o in range(nb_subjects):
        subject_bias[o, :] = np.nanstd(opinion_scores, axis=0) * subject_bias_factor[o]
    opinion_scores_unbiased = opinion_scores - subject_bias

    # set the weights for weighted percentile calculation
    weights = subject_inconsistency ** -2
    # initialize a zero numpy array to store percentile values per stimulus
    percentile_values = np.zeros(nb_stimuli)

    for snum in range(nb_stimuli):
        # calculate total weight of the subjects
        total_weight = np.sum(weights[~np.isnan(opinion_scores_unbiased[:, snum])])
        # calculate the weight threshold that corresponds to the input percentile value
        weight_percentile = total_weight*percentile/100

        # get the indices of the subjects when sorted according to ascending opinion values
        sorted_idx = np.argsort(opinion_scores_unbiased[:, snum])
        # sort the subjects' opinion scores and corresponding weights with the sorted indices
        sorted_opinions = opinion_scores_unbiased[sorted_idx, snum]
        sorted_weights = weights[sorted_idx]

        # initialize the parameters to track in the while loop
        curr_weight = 0
        counter = 0
        # loop over the sorted opinion scores until the weight threshold is reached
        while curr_weight < weight_percentile:
            curr_weight += sorted_weights[counter]
            percentile_values[snum] = sorted_opinions[counter]
            counter += 1

    return percentile_values, subject_inconsistency, subject_bias_factor

# Load dataset as subjects are arranged on the rows and stimuli are on the columns
# Conver it to a numpy array
HD_VJND_scores = pd.read_csv('./data/NETFLIX.csv', header=None).to_numpy()

# Call MOS recovery function
zrecmos, zrecmos_95ci, subject_inconsistency, subject_bias, content_ambiguity = zrec_mos_recovery(HD_VJND_scores)
# or call percentile recovery function (depending on the use-case)
percentiles, _, _ = zrec_percentile_recovery(HD_VJND_scores, 25)

# call "plot_subject_inconsistency_bias" plotting function to visualize the subject bias and inconsistency
subject_inconsistency_bias_figure = plot_subject_inconsistency_bias(subject_inconsistency, subject_bias)
subject_inconsistency_bias_figure.savefig('./figs/NETFLIX_subject_inconsistency_bias.png')

# call "plot_mos_ci" plotting function to visualize the subject bias and inconsistency
zrec_mos_ci_figure = plot_mos_ci(zrecmos, zrecmos_95ci)
zrec_mos_ci_figure.savefig('./figs/NETFLIX_zrec_mos_ci.png')