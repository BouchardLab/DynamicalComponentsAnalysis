import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from . import style


dim_colors = ["red", "coral", "gray", "black"]


def decoding_fix_axes(fig_width=10, fig_height=5, wpad_left=0, wpad_right=0.,
                      wpad_mid=.1, hpad_bot=0, hpad_mid=.1):
    fig = plt.figure(figsize=(fig_width, fig_height))
    sq_width = (1 - wpad_left - wpad_right - 3 * wpad_mid) / 4
    sq_height = sq_width * fig_width / fig_height

    # top row
    ax1 = fig.add_axes((wpad_left, hpad_bot + sq_height + hpad_mid, sq_width, sq_height))
    ax2 = fig.add_axes((wpad_left + sq_width + wpad_mid, hpad_bot + sq_height + hpad_mid,
                        sq_width, sq_height))
    ax3 = fig.add_axes((wpad_left + 2 * sq_width + 2 * wpad_mid, hpad_bot + sq_height + hpad_mid,
                        sq_width, sq_height))
    ax4 = fig.add_axes((wpad_left + 3 * sq_width + 3 * wpad_mid, hpad_bot + sq_height + hpad_mid,
                        sq_width, sq_height))

    # bottom row
    ax5 = fig.add_axes((wpad_left, hpad_bot, sq_width, sq_height))
    ax6 = fig.add_axes((wpad_left + sq_width + wpad_mid, hpad_bot, sq_width, sq_height))
    ax7 = fig.add_axes((wpad_left + 2 * sq_width + 2 * wpad_mid, hpad_bot, sq_width, sq_height))
    ax8 = fig.add_axes((wpad_left + 3 * sq_width + 3 * wpad_mid, hpad_bot, sq_width, sq_height))

    axes = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8)
    return fig, axes


def decoding_fix_axes2(fig_width=10, fig_height=5, wpad_left=0, wpad_right=0., wpad_mid=.1,
                       hpad_bot=0, hpad_mid=.1):
    fig = plt.figure(figsize=(fig_width, fig_height))
    sq_width = (1 - wpad_left - wpad_right - wpad_mid) / 2
    sq_height = sq_width * fig_width / fig_height

    # top row
    ax1 = fig.add_axes((wpad_left, hpad_bot + sq_height + hpad_mid, sq_width, sq_height))
    ax2 = fig.add_axes((wpad_left + sq_width + wpad_mid, hpad_bot + sq_height + hpad_mid,
                        sq_width, sq_height))

    # bottom row
    ax3 = fig.add_axes((wpad_left, hpad_bot, sq_width, sq_height))
    ax4 = fig.add_axes((wpad_left + sq_width + wpad_mid, hpad_bot, sq_width, sq_height))

    axes = (ax1, ax2, ax3, ax4)
    return fig, axes


def scatter_r2_vals(r2_vals, T_pi_idx, dim_vals, offset_vals, T_pi_vals,
                    min_val=None, max_val=None,
                    legend_both_cols=True, timestep=1, timestep_units="",
                    ax=None, xlabel=True, ylabel=True, title=None, legend=True, legendtext=True,
                    bbox_to_anchor=None, loc=None, pca_label="PCA"):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    # calculate means across CV folds
    vals_mean = np.mean(r2_vals, axis=0)
    dca_mean = vals_mean[:, :, T_pi_idx + 2]
    pca_mean = vals_mean[:, :, 0]

    # set plot bounds
    if min_val is None:
        min_val = np.min(np.concatenate((dca_mean, pca_mean)))
    if max_val is None:
        max_val = np.max(np.concatenate((dca_mean, pca_mean)))
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])

    # set ticks
    ax.set_xticks([min_val, max_val])
    ax.set_xticklabels([min_val, max_val], fontsize=style.ticklabel_fontsize)
    ax.set_yticks([min_val, max_val])
    ax.set_yticklabels([min_val, max_val], fontsize=style.ticklabel_fontsize)

    # plot diagonal line
    t = [min_val, max_val]
    ax.plot(t, t, c="black", linestyle="--", zorder=0, linewidth=1.)
    ax.text(.05, .9, 'T = {} bins'.format(T_pi_vals[T_pi_idx]),
            transform=ax.transAxes, fontsize=style.ticklabel_fontsize)

    # make scatter
    markers = ['x', '+', 'v', 's']
    for dim_idx in range(len(dim_vals)):
        for offset_idx in range(len(offset_vals)):
            x, y = pca_mean[dim_idx, offset_idx], dca_mean[dim_idx, offset_idx]
            ax.scatter(x, y, c=[dim_colors[dim_idx]],
                       marker=markers[offset_idx], s=12)

    # make legend
    # only plot dim vals if we're supposed to
    if legend_both_cols:
        for dim_idx in range(len(dim_vals)):
            dim_str = "dim: " + str(dim_vals[dim_idx])
            dim_str = str(dim_vals[dim_idx])
            ax.scatter(-1, -1, c=[dim_colors[dim_idx]], marker="o",
                       label=dim_str, s=16)
        ncol = 2
    else:
        ncol = 1
    # always plot offset (lag) vals
    for offset_idx in range(len(offset_vals)):
        lag_str = "lag: " + str(offset_vals[offset_idx] * timestep) + " " + timestep_units
        lag_str = '{} {}'.format(str(offset_vals[offset_idx] * timestep), timestep_units)
        ax.scatter(-1, -1, c="black", marker=markers[offset_idx],
                   label=lag_str, s=16)
    if legend:
        ax.legend(ncol=ncol, columnspacing=0.5,
                  handletextpad=0, fontsize=style.ticklabel_fontsize - 1,
                  fancybox=True, markerscale=.8, frameon=True,
                  bbox_to_anchor=bbox_to_anchor, loc=loc, handlelength=1.25)
        if legendtext:
            ax.text(.6, .5, 'dim',
                    transform=ax.transAxes, fontsize=style.ticklabel_fontsize)
            ax.text(.85, .5, 'lag',
                    transform=ax.transAxes, fontsize=style.ticklabel_fontsize)

    # add labels/titles
    if xlabel:
        ax.set_xlabel(pca_label + " $R^2$", fontsize=style.axis_label_fontsize,
                      labelpad=-8)
    if ylabel:
        ax.set_ylabel("DCA $R^2$", fontsize=style.axis_label_fontsize, labelpad=-8)
    if title is not None:
        ax.set_title(title, fontsize=style.title_fontsize)


def plot_r2_vs_T(r2_vals, T_pi_vals, dim_vals, offset_vals, offset_idx=0, min_max_val=None,
                 legend=True, timestep=1, timestep_units="", ax=None,
                 xlabel=True, ylabel=True, bbox_to_anchor=None, loc=None):

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    # calculate mean improvement across CV folds
    sfa_mean = r2_vals[:, :, offset_idx, 1]
    dca_mean = r2_vals[:, :, offset_idx, 2:]
    improvement_mean = np.mean(dca_mean - sfa_mean[..., np.newaxis], axis=0)

    # set plot bounds
    if min_max_val is None:
        min_max_val = np.max(np.abs(improvement_mean))
    ax.set_ylim([-min_max_val / 2., min_max_val])
    ax.set_yticks([-min_max_val / 2., 0, min_max_val])
    ax.set_yticklabels([-min_max_val / 2., 0., min_max_val], fontsize=style.ticklabel_fontsize)
    ax.text(.1, .1, 'lag = {} bins'.format(offset_vals[offset_idx]),
            transform=ax.transAxes, fontsize=style.ticklabel_fontsize)

    # set ticks
    x_vals = T_pi_vals
    x_ticks = x_vals[1::2]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks.astype(np.int), fontsize=style.ticklabel_fontsize)

    # plot zero line
    ax.axhline(0, c="black", linestyle="-", zorder=0, lw=1)

    # plot data
    for dim_idx in range(len(dim_vals)):
        dim_str = "dim: " + str(dim_vals[dim_idx])
        ax.plot(x_vals, improvement_mean[dim_idx],
                color=dim_colors[dim_idx], linewidth=1.)
        ax.scatter(x_vals, improvement_mean[dim_idx],
                   color=dim_colors[dim_idx],
                   marker=".", s=16,
                   label=dim_str)

    # make legend
    if legend:
        ax.legend(frameon=True, fontsize=style.ticklabel_fontsize, fancybox=True,
                  bbox_to_anchor=bbox_to_anchor, loc=loc)

    # add labels/titles
    if xlabel:
        ax.set_xlabel(r"T ({} {} bins)".format(timestep, timestep_units),
                      fontsize=style.axis_label_fontsize, labelpad=0)
    if ylabel:
        ax.set_ylabel('$\Delta R^2$ improvement\nover SFA',
                      fontsize=style.axis_label_fontsize, labelpad=-8)


def plot_absolute_r2_vs_T(r2_vals, T_pi_vals, dim_vals, offset_vals, offset_idx=0, min_max_val=None,
                          legend=True, timestep=1, timestep_units="", ax=None,
                          xlabel=True, ylabel=True, bbox_to_anchor=None, loc=None,
                          dca=True):

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    # calculate mean improvement across CV folds
    dca_mean = np.mean(r2_vals[:, :, offset_idx, 2:], axis=0)
    sfa_mean = np.mean(r2_vals[:, :, offset_idx, 1][..., np.newaxis], axis=0)
    sfa_mean = np.tile(sfa_mean, (1, dca_mean.shape[-1]))
    if dca:
        name = 'DCA'
        mean = dca_mean
        print(mean.shape)
    else:
        name = 'SFA'
        mean = sfa_mean
        print(mean.shape)

    # set plot bounds
    if min_max_val is None:
        min_max_val = np.max(np.abs(mean))
    ax.set_ylim([-min_max_val / 2., min_max_val])
    if min_max_val > 1.:
        ax.set_yticks([-.5, 0, 1])
        ax.set_yticklabels([-.5, '0', 1], fontsize=style.ticklabel_fontsize)
    else:
        ax.set_yticks([-min_max_val / 2., 0, min_max_val])
        ax.set_yticklabels([-min_max_val / 2., '0', min_max_val], fontsize=style.ticklabel_fontsize)
    ax.text(.1, .1, 'lag = {} bins'.format(offset_vals[offset_idx]),
            transform=ax.transAxes, fontsize=style.ticklabel_fontsize)

    # set ticks
    x_vals = T_pi_vals
    x_ticks = x_vals[1::2]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks.astype(np.int), fontsize=style.ticklabel_fontsize)

    # plot zero line
    ax.axhline(0, c="black", linestyle="-", zorder=0, lw=1)

    # plot data
    for dim_idx in range(len(dim_vals)):
        dim_str = "dim: " + str(dim_vals[dim_idx])
        ax.plot(x_vals, mean[dim_idx],
                color=dim_colors[dim_idx], linewidth=1.)
        ax.scatter(x_vals, mean[dim_idx],
                   color=dim_colors[dim_idx],
                   marker=".", s=16,
                   label=dim_str)

    # make legend
    if legend:
        ax.legend(frameon=True, fontsize=style.ticklabel_fontsize, fancybox=True,
                  bbox_to_anchor=bbox_to_anchor, loc=loc)

    # add labels/titles
    if xlabel:
        ax.set_xlabel(r"T ({} {} bins)".format(timestep, timestep_units),
                      fontsize=style.axis_label_fontsize, labelpad=0)
    if ylabel:
        ax.set_ylabel("{} $R^2$".format(name), fontsize=style.axis_label_fontsize, labelpad=-8)


def make_comparison_axes(fig_width, fig_height, wpad_edge=0, wpad_mid=0, hpad_bottom=0, hpad_top=0,
                         inset_x_rel=0.5, inset_y_rel=0.5, inset_height_rel=0.25):
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax_height = 1. - hpad_top - hpad_bottom
    ax_width = (1. - 2 * wpad_edge - 2 * wpad_mid) / 3

    ax1 = fig.add_axes((wpad_edge, hpad_bottom, ax_width, ax_height))
    ax2 = fig.add_axes((wpad_edge + ax_width + wpad_mid, hpad_bottom, ax_width, ax_height))
    ax3 = fig.add_axes((wpad_edge + 2 * ax_width + 2 * wpad_mid, hpad_bottom, ax_width, ax_height))

    # Make inset
    inset_width_rel = 1.5 * inset_height_rel * (ax_height * fig_height) / (ax_width * fig_width)
    inset_x_abs = wpad_edge + ax_width * inset_x_rel
    inset_y_abs = hpad_bottom + ax_height * inset_y_rel
    inset_width_abs = ax_width * inset_width_rel
    inset_height_abs = ax_height * inset_height_rel

    ax_inset = fig.add_axes((inset_x_abs, inset_y_abs, inset_width_abs, inset_height_abs))

    fig.text(wpad_edge / 2, hpad_bottom + ax_height,
             "A", va="bottom", ha="right", color="black",
             **style.panel_letter_fontstyle)
    fig.text(wpad_edge + ax_width + wpad_mid - wpad_edge + wpad_edge / 2,
             hpad_bottom + ax_height,
             "B", va="bottom", ha="right", color="black",
             **style.panel_letter_fontstyle)
    fig.text(wpad_edge + 2 * ax_width + 2 * wpad_mid - wpad_edge + wpad_edge / 2,
             hpad_bottom + ax_height,
             "C", va="bottom", ha="right", color="black",
             **style.panel_letter_fontstyle)

    return ax1, ax_inset, ax2, ax3


def cycle_from_style(scheme):
    return [color_dict["color"] for color_dict in plt.style.library[scheme]['axes.prop_cycle']]


def plot_mi_vs_dim(mi_vals, labels, ax=None, legend=False, xlabel=False, max_dim=None, title=None):
    method_colors = ["black", "#e83535", "tab:blue", "gray"]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    num_methods, N = mi_vals.shape
    if max_dim is None:
        max_dim = N
    pi_from_zero = np.concatenate((np.zeros(num_methods).reshape((num_methods, 1)),
                                   mi_vals[:, :max_dim]), axis=1)
    dims = np.arange(max_dim + 1)
    max_mi = np.max(mi_vals)

    ax.set_ylim([0, max_mi * 1.05])
    ax.set_xlim([0, max_dim * 1.02])

    xticks = np.arange(0, max_dim + 1, 5, dtype=np.int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=style.ticklabel_fontsize)

    yticks = [0, max_mi]
    max_ylabel = str(np.round(max_mi, 1))
    while len(max_ylabel) < 4:
        max_ylabel = "  " + max_ylabel
    ytick_labels = ["0", max_ylabel]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=style.ticklabel_fontsize)
    ax.tick_params(axis='y', which='major', pad=1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(0, max_mi)
    ax.spines['bottom'].set_bounds(0, max_dim)

    for i in range(num_methods):
        linewidth = 1 if labels[i] == "CCA" else 0.75
        markersize = 3 if labels[i] == "CCA" else 2
        ax.plot(dims, pi_from_zero[i], label=labels[i], linewidth=linewidth,
                color=method_colors[i], marker=".", markersize=markersize)
    if legend:
        ax.legend(loc="upper left", fontsize=style.ticklabel_fontsize * 0.9,
                  frameon=False, ncol=2, labelspacing=0.1, columnspacing=0.55,
                  bbox_to_anchor=(-0.0, 0.025, 1, 1))
    if xlabel:
        ax.set_xlabel("dimensions retained", fontsize=style.axis_label_fontsize, labelpad=1)
    ax.set_ylabel("MI (nats)", fontsize=style.axis_label_fontsize,
                  labelpad=-style.axis_label_fontsize)
    if title is not None:
        ax.set_title(title, fontsize=style.axis_label_fontsize, pad=3)


def plot_dca_autocorr_fns(ax, ax_inset, autocorr_1, autocorr_2):
    linewidth = 1
    colors = ["tab:blue", "#e83535"]

    ax.plot(autocorr_1, linewidth=linewidth, c=colors[0], label="$T = 1$")
    ax.plot(autocorr_2, linewidth=linewidth, c=colors[1], label="$T = 20$")
    ax_inset.plot(autocorr_1[:3], linewidth=linewidth, c=colors[0])
    ax_inset.plot(autocorr_2[:3], linewidth=linewidth, c=colors[1])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax_inset.spines['right'].set_visible(False)
    ax_inset.spines['top'].set_visible(False)

    # main plot tick labels
    max_dt = 20
    xticks = np.arange(0, max_dt + 1, 5, dtype=np.int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=style.ticklabel_fontsize)
    ax.set_xlim([0, max_dt])
    y_min_main = np.min(np.concatenate((autocorr_1, autocorr_2)))
    y_min_main -= (1 - y_min_main) * .05
    yticks = [y_min_main, 0, 1]
    main_ytick_labels = [np.round(y_min_main, 1), "0.0", "1.0"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(main_ytick_labels, fontsize=style.ticklabel_fontsize)
    ax.set_ylim([y_min_main, 1])
    ax.axhline(0, linestyle="-", linewidth=linewidth, color="black", zorder=0)
    ax.yaxis.set_tick_params(pad=1)
    ax.set_ylabel("autocorrelation", fontsize=style.axis_label_fontsize, labelpad=1)
    ax.set_xlabel("$\Delta t$ (100 ms bins)", fontsize=style.axis_label_fontsize, labelpad=1)

    # inset tick labels
    xticks = [0, 1, 2]
    ax_inset.set_xticks(xticks)
    ax_inset.set_xticklabels(xticks, fontsize=style.ticklabel_fontsize * 0.75)
    ax_inset.spines['bottom'].set_bounds(0, 2)
    ax_inset.set_xlim([0, 2.065])
    y_min_inset = np.min(np.concatenate((autocorr_1[:3], autocorr_2[:3])))
    y_min_inset -= (1 - y_min_inset) * .05
    ax_inset.set_ylim([y_min_inset, 1])
    ax_inset.set_yticks([y_min_inset, 1])
    inset_ytick_labels = [np.round(y_min_inset, 1), "1.0"]
    ax_inset.set_yticklabels(inset_ytick_labels, fontsize=style.ticklabel_fontsize * 0.75)
    ax_inset.yaxis.set_tick_params(pad=1)
    ax_inset.xaxis.set_tick_params(pad=1)
    ax_inset.axvline(1, c="black", linestyle="--", linewidth=linewidth)

    rect = patches.Rectangle((0.25, y_min_inset - 0.01),
                             2, 1 - y_min_inset,
                             linestyle="-",
                             linewidth=0.9, edgecolor="black", facecolor="none",
                             zorder=100)
    ax.add_patch(rect)
    ax.quiver(2.5, 0.7, 2.7, 0.1, angles='xy', scale_units='xy', scale=1, width=0.01, color="black")
    ax_inset.set_facecolor([0, 0, 0, 0])
    ax.legend(loc="upper right", fontsize=style.ticklabel_fontsize * .9,
                  frameon=False, ncol=1, labelspacing=0.1, columnspacing=0.6,
                  bbox_to_anchor=(0.05, 0.175, 1, 1))
