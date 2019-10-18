import numpy as np
import matplotlib.pyplot as plt
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


def scatter_r2_vals(r2_vals, T_pi_idx, dim_vals, offset_vals, T_pi_vals,
                    min_val=None, max_val=None,
                    legend_both_cols=True, timestep=1, timestep_units="",
                    ax=None, xlabel=True, ylabel=True, title=None, legend=True,
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
    t = np.linspace(min_val, max_val, 100)
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
        ax.text(.6, .5, 'dim',
                transform=ax.transAxes, fontsize=style.ticklabel_fontsize)
        ax.text(.85, .5, 'lag',
                transform=ax.transAxes, fontsize=style.ticklabel_fontsize)

    # add labels/titles
    if xlabel:
        ax.set_xlabel(pca_label + " $R^2$", fontsize=style.axis_label_fontsize,
                      labelpad=-8)
    if ylabel:
        ax.set_ylabel("DCA $R^2$", fontsize=style.axis_label_fontsize,
                      labelpad=-8)
    if title is not None:
        ax.set_title(title, fontsize=style.title_fontsize)


def plot_pi_vs_T(r2_vals, T_pi_vals, dim_vals, offset_vals, offset_idx=0, min_max_val=None,
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
    ax.set_yticks([-min_max_val / 2., min_max_val])
    ax.set_yticklabels([-min_max_val / 2., min_max_val], fontsize=style.ticklabel_fontsize)
    ax.text(.4, .1, 'lag = {} bins'.format(offset_vals[offset_idx]),
            transform=ax.transAxes, fontsize=style.ticklabel_fontsize)

    # set ticks
    x_vals = T_pi_vals
    x_ticks = x_vals[1::2]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks.astype(np.int), fontsize=style.ticklabel_fontsize)

    # plot zero line
    ax.axhline(0, c="black", linestyle="--", zorder=0)

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
        ax.set_ylabel("$\Delta R^2$ improvement\nover SFA",
                      fontsize=style.axis_label_fontsize, labelpad=-8)
