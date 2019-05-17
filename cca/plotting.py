import numpy as np
import matplotlib.pyplot as plt
dim_colors = ["red", "coral", "gray", "black"]

def decoding_fix_axes(fig_width=10, fig_height=5, wpad_edge=0, wpad_mid=.1, hpad_edge=0, hpad_mid=.1):
    fig = plt.figure(figsize=(fig_width, fig_height))
    sq_width = (1 - 2*wpad_edge - 2*wpad_mid)/3
    sq_height = sq_width * fig_width/fig_height

    #top row
    ax1 = fig.add_axes((wpad_edge, hpad_edge + sq_height + hpad_mid, sq_width, sq_height))
    ax2 = fig.add_axes((wpad_edge + sq_width + wpad_mid, hpad_edge + sq_height + hpad_mid, sq_width, sq_height))
    ax3 = fig.add_axes((wpad_edge + 2*sq_width + 2*wpad_mid, hpad_edge + sq_height + hpad_mid, sq_width, sq_height))
    ax4 = fig.add_axes((wpad_edge + 3*sq_width + 3*wpad_mid, hpad_edge + sq_height + hpad_mid, sq_width, sq_height))

    #bottom row
    ax5 = fig.add_axes((wpad_edge, hpad_edge, sq_width, sq_height))
    ax6 = fig.add_axes((wpad_edge + sq_width + wpad_mid, hpad_edge, sq_width, sq_height))
    ax7 = fig.add_axes((wpad_edge + 2*sq_width + 2*wpad_mid, hpad_edge, sq_width, sq_height))
    ax8 = fig.add_axes((wpad_edge + 3*sq_width + 3*wpad_mid, hpad_edge, sq_width, sq_height))

    axes = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8)
    return fig, axes

def scatter_r2_vals(r2_vals, T_pi_idx, dim_vals, offset_vals, min_val=None, max_val=None,
                    legend_both_cols=True, timestep=1, timestep_units="", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    #calculate means across CV folds
    vals_mean = np.mean(r2_vals, axis=0)
    dca_mean = vals_mean[:, :, T_pi_idx + 2]
    pca_mean = vals_mean[:, :, 0]

    #set plot bounds
    if min_val is None:
        min_val = np.min(np.concatenate(( dca_mean, pca_mean )))
    if max_val is None:
        max_val = np.max(np.concatenate(( dca_mean, pca_mean )))
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])

    #set ticks
    ax.set_xticks([min_val, max_val])
    ax.set_xticklabels([min_val, max_val], fontsize=12)
    ax.set_yticks([min_val, max_val])
    ax.set_yticklabels([min_val, max_val], fontsize=12)

    #plot diagonal line
    t = np.linspace(min_val, max_val, 100)
    ax.plot(t, t, c="black", linestyle="--", zorder=0)

    #make scatter
    markers = ['x', '+', 'v', 's']
    for dim_idx in range(len(dim_vals)):
        for offset_idx in range(len(offset_vals)):
            x, y = pca_mean[dim_idx, offset_idx], dca_mean[dim_idx, offset_idx]
            ax.scatter(x, y, c=[dim_colors[dim_idx]], marker=markers[offset_idx], s=60)

    #make legend
    #only plot dim vals if we're supposed to
    if legend_both_cols:
        for dim_idx in range(len(dim_vals)):
            dim_str = "dim: " + str(dim_vals[dim_idx])
            ax.scatter(-1, -1, c=[dim_colors[dim_idx]], marker="o", label=dim_str, s=50)
        ncol = 2
    else:
        ncol = 1
    #always plot offset (lag) vals
    for offset_idx in range(len(offset_vals)):
        lag_str = "lag: " + str(offset_vals[offset_idx] * timestep) + " " + timestep_units
        ax.scatter(-1, -1, c="black", marker=markers[offset_idx], label=lag_str, s=50)
    ax.legend(frameon=True, ncol=ncol, columnspacing=0.5,
              handletextpad=0, fontsize=9,
              loc="lower right", fancybox=True)

    #add labels/titles
    ax.set_xlabel("PCA $R^2$", fontsize=16, labelpad=-5)
    ax.set_ylabel("DCA $R^2$", fontsize=16, labelpad=0)

def plot_pi_vs_T(r2_vals, T_pi_vals, dim_vals, offset_idx=0, min_max_val=None,
                 legend=True, timestep=1, timestep_units="", ax=None):

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    #calculate mean improvement across CV folds
    sfa_mean = r2_vals[:, :, offset_idx, 1]
    dca_mean = r2_vals[:, :, offset_idx, 2:]
    improvement_mean = np.mean(dca_mean - sfa_mean[..., np.newaxis], axis=0)

    #set plot bounds
    if min_max_val is None:
        min_max_val = np.max(np.abs(improvement_mean))
    ax.set_ylim([-min_max_val, min_max_val])

    #set ticks
    ax.set_yticks([-min_max_val, min_max_val])
    x_vals = T_pi_vals * timestep
    x_ticks = x_vals[1::2]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks.astype(np.int), fontsize=12)

    #plot zero line
    ax.axhline(0, c="black", linestyle="--", zorder=0)

    #plot data
    for dim_idx in range(len(dim_vals)):
        dim_str = "dim: " + str(dim_vals[dim_idx])
        ax.plot(x_vals, improvement_mean[dim_idx],
                color=dim_colors[dim_idx],
                marker=".", markersize=10,
                label=dim_str)

    #make legend
    if legend:
        ax.legend(frameon=True, fontsize=9, loc="lower right", fancybox=True)

    #add labels/titles
    ax.set_xlabel("$T_{PI}$ (" + timestep_units + ")", fontsize=16, labelpad=0)
    ax.set_ylabel(r"$\Delta$ $R^2$ improvement over SFA", fontsize=16, labelpad=0)
