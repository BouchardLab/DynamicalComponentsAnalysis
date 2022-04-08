import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .. import style


def make_axes(fig_width, wpad_edge=0, wpad_mid=0.05, hpad_top=0.05, hpad_bottom=0.05,
              small_sq_width=0.07):
    sq_width = (1. - 2 * wpad_edge - small_sq_width - 4 * wpad_mid) / 4.
    sq_height = 1. - hpad_top - hpad_bottom
    fig_height = sq_width * fig_width / sq_height
    small_sq_height = small_sq_width * fig_width / fig_height
    fig = plt.figure(figsize=(fig_width, fig_height))
    # 2 small squares
    ax2 = fig.add_axes((wpad_edge, hpad_bottom, small_sq_width, small_sq_height))
    ax1 = fig.add_axes((wpad_edge, 1. - hpad_top - small_sq_height,
                        small_sq_width, small_sq_height))
    # 3 big squares
    ax3 = fig.add_axes((wpad_edge + small_sq_width + wpad_mid, hpad_bottom, sq_width, sq_height))
    ax4 = fig.add_axes((wpad_edge + small_sq_width + 2 * wpad_mid + sq_width, hpad_bottom,
                        sq_width, sq_height))
    ax5 = fig.add_axes((wpad_edge + small_sq_width + 3 * wpad_mid + 2 * sq_width, hpad_bottom,
                        sq_width, sq_height))
    ax6 = fig.add_axes((wpad_edge + small_sq_width + 4 * wpad_mid + 3 * sq_width, hpad_bottom,
                        sq_width, sq_height))
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    label_dx = -0.02
    label_dy = 0.05
    label_y = hpad_bottom + sq_height + label_dy
    fig.text(wpad_edge + label_dx, label_y,
             "A", va="bottom", ha="right", color="black",
             **style.panel_letter_fontstyle)
    fig.text(wpad_edge + small_sq_width + wpad_mid, label_y,
             "B", va="bottom", ha="center", color="black",
             **style.panel_letter_fontstyle)
    fig.text(wpad_edge + small_sq_width + 2 * wpad_mid + sq_width, label_y,
             "C", va="bottom", ha="center", color="black",
             **style.panel_letter_fontstyle)
    fig.text(wpad_edge + small_sq_width + 3 * wpad_mid + 2 * sq_width, label_y,
             "D", va="bottom", ha="center", color="black",
             **style.panel_letter_fontstyle)
    fig.text(wpad_edge + small_sq_width + 4 * wpad_mid + 3 * sq_width, label_y,
             "E", va="bottom", ha="center", color="black",
             **style.panel_letter_fontstyle)
    return axes


def disp_heatmap(ax, heatmap, show_xlabels=True, show_ylabels=True, title=None):
    N_theta, N_phi = heatmap.shape
    ax.imshow(heatmap, origin="lower left", cmap="gray", aspect="equal")
    if show_xlabels:
        ax.set_xlabel("$\phi$", fontsize=style.axis_label_fontsize, labelpad=-8.5)
        ax.set_xticks([0, N_phi - 1])
        ax.set_xticklabels(["0", "$\pi$"], fontsize=style.ticklabel_fontsize)
    else:
        ax.set_xticks([])
    if show_ylabels:
        ax.set_ylabel("$\\theta$", fontsize=style.axis_label_fontsize, labelpad=-8.5)
        ax.set_yticks([0, N_theta - 1])
        ax.set_yticklabels(["0", "$\pi$"], fontsize=style.ticklabel_fontsize)
    else:
        ax.set_yticks([])
    ax.set_xlim([0, heatmap.shape[1] - 1])
    ax.set_ylim([0, heatmap.shape[0] - 1])
    if title is not None:
        ax.set_title(title, fontsize=style.axis_label_fontsize * 0.8, pad=1)


def disp_scatter(ax, pi_gp, pi_knn, trajectories=None, diag_text=False,
                 arrow=True, xlabel="full PI", ylabel="Gaussian PI"):
    # Note that gp=y and knn=x, but 0 index is gp and 1 is knn in data arrays!
    traj_color = "#C63F3A"
    all_gp_vals = [pi_gp]
    all_knn_vals = [pi_knn]
    if trajectories is not None:
        all_gp_vals += [traj[:, 0] for traj in trajectories]
        all_knn_vals += [traj[:, 1] for traj in trajectories]
    all_gp_vals = np.concatenate(all_gp_vals)
    all_knn_vals = np.concatenate(all_knn_vals)
    min_gp, max_gp = all_gp_vals.min(), all_gp_vals.max()
    range_gp = max_gp - min_gp
    min_knn, max_knn = all_knn_vals.min(), all_knn_vals.max()
    range_knn = max_knn - min_knn
    pi_gp_norm = (pi_gp - min_gp) / range_gp
    pi_knn_norm = (pi_knn - min_knn) / range_knn
    ax.hexbin(pi_knn_norm, pi_gp_norm, gridsize=50, extent=(0, 1, 0, 1),
              cmap="gray_r", bins="log", linewidth=0.05)
    if trajectories is not None:
        for traj_idx in range(len(trajectories)):
            traj = np.copy(trajectories[traj_idx])
            traj[:, 0] = (traj[:, 0] - min_gp) / range_gp
            traj[:, 1] = (traj[:, 1] - min_knn) / range_knn
            ax.plot(traj[:, 1], traj[:, 0], linewidth=0.5, color=traj_color)
    ax.set_xlim([0, 1.025])
    ax.set_ylim([0, 1.025])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([0, 1], fontsize=style.ticklabel_fontsize)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([0, 1], fontsize=style.ticklabel_fontsize)
    ax.spines['left'].set_bounds(0, 1)
    ax.spines['bottom'].set_bounds(0, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=style.axis_label_fontsize, labelpad=-9.5)
    ax.set_ylabel(ylabel, fontsize=style.axis_label_fontsize, labelpad=-8)

    theta_deg = 47
    if diag_text:
        ax.text(0.5, 0.65, "DCA trajectories", fontsize=style.ticklabel_fontsize * 0.8,
                rotation=theta_deg, rotation_mode="anchor", ha="center", va="center",
                color=traj_color)
    if arrow:
        len_x = np.cos(np.deg2rad(theta_deg))
        len_y = np.sin(np.deg2rad(theta_deg))
        mag = 0.425
        ax.quiver(0.475, 0.40, mag * len_x, mag * len_y,
                  angles='xy', scale_units='xy', scale=1, width=0.015,
                  color=traj_color)


def plot_deflation_results(ax, pi_regular, pi_def, pi_fft):
    dim_vals = np.arange(len(pi_def) + 1)
    pi_vals = (pi_regular, pi_def, pi_fft)
    labels = ["DCA", "deflation", "FFT deflation"]
    markersize = 1.5
    colors = ["#C63F3A", "gray", "black"]
    for i in range(len(pi_vals)):
        if i < 2:
            ax.plot(dim_vals, [0] + list(pi_vals[i]), label=labels[i],
                    linewidth=0.85, color=colors[i], linestyle="-")
        else:
            ax.plot(dim_vals, [0] + list(pi_vals[i]), label=labels[i],
                    linewidth=0, marker=".", markersize=markersize,
                    color=colors[i])

    ax.legend(fontsize=style.ticklabel_fontsize * 0.8, frameon=False,
              labelspacing=0.1, bbox_to_anchor=(0.2, 0, 1, 1))
    ax.set_xlabel("dimension", fontsize=style.axis_label_fontsize, labelpad=-9.5)
    ax.set_ylabel("PI (nats)", fontsize=style.axis_label_fontsize, labelpad=-13)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    max_dim = len(pi_regular)
    max_dim_padded = max_dim * 1.025
    ax.set_xticks([0, max_dim])
    ax.set_xticklabels([0, max_dim], fontsize=style.ticklabel_fontsize)
    ax.set_xlim([0, max_dim_padded])
    ax.spines["bottom"].set_bounds(0, max_dim)

    max_pi = np.max(np.concatenate(pi_vals))
    max_pi_padded = max_pi * 1.025
    ax.set_yticks([0, max_pi])
    ax.set_yticklabels([0, np.round(max_pi, 1)], fontsize=style.ticklabel_fontsize)
    ax.set_ylim([0, max_pi_padded])
    ax.spines["left"].set_bounds(0, max_pi)
