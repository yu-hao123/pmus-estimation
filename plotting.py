from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from utils import Cycle

def plot_cycle(
    cycle: Cycle,
    estimates: dict[str, tuple[np.ndarray, float, float]] | None = None,
    *,
    switches: np.ndarray | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a single cycle's waveforms (paw | flow | pmus) in three stacked panels.

    estimates:  label -> (pmus_hat, resistance, elastance).
    switches:   pmus binary drawn as vertical lines across all panels.

    Returns (fig, axes) so callers can add further decorations
    """
    time = cycle.time - cycle.time[0]
    flow_ml_s = cycle.flow * 1000.0 / 60.0

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))

    axes[0].plot(time, cycle.pressure, "k", label="paw")
    axes[0].set_ylabel("paw [cmH2O]"); axes[0].grid(True)

    axes[1].plot(time, cycle.flow, "k")
    axes[1].set_ylabel("flow [L/min]"); axes[1].grid(True)

    axes[2].plot(time, cycle.pmus, "k", label="pmus_true")
    axes[2].set_ylabel("pmus [cmH2O]"); axes[2].grid(True)
    axes[2].set_xlabel("time [s]")

    for index, (label, (pmus_hat, resistance, elastance)) in enumerate((estimates or {}).items()):
        paw_est = pmus_hat + resistance * flow_ml_s + elastance * cycle.volume
        line, = axes[2].plot(time, pmus_hat, label=label)
        axes[0].plot(time, paw_est, color=line.get_color(), label=f"paw_est ({label.split("_")[1]})")

    ins_mark = int(np.where(cycle.insexp >= 0.5)[0][0])
    exp_mark = int(np.where(np.diff(cycle.insexp) <= -0.5)[0][0]) + 1
    for ax, channel in zip(axes, [cycle.pressure, cycle.flow, cycle.pmus]):
        ax.plot(time[ins_mark], channel[ins_mark], "^", color="tab:green", markersize=8)
        ax.plot(time[exp_mark], channel[exp_mark], "v", color="tab:red", markersize=8)
    axes[1].plot([], [], "^", color="tab:green", label="ins mark")
    axes[1].plot([], [], "v", color="tab:red", label="exp mark")

    if switches is not None:
        for ax in axes:
            for switch in switches:
                ax.axvline(time[int(switch)], color="tab:red", linestyle="--", linewidth=1.0)
        axes[1].plot([], [], color="tab:red", linestyle="--", label="binary switches")

    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].legend(loc="lower right", fontsize=9)
    axes[2].legend(loc="lower right", fontsize=9)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
