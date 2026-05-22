from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import get_ins_exp_marks, load_recording

DEFAULT_PATH = Path(__file__).parent / "data" / "ASL_spont_01.npz"

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=DEFAULT_PATH,
        help=f"npz file to plot (default: {DEFAULT_PATH.name})"
    )
    selector = p.add_mutually_exclusive_group()
    selector.add_argument(
        "--slice",
        default=None,
        help="sample range start:stop",
    )
    selector.add_argument(
        "--cycles",
        default=None,
        help="ins-mark index range start:stop, e.g. 345:350",
    )
    args = p.parse_args()

    data, fs = load_recording(args.path)
    print(
        f"\nloaded {args.path.name};\n"
        f"columns: {list(data.columns)}\n"
        f"dataset size: {data.size} \n"
        )

    channels = ["pressure", "flow", "pmus"]

    ins_marks, exp_marks = get_ins_exp_marks(args.path, data, fs)

    if args.cycles is not None:
        k_start, k_stop = (int(v) for v in args.cycles.split(":"))
        start = int(ins_marks[k_start] - 10)
        stop = int(ins_marks[k_stop]) if k_stop < ins_marks.size else len(data)
    else:
        slice_arg = args.slice if args.slice is not None else "2600000:2650000"
        start, stop = (int(v) for v in slice_arg.split(":"))

    is_ins_inside = (ins_marks >= start) & (ins_marks < stop)
    is_exp_inside = (exp_marks >= start) & (exp_marks < stop)
    ins_cycle_indices = np.flatnonzero(is_ins_inside)
    exp_cycle_indices = np.flatnonzero(is_exp_inside)
    ins_marks_rebased = ins_marks[is_ins_inside] - start
    exp_marks_rebased = exp_marks[is_exp_inside] - start

    data_slice = data.iloc[start:stop]
    t = data_slice["time"].to_numpy() - data_slice["time"].iloc[0]
    t_ins = ins_marks_rebased / fs
    t_exp = exp_marks_rebased / fs

    ins_range = f"{ins_cycle_indices[0]}..{ins_cycle_indices[-1]}"
    exp_range = f"{exp_cycle_indices[0]}..{exp_cycle_indices[-1]}"
    print(
        f"marks in slice: {ins_marks_rebased.size} ins ({ins_range}), "
        f" {exp_marks_rebased.size} exp ({exp_range})"
    )

    fig, axes = plt.subplots(len(channels), 1, sharex=True)
    for ax, name in zip(axes, channels):
        y = data_slice[name].to_numpy()
        label = name if name == "pmus" else None
        ax.plot(t, y, "k", linewidth=1, label=label)
        ax.plot(t_ins, y[ins_marks_rebased], "^", color="tab:green", markersize=6)
        ax.plot(t_exp, y[exp_marks_rebased], "v", color="tab:red", markersize=6)
        ax.set_ylabel(name)
        ax.grid(True)
    #axes[-1].plot(t, data_slice["pmus_mag"].to_numpy(), color="tab:blue", label="pmus_mag")
    #axes[-1].legend(loc="upper right", fontsize=10)
    axes[0].plot([], [], "^", color="tab:green", label="ins")
    axes[0].plot([], [], "v", color="tab:red", label="exp")
    axes[0].legend(loc="upper right", fontsize=10)
    axes[-1].set_xlabel("time [s]") # relative to slice start
    fig.suptitle(f"{args.path.name}  [{start}:{stop}]")
    fig.tight_layout()

    # nearest-mark lookup, shared by hover and click handlers
    all_t = np.concatenate([t_ins, t_exp])
    all_idx = np.concatenate([ins_cycle_indices, exp_cycle_indices])
    all_sample = np.concatenate([ins_marks[is_ins_inside], exp_marks[is_exp_inside]])
    all_type = np.array(["ins"] * t_ins.size + ["exp"] * t_exp.size)

    def nearest(x):
        if all_t.size == 0:
            return None
        return int(np.argmin(np.abs(all_t - x)))

    def format_coord(x, y):
        k = nearest(x)
        if k is None:
            return f"(x, y) = ({x:.2f}, {y:.2f})"
        return (
            f"(x, y) = ({x:.2f}, {y:.2f}) | "
            f"nearest {all_type[k]} #{all_idx[k]} @ t = {all_t[k]:.2f}s"
        )
    for ax in axes:
        ax.format_coord = format_coord

    # click anywhere in an axes, log the nearest mark's global index
    def on_click(event):
        if event.inaxes is None or event.button != 1:
            return
        if fig.canvas.toolbar is not None and fig.canvas.toolbar.mode:
            return  # ignore clicks while pan/zoom tool is active
        k = nearest(event.xdata)
        if k is None:
            return
        print(
            f"{all_type[k]} #{int(all_idx[k])} at sample {int(all_sample[k])}, "
            f"t = {all_t[k]:.3f}s"
        )

    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()


if __name__ == "__main__":
    main()
