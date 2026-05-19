from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import load_recording, retrieve_parity_marks

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
    p.add_argument(
        "--slice",
        default="2600000:2650000",
        help="sample range start:stop"
    )
    args = p.parse_args()

    data, fs = load_recording(args.path)
    print(f"loaded {args.path.name}; columns: {list(data.columns)}")

    channels = ["pressure", "flow", "pmus"]

    parts = args.slice.split(":")
    start = int(parts[0])
    stop = int(parts[1])
    print(f"plotting samples [{start}:{stop}]")

    data_slice = data.iloc[start:stop]
    t = data_slice["time"].to_numpy() - data_slice["time"].iloc[0]
    ins_marks, exp_marks = retrieve_parity_marks(data_slice["volume"].to_numpy() * 10)
    t_ins = ins_marks / fs
    t_exp = exp_marks / fs
    print(f"marks in slice: {ins_marks.size} ins, {exp_marks.size} exp")

    fig, axes = plt.subplots(len(channels), 1, sharex=True)
    for ax, name in zip(axes, channels):
        y = data_slice[name].to_numpy()
        label = name if name == "pmus" else None
        ax.plot(t, y, "k", linewidth=1, label=label)
        ax.plot(t_ins, y[ins_marks], "^", color="tab:green", markersize=6)
        ax.plot(t_exp, y[exp_marks], "v", color="tab:red", markersize=6)
        ax.set_ylabel(name)
        ax.grid(True)
    axes[-1].plot(t, data_slice["pmus_mag"].to_numpy(), color="tab:blue", label="pmus_mag")
    axes[-1].legend(loc="upper right", fontsize=10)
    axes[0].plot([], [], "^", color="tab:green", label="ins")
    axes[0].plot([], [], "v", color="tab:red", label="exp")
    axes[0].legend(loc="upper right", fontsize=10)
    axes[-1].set_xlabel("time [s]") # relative to slice start
    fig.suptitle(f"{args.path.name}  [{start}:{stop}]")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
