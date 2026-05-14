from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

    data = np.load(args.path)
    print(f"loaded {args.path.name}; fields: {list(data.files)}")

    channels = ["pressure", "flow", "pmus"]
    fs = float(data["fs"])
    nsamples = data[channels[0]].size
    time = np.arange(nsamples) / fs

    parts = args.slice.split(":")
    start = int(parts[0])
    stop = int(parts[1])
    interval = slice(start, stop)
    t = time[interval] - time[interval][0]
    print(f"plotting samples [{start}:{stop}]")

    fig, axes = plt.subplots(len(channels), 1, sharex=True)
    for ax, name in zip(axes, channels):
        ax.plot(t, data[name][interval], "k", linewidth=0.8)
        ax.set_ylabel(name)
        ax.grid(True)
    axes[-1].set_xlabel("time [s]") # relative to slice start
    fig.suptitle(f"{args.path.name}  [{start}:{stop}]")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
