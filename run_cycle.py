import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from utils import load_recording, retrieve_parity_marks, extract_single_cycle

DEFAULT_PATH = Path(__file__).parent / "data" / "ASL_spont_01.npz"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, nargs="?", default=DEFAULT_PATH,
        help=f"npz file default: {DEFAULT_PATH.name}"
    )
    parser.add_argument(
        "--cycle", type=int, default=8098,
        help="respiratory cycle index"
    )
    peep = 5
    offset = 50 # considering fs = 50Hz
    args = parser.parse_args()

    data, fs = load_recording(args.path)
    print(f"loaded {args.path.name}; fs = {fs}Hz; {len(data)} samples")

    ins_marks, exp_marks = retrieve_parity_marks(data["volume"].to_numpy() * 10)
    print(f"found {ins_marks.size} ins marks, {exp_marks.size} exp marks")

    i = args.cycle
    cycle = extract_single_cycle(
        df = data, fs = fs,
        ins_mark=int(ins_marks[i]),
        next_ins_mark=int(ins_marks[i+1]),
        exp_mark=int(exp_marks[i]),
        peep=peep,
        offset=offset
    )

    t = cycle.time - cycle.time[0]
    t_exp = t[cycle.insexp == 0][0]

    channels = [
        ("paw", cycle.pressure),
        ("flow", cycle.flow),
        ("pmus", cycle.pmus),
    ]

    fig, axes = plt.subplots(len(channels), 1, sharex=True, figsize=(6, 6))
    for ax, (label, y) in zip(axes, channels):
        ax.plot(t, y, "k")
        ax.axvspan(t_exp, t[-1], color="tab:red", alpha=0.08)
        ax.set_ylabel(label)
        ax.grid(True)

    axes[-1].plot(t, cycle.pmus_mag, color="tab:blue", label="pmus_mag")
    axes[-1].plot([], [], "k", label="pmus")
    axes[-1].legend(loc="upper right", fontsize=10)
    axes[-1].set_xlabel("time [s]")

    fig.suptitle(f"{args.path.name} cycle {i}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()