from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmus_miqp import pmus_miqp_full
from utils import (
    Cycle,
    extract_single_cycle,
    load_recording,
    retrieve_flow_marks,
    retrieve_parity_marks,
)

DEFAULT_PATH = Path(__file__).parent / "data" / "ASL_spont_01.npz"


# parity marks trick only works on ASL_spont_* recordings
def get_marks(
    path: Path, data: pd.DataFrame, fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    if path.name.startswith("ASL_spont_"):
        return retrieve_parity_marks(data["volume"].to_numpy() * 10)
    return retrieve_flow_marks(data["flow"].to_numpy(), fs)


def select_cycles(
    path: Path,
    data: pd.DataFrame,
    fs: float,
    slice_arg: str | None,
    cycles_arg: str | None,
    offset: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    ins_marks, exp_marks = get_marks(path, data, fs)
    n_cycles = min(len(ins_marks) - 1, len(exp_marks))

    if cycles_arg is not None:
        k_start, k_stop = (int(v) for v in cycles_arg.split(":"))
        cycle_indices = list(range(k_start, min(k_stop, n_cycles)))
        print(f"cycles [{k_start}:{k_stop}] selected: {len(cycle_indices)} cycles")
    else:
        start, stop = (int(v) for v in slice_arg.split(":"))
        cycle_indices = []
        for k in range(n_cycles):
            cycle_start = int(ins_marks[k]) - offset
            cycle_stop = int(ins_marks[k + 1]) - offset
            if cycle_start >= start and cycle_stop <= stop:
                cycle_indices.append(k)
        print(f"slice [{start}:{stop}] contains {len(cycle_indices)} full cycles")

    if not cycle_indices:
        raise SystemExit("no cycles selected, check range")
    return ins_marks, exp_marks, cycle_indices


def solve_cycle(cycle: Cycle, k: int, tau: int) -> dict:
    pmus_hat, R_hat, E_hat, _, solver_time = pmus_miqp_full(cycle, tau_soe=tau)
    flow_ml_s = cycle.flow * 1000.0 / 60.0
    paw_est = pmus_hat + R_hat * flow_ml_s + E_hat * cycle.volume
    residual = float(np.linalg.norm(cycle.pressure - paw_est))
    print(
        f"cycle #{k:2d}: R = {R_hat*1000:.2f}, C = {1/E_hat:.2f}, "
        f"J = {residual:.3f} ({solver_time:.1f}s)"
    )
    return {
        "time":      cycle.time,
        "pressure":  cycle.pressure,
        "paw_est":   paw_est,
        "flow":      cycle.flow,
        "pmus":      cycle.pmus,
        "pmus_miqp": pmus_hat,
    }


def plot_segments(segments: list[dict], title: str) -> None:
    cat = lambda key: np.concatenate([s[key] for s in segments])
    t = cat("time") - segments[0]["time"][0]
    cycle_boundaries = np.cumsum([s["pressure"].size for s in segments])

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 7))
    axes[0].plot(t, cat("pressure"), "k", label="paw")
    axes[0].plot(t, cat("paw_est"), "tab:orange", label="paw_est (MIQP)")
    axes[0].set_ylabel("paw [cmH2O]"); axes[0].grid(True)
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].plot(t, cat("flow"), "k")
    axes[1].set_ylabel("flow [L/min]"); axes[1].grid(True)

    axes[2].plot(t, cat("pmus"), "k", label="pmus_ASL")
    axes[2].plot(t, cat("pmus_miqp"), "tab:orange", label="pmus_miqp")
    axes[2].set_ylabel("pmus [cmH2O]"); axes[2].grid(True)
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right", fontsize=9)

    for ax in axes:
        for b in cycle_boundaries[:-1]:
            ax.axvline(t[b], color="tab:gray", linestyle=":", linewidth=0.8)

    fig.suptitle(title)
    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, nargs="?", default=DEFAULT_PATH,
        help=f"npz file (default: {DEFAULT_PATH.name})",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--slice", default=None,
        help="sample range, cycles fully contained in it are solved",
    )
    selector.add_argument(
        "--cycles", default=None,
        help="cycles index range, e.g. 9:11 picks cycles #9 and #10",
    )
    parser.add_argument("--peep", type=float, default=5.0)
    parser.add_argument("--offset", type=int, default=50)
    parser.add_argument("--tau", type=int, default=50)
    args = parser.parse_args()

    if args.slice is None and args.cycles is None:
        args.slice = "2600000:2601500"  # ASL_spont default slice

    data, fs = load_recording(args.path)
    print(f"loaded {args.path.name}; columns: {list(data.columns)}")

    ins_marks, exp_marks, cycle_indices = select_cycles(
        args.path, data, fs,
        args.slice, args.cycles, args.offset,
    )

    t0 = time.perf_counter()
    segments = []
    for k in cycle_indices:
        cycle = extract_single_cycle(
            df=data, fs=fs,
            ins_mark=int(ins_marks[k]),
            next_ins_mark=int(ins_marks[k + 1]),
            exp_mark=int(exp_marks[k]),
            peep=args.peep, offset=args.offset,
        )
        segments.append(solve_cycle(cycle, k, args.tau))
    print(f"all cycles done in {time.perf_counter() - t0:.1f} s")

    plot_segments(
        segments,
        title=f"{args.path.name} cycles #{cycle_indices[0]}..{cycle_indices[-1]}",
    )
    plt.show()


if __name__ == "__main__":
    main()
