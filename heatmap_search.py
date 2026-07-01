from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from pmus_miqp import pmus_miqp_fixed
from plotting import plot_cycle
from utils import (
    Cycle,
    extract_single_cycle,
    get_ins_exp_marks,
    load_recording,
)

DEFAULT_PATH = Path(__file__).parent / "data" / "ASL_spont_01.npz"
DEFAULT_CYCLE = 345
PEEP = 5.0
OFFSET_SECONDS = 0.3
TAU_SECONDS = 0.5

def time_to_samples(seconds: float, fs: float) -> int:
    return round(seconds * fs)


def cycle_fs(cycle: Cycle) -> float:
    return 1.0 / (cycle.time[1] - cycle.time[0])


def evaluate(cycle: Cycle, R: float, C_ext: float) -> float:
    R_ml = R / 1000.0
    E = 1.0 / C_ext
    tau_soe = time_to_samples(TAU_SECONDS, cycle_fs(cycle))
    try:
        # threads=1: one core per solve, joblib drives the outer parallelism
        pmus_hat, _, _ = pmus_miqp_fixed(
            cycle, R_ml, E, l2_reg=True, threads=1, tau_soe=tau_soe,
        )
        flow_ml_s = cycle.flow * 1000.0 / 60.0
        residual = (
            cycle.pressure - pmus_hat
            - R_ml * flow_ml_s - E * cycle.volume
        )
        return float(np.linalg.norm(residual))
    except Exception:
        return float("nan")


def load_cycle(path: Path, cycle_idx: int) -> Cycle:
    data, fs = load_recording(path)
    ins_marks, exp_marks = get_ins_exp_marks(path, data, fs)
    return extract_single_cycle(
        df=data,
        ins_mark=int(ins_marks[cycle_idx]),
        next_ins_mark=int(ins_marks[cycle_idx + 1]),
        exp_mark=int(exp_marks[cycle_idx]),
        peep=PEEP, offset=time_to_samples(OFFSET_SECONDS, fs),
    )


def lse_true(cycle: Cycle) -> tuple[float, float]:
    flow_ml_s = cycle.flow * 1000.0 / 60.0
    A = np.column_stack([flow_ml_s, cycle.volume])
    (R, E), *_ = np.linalg.lstsq(A, cycle.pressure - cycle.pmus, rcond=None)
    return float(R * 1000.0), float(1.0 / E)


def run_grid(
    cycle: Cycle, R_values: np.ndarray, C_values: np.ndarray, jobs: int,
) -> np.ndarray:
    nR, nC = len(R_values), len(C_values)
    print(f"grid: {nR} x {nC} = {nR * nC} solves, jobs={jobs}, n={cycle.pressure.size}")

    t0 = time.perf_counter()
    costs = Parallel(n_jobs=jobs, verbose=10)(
        delayed(evaluate)(cycle, float(R), float(C))
        for C in C_values
        for R in R_values
    )
    print(f"grid search done in {time.perf_counter() - t0:.1f} s")
    return np.array(costs).reshape(nC, nR)


def plot_surface(
    cost_matrix: np.ndarray,
    R_values: np.ndarray, C_values: np.ndarray,
    R_true: float, C_true: float,
    R_best: float, C_best: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    extent = [R_values[0], R_values[-1], C_values[0], C_values[-1]]
    im = ax.imshow(
        np.log10(cost_matrix),
        origin="lower", aspect="auto", extent=extent, cmap="turbo",
    )
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ residual cost")
    ax.set_xlabel(r"Resistance R [(cmH$_2$O$\cdot$s)/L]")
    ax.set_ylabel(r"Compliance C [mL/cmH$_2$O]")
    ax.set_title("MIQP residual cost surface")
    ax.plot(R_true, C_true, "r*", markersize=10, mec="w", label="LSE true")
    ax.plot(R_best, C_best, "go", markersize=8, mec="w", label="surface minimum")
    ax.legend(loc="upper right")
    ax.format_coord = lambda x, y:    f"  (R, C) = ({x:.2f}, {y:.2f})"
    im.format_cursor_data = lambda v: f"  cost J = [{10**v:.3f}]         "
    fig.tight_layout()


def plot_best(cycle: Cycle, R_best: float, C_best: float) -> tuple[plt.Figure, np.ndarray]:
    tau_soe = time_to_samples(TAU_SECONDS, cycle_fs(cycle))
    R_ml_best, E_best = R_best / 1000.0, 1.0 / C_best
    pmus_best, _, _ = pmus_miqp_fixed(cycle, R_ml_best, E_best, l2_reg=True, tau_soe=tau_soe)

    R_lse, C_lse = lse_true(cycle)
    R_ml_lse, E_lse = R_lse / 1000.0, 1.0 / C_lse
    pmus_lse, _, _ = pmus_miqp_fixed(cycle, R_ml_lse, E_lse, l2_reg=True, tau_soe=tau_soe)

    fig, axes = plot_cycle(
        cycle,
        estimates={
            "pmus_MIQP": (pmus_best, R_ml_best, E_best),
            f"pmus_MIQP-LSE_(R={R_lse:.1f}, C={C_lse:.1f})": (pmus_lse, R_ml_lse, E_lse),
        },
    )

    # paw reconstructed from the real pmus and LSE R, C parameters
    # best paw_est achievable (independent of MIQP)
    time = cycle.time - cycle.time[0]
    flow_ml_s = cycle.flow * 1000.0 / 60.0
    paw_est_lse_true = cycle.pmus + R_ml_lse * flow_ml_s + E_lse * cycle.volume
    axes[0].plot(
        time, paw_est_lse_true, "tab:purple", linestyle="--",
        label="paw_est (pmus_true @ LSE (R, C))",
    )
    axes[0].legend(loc="upper right", fontsize=9)

    # reference pmus_mag waveform on the pmus panel, if available
    if not np.all(np.isnan(cycle.pmus_mag)):
        axes[2].plot(time, cycle.pmus_mag, color="tab:green", label="pmus_mag_AI")
        axes[2].legend(loc="lower right", fontsize=9)

    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path", type=Path, nargs="?", default=DEFAULT_PATH,
        help=f"npz file (default: {DEFAULT_PATH.name})",
    )
    parser.add_argument(
        "--cycle", type=int, default=DEFAULT_CYCLE,
        help=f"cycle index (default: {DEFAULT_CYCLE})",
    )
    parser.add_argument(
        "--dim", type=int, default=20,
        help="(R, C) grid dim per axis"
    )
    parser.add_argument(
        "--jobs", type=int, default=8,
        help="parallel workers (each spawns its own Gurobi env)"
    )
    parser.add_argument(
        "--load", type=Path, default=None,
        help="path to a saved heatmap .npz; skips the grid search and just plots"
    )
    parser.add_argument(
        "--no-plot", dest="plot", action="store_false",
        help="run the grid search and save results without showing any plots"
    )
    args = parser.parse_args()

    if args.load is not None:
        npz = np.load(args.load, allow_pickle=True)
        plot_surface(
            npz["cost_matrix"], npz["R_values"], npz["C_values"],
            float(npz["R_true"]), float(npz["C_true"]),
            float(npz["R_best"]), float(npz["C_best"]),
        )
        if "cycle" in npz.files:
            R_best, C_best = float(npz["R_best"]), float(npz["C_best"])
            fig, _ = plot_best(npz["cycle"].item(), R_best, C_best)
            fig.suptitle(
                f"cycle #{int(npz['cycle_idx'])}: R = {R_best:.2f}, "
                f"C = {C_best:.2f}, J = {float(npz['cost_best']):.2f}"
            )
            fig.tight_layout()
        else:
            print("no cycle stored in this dataset; skipping pmus best plot")
        plt.show()
        return

    cycle = load_cycle(args.path, args.cycle)

    R_true, C_true = lse_true(cycle)
    print(f"PEEP: {PEEP}")
    print(f"LSE-true: R = {R_true:.2f}, C = {C_true:.2f}")

    # show the raw cycle to be analyzed before starting the grid search
    if args.plot:
        plot_cycle(cycle, title=f"cycle #{args.cycle} (to be analyzed)")
        plt.show(block=False)
        plt.pause(0.1)

    R_values = np.linspace( 5.0, 50.0, args.dim) # (cmH2O.s)/L
    C_values = np.linspace(10.0, 80.0, args.dim) # mL/cmH2O
    cost_matrix = run_grid(cycle, R_values, C_values, args.jobs)

    if np.isnan(cost_matrix).all():
        raise RuntimeError("grid solves failed, no usable cost matrix")

    C_best_idx, R_best_idx = np.unravel_index(np.nanargmin(cost_matrix), cost_matrix.shape)
    R_best = float(R_values[R_best_idx])
    C_best = float(C_values[C_best_idx])
    cost_best = float(cost_matrix[C_best_idx, R_best_idx])
    print(f"best grid: R = {R_best:.2f}, C = {C_best:.2f}, cost = {cost_best:.4f}")

    filename = f"heatmap_miqp_{args.dim}x{args.dim}_{args.path.stem}_idx_{args.cycle}.npz"
    out_path = Path(__file__).parent / filename
    np.savez(
        out_path,
        cost_matrix=cost_matrix,
        R_values=R_values,
        C_values=C_values,
        R_true=R_true, C_true=C_true,
        R_best=R_best, C_best=C_best, cost_best=cost_best,
        cycle_idx=args.cycle, peep=PEEP, offset_seconds=OFFSET_SECONDS,
        cycle=np.array(cycle, dtype=object),
    )
    print(f"saved results to {out_path.name}")

    if args.plot:
        plot_surface(cost_matrix, R_values, C_values, R_true, C_true, R_best, C_best)
        fig, _ = plot_best(cycle, R_best, C_best)
        fig.suptitle(f"cycle #{args.cycle}: R = {R_best:.2f}, C = {C_best:.2f}, J = {cost_best:.2f}")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
