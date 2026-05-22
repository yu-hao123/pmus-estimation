from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from pmus_miqp import pmus_miqp_fixed
from utils import (
    Cycle,
    extract_single_cycle,
    load_recording,
    retrieve_parity_marks,
)

DATA_PATH = Path(__file__).parent / "data" / "ASL_spont_01.npz"
CYCLE_IDX = 345
PEEP = 5.0
OFFSET = 30

def evaluate(cycle: Cycle, R_ext: float, C_ext: float) -> float:
    R = R_ext / 1000.0
    E = 1.0 / C_ext
    try:
        # threads=1: one core per solve, joblib drives the outer parallelism
        pmus_hat, _, _ = pmus_miqp_fixed(cycle, R, E, l2_reg=True, threads=1, tau_soe=100)
        flow_ml_s = cycle.flow * 1000.0 / 60.0
        residual = (
            cycle.pressure - pmus_hat
            - R * flow_ml_s - E * cycle.volume
        )
        return float(np.linalg.norm(residual))
    except Exception:
        return float("nan")


def load_cycle() -> Cycle:
    data, fs = load_recording(DATA_PATH)
    ins_marks, exp_marks = retrieve_parity_marks(data["volume"].to_numpy() * 10)
    return extract_single_cycle(
        df=data, fs=fs,
        ins_mark=int(ins_marks[CYCLE_IDX]),
        next_ins_mark=int(ins_marks[CYCLE_IDX + 1]),
        exp_mark=int(exp_marks[CYCLE_IDX]),
        peep=PEEP, offset=OFFSET,
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
    print(f"grid: {nR} x {nC} = {nR * nC} solves, jobs={jobs}")

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
    best_R: float, best_C: float,
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
    ax.plot(best_R, best_C, "go", markersize=8, mec="w", label="surface minimum")
    ax.legend(loc="upper right")
    ax.format_coord = lambda x, y:    f"  (R, C) = ({x:.2f}, {y:.2f})"
    im.format_cursor_data = lambda v: f"  cost J = [{10**v:.3f}]         "
    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    args = parser.parse_args()

    if args.load is not None:
        npz = np.load(args.load)
        plot_surface(
            npz["cost_matrix"], npz["R_values"], npz["C_values"],
            float(npz["R_true"]), float(npz["C_true"]),
            float(npz["best_R"]), float(npz["best_C"]),
        )
        plt.show()
        return

    cycle = load_cycle()

    R_true, C_true = lse_true(cycle)
    print(f"LSE-true: R = {R_true:.2f}, C = {C_true:.2f}")

    R_values = np.linspace( 5.0, 50.0, args.dim) # (cmH2O.s)/L
    C_values = np.linspace(10.0, 80.0, args.dim) # mL/cmH2O
    cost_matrix = run_grid(cycle, R_values, C_values, args.jobs)

    if np.isnan(cost_matrix).all():
        raise RuntimeError("grid solves failed, no usable cost matrix")

    best_C_idx, best_R_idx = np.unravel_index(np.nanargmin(cost_matrix), cost_matrix.shape)
    best_R = float(R_values[best_R_idx])
    best_C = float(C_values[best_C_idx])
    best_cost = float(cost_matrix[best_C_idx, best_R_idx])
    print(f"best grid: R = {best_R:.2f}, C = {best_C:.2f}, cost = {best_cost:.4f}")

    filename = f"heatmap_miqp_{args.dim}x{args.dim}_idx_{CYCLE_IDX}.npz"
    out_path = Path(__file__).parent / filename
    np.savez(
        out_path,
        cost_matrix=cost_matrix,
        R_values=R_values,
        C_values=C_values,
        R_true=R_true, C_true=C_true,
        best_R=best_R, best_C=best_C, best_cost=best_cost,
        cycle_idx=CYCLE_IDX, peep=PEEP, offset=OFFSET,
    )
    print(f"saved results to {out_path.name}")

    plot_surface(cost_matrix, R_values, C_values, R_true, C_true, best_R, best_C)
    plt.show()


if __name__ == "__main__":
    main()
