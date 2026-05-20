import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils import Cycle

# quadratic optimization demo (minimal gurobi setup)
def pmus_qp_fixed_RE(
    cycle: Cycle, R: float, E: float, verbose: bool = False
) -> np.ndarray:
    flow_ml_s = cycle.flow * 1000.0 / 60.0 # converting from L/min to mL/s
    volume = cycle.volume
    pressure = cycle.pressure
    nsamples = pressure.size

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()
    model = gp.Model(env=env)
    model.Params.TimeLimit = 60.0

    pmus = model.addMVar(nsamples, lb=-25.0, ub=1.0, name="pmus")
    residual = pressure - (pmus + R * flow_ml_s + E * volume)
    model.setObjective(residual @ residual, GRB.MINIMIZE)
    model.optimize()

    pmus_hat = np.asarray(pmus.X).ravel()
    model.dispose()
    env.dispose()

    return pmus_hat


def pmus_qp_joint(cycle: Cycle, verbose: bool = False) -> tuple[np.ndarray, float, float]:
    flow_ml_s = cycle.flow * 1000.0 / 60.0

    volume = cycle.volume
    pressure = cycle.pressure
    nsamples = pressure.size

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()
    model = gp.Model(env=env)
    model.Params.TimeLimit = 60.0

    pmus = model.addMVar(nsamples, lb=-25.0, ub=1.0, name="pmus")
    RE = model.addMVar(2, lb=[0.0, 0.005], ub=[0.1, 1.0], name="RE")

    A = np.column_stack([flow_ml_s, volume]) # (N, 2)
    residual = pressure - pmus - A @ RE
    model.setObjective(residual @ residual, GRB.MINIMIZE)
    model.optimize()

    pmus_hat = np.asarray(pmus.X).ravel()
    R_hat, E_hat = float(RE.X[0]), float(RE.X[1])
    model.dispose()
    env.dispose()
    return pmus_hat, R_hat, E_hat


if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    from utils import load_recording, retrieve_parity_marks, extract_single_cycle

    data, fs = load_recording(Path(__file__).parent / "data" / "ASL_spont_01.npz")
    ins_marks, exp_marks = retrieve_parity_marks(data["volume"].to_numpy() * 10)
    cycle = extract_single_cycle(
        df=data, fs=fs,
        ins_mark=int(ins_marks[8098]), next_ins_mark=int(ins_marks[8099]),
        exp_mark=int(exp_marks[8098]),
        peep=5.0, offset=50,
    )

    # applying least squares in real pmus waveform
    flow_ml_s = cycle.flow / 60.0 * 1000.0
    A = np.column_stack([flow_ml_s, cycle.volume])
    b = cycle.pressure - cycle.pmus
    (R_lse, E_lse), *_ = np.linalg.lstsq(A, b, rcond=None)
    print(f"LSE true (external): R = {R_lse * 1000:.2f}, C= {1 / (E_lse):.2f}")

    pmus_hat = pmus_qp_fixed_RE(cycle, R_lse, E_lse, verbose=True)
    pmus_joint, R_joint, E_joint = pmus_qp_joint(cycle, verbose=True)

    t = cycle.time - cycle.time[0]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

    axes[0].plot(t, cycle.pressure, "k")
    axes[0].set_ylabel("paw [cmH2O]"); axes[0].grid(True)

    axes[1].plot(t, cycle.flow, "k")
    axes[1].set_ylabel("flow [L/min]"); axes[1].grid(True)

    axes[2].plot(t, cycle.pmus, "k", label="pmus_true")
    axes[2].plot(t, pmus_hat, "tab:red", label="pmus_hat (QP solver)")
    axes[2].plot(t, pmus_joint, "tab:green", label="pmus_joint (no switching)")
    axes[2].set_ylabel("pmus [cmH2O]"); axes[2].grid(True)
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="lower right", fontsize=10)

    fig.suptitle(f"Demo gurobi QP setup R={R_lse*1000:.2f}, C={1000/(E_lse*1000):.2f}")
    fig.tight_layout()
    plt.show()