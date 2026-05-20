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
    n = pressure.size

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()
    model = gp.Model(env=env)
    model.Params.TimeLimit = 60.0

    pmus = model.addMVar(n, lb=-25.0, ub=1.0, name="pmus")
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
    n = pressure.size

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()
    model = gp.Model(env=env)
    model.Params.TimeLimit = 60.0

    pmus = model.addMVar(n, lb=-25.0, ub=1.0, name="pmus")
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

def pmus_miqp_fixed(
    cycle: Cycle, R: float, E: float,
    tau_soe: int = 50, epsilon: float = 1e-3,
    l2_reg: bool = True, verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    flow_ml_s = cycle.flow * 1000.0 / 60.0
    volume = cycle.volume
    pressure = cycle.pressure
    n = pressure.size
    ns = 2  # no initial delay for now

    # k_soe: 0-based first expiratory sample (where insexp drops 1 → 0)
    diffs = np.diff(cycle.insexp)
    k_soe = int(np.where(diffs <= -0.5)[0][0]) + 1

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()
    model = gp.Model(env=env)
    model.Params.TimeLimit = 60.0

    pmus = model.addMVar(n, lb=-20.0, ub=1.0, name="pmus")
    tik = model.addMVar((n, ns), vtype=GRB.BINARY, name="tik")

    # uniqueness: each switch occurs exactly once
    for s in range(ns):
        model.addConstr(tik[:, s].sum() == 1, name=f"unique_s{s}")

    # ordering: switch indices are non-decreasing
    weights = np.arange(n)
    for s in range(1, ns):
        model.addConstr(weights @ tik[:, s - 1] <= weights @ tik[:, s], name=f"order_s{s}")

    # exhalation: last switch must lie at or before k_soe + tau_soe
    model.addConstr(weights @ tik[:, ns - 1] <= k_soe + tau_soe, name="exhalation")

    # cumulative sum aux binaries: c[i, s] = sum(tik[0:i+1, s])
    c = model.addMVar((n, ns), vtype=GRB.BINARY, name="c")
    for s in range(ns):
        model.addConstr(c[0, s] == tik[0, s], name=f"c_init_s{s}")
        for i in range(1, n):
            model.addConstr(c[i, s] - c[i - 1, s] == tik[i, s], name=f"c_step_{i}_s{s}")

    # region contraints: c[i, s] activates the per-region monotonic shape rule on pmus
    for i in range(n - 1):
        # region 1 (before switch 0): pmus decreasing
        # c[i, 0] == 0 then pmus[i+1] + epsilon <= pmus[i]
        model.addGenConstrIndicator(
            c[i, 0].item(), 0,
            pmus[i + 1] + epsilon <= pmus[i],
            name=f"reg1_{i}"
        )

        # region 2 (between switches): pmus increasing
        # need an aux binary b_mid = c[i, 0] - c[i, 1] because the activation
        # condition is a difference of binaries, not a single Var.
        b_mid = model.addVar(vtype=GRB.BINARY, name=f"b_mid_{i}")
        model.addConstr(b_mid == c[i, 0] - c[i, 1])
        model.addGenConstrIndicator(
            b_mid, 1,
            pmus[i] + epsilon <= pmus[i + 1],
            name=f"reg2_{i}"
        )

        # region 3 (after switch 1): pmus constant
        # c[i, 1] == 1 -> pmus[i] == pmus[i+1]
        model.addGenConstrIndicator(
            c[i, 1].item(), 1,
            pmus[i] == pmus[i + 1],
            name=f"reg3_{i}"
        )

    residual = pressure - pmus - R * flow_ml_s - E * volume
    cost = residual @ residual
    if l2_reg:
        cost = cost + 1e-3 * (pmus @ pmus)
    model.setObjective(cost, GRB.MINIMIZE)
    #model.write("debug_miqp.lp")
    model.optimize()

    pmus_hat = np.asarray(pmus.X).ravel()
    switching_times = (weights @ tik.X).astype(int)
    model.dispose()
    env.dispose()
    return pmus_hat, switching_times

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
        peep=5.0, offset=30,
    )

    # applying least squares in real pmus waveform
    flow_ml_s = cycle.flow / 60.0 * 1000.0
    A = np.column_stack([flow_ml_s, cycle.volume])
    b = cycle.pressure - cycle.pmus
    (R_lse, E_lse), *_ = np.linalg.lstsq(A, b, rcond=None)
    print(f"LSE true (external): R = {R_lse * 1000:.2f}, C= {1 / (E_lse):.2f}")

    pmus_hat = pmus_qp_fixed_RE(cycle, R_lse, E_lse)
    pmus_fixed_miqp, switches = pmus_miqp_fixed(cycle, R_lse, E_lse, verbose=True)
    cost_miqp = np.linalg.norm(
        cycle.pressure - pmus_fixed_miqp - R_lse * flow_ml_s - E_lse * cycle.volume
    )
    print(f"||paw - paw_hat|| = {cost_miqp:.4f}")

    t = cycle.time - cycle.time[0]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

    axes[0].plot(t, cycle.pressure, "k")
    axes[0].set_ylabel("paw [cmH2O]"); axes[0].grid(True)

    axes[1].plot(t, cycle.flow, "k")
    axes[1].set_ylabel("flow [L/min]"); axes[1].grid(True)

    axes[2].plot(t, cycle.pmus, "k", label="pmus_true")
    axes[2].plot(t, pmus_hat, "tab:red", label="pmus_hat (QP solver)")
    axes[2].plot(t, pmus_fixed_miqp, "tab:purple", label="pmus_miqp_fixed")
    axes[2].set_ylabel("pmus [cmH2O]"); axes[2].grid(True)
    axes[2].set_xlabel("time [s]")

    for ax in axes:
        for s in switches:
            ax.axvline(t[s], color="tab:orange", linestyle="--", linewidth=1.0)
    axes[2].plot([], [], color="tab:orange", linestyle="--",
                 linewidth=1.0, label="switches (MIQP)")
    axes[2].legend(loc="lower right", fontsize=10)

    fig.suptitle(f"Demo gurobi QP setup R={R_lse*1000:.2f}, C={1000/(E_lse*1000):.2f}")
    fig.tight_layout()
    plt.show()