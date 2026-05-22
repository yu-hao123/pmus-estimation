import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils import Cycle

# quadratic optimization demo (minimal gurobi setup)
def pmus_qp_fixed(
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


def define_constraints(
    model: gp.Model, insexp: np.ndarray,
    tau_soe: int, epsilon: float,
    initial_delay: bool = False,
    prefix: str = "",
) -> tuple[gp.MVar, gp.MVar]:
    # prefix is useful for distinct naming when building one model
    # with multiple regions (dual)
    n = insexp.size
    ns = 3 if initial_delay else 2  # initial_delay adds a pre-inspiratory flat-zero region

    # k_soe: 0-based first expiratory sample (where insexp drops 1 -> 0)
    # works whether insexp is padded or not,
    # np.diff finds the same transition either way
    # shifted by delay_length when padded
    diffs = np.diff(insexp)
    k_soe = int(np.where(diffs <= -0.5)[0][0]) + 1

    pmus = model.addMVar(n, lb=-20.0, ub=1.0, name=f"{prefix}pmus")
    tik = model.addMVar((n, ns), vtype=GRB.BINARY, name=f"{prefix}tik")

    # uniqueness: each switch occurs exactly once
    for s in range(ns):
        model.addConstr(tik[:, s].sum() == 1, name=f"{prefix}unique_s{s}")

    # ordering: switch indices are non-decreasing
    weights = np.arange(n)
    for s in range(1, ns):
        model.addConstr(weights @ tik[:, s - 1] <= weights @ tik[:, s], name=f"{prefix}order_s{s}")

    # exhalation: last switch must lie at or before k_soe + tau_soe
    model.addConstr(weights @ tik[:, ns - 1] <= k_soe + tau_soe, name=f"{prefix}exhalation")

    # cumulative sum aux binaries: c[i, s] = sum(tik[0:i+1, s])
    c = model.addMVar((n, ns), vtype=GRB.BINARY, name=f"{prefix}c")
    for s in range(ns):
        model.addConstr(c[0, s] == tik[0, s], name=f"{prefix}c_init_s{s}")
        for i in range(1, n):
            model.addConstr(c[i, s] - c[i - 1, s] == tik[i, s], name=f"{prefix}c_step_{i}_s{s}")

    # flat-zero regions, only when initial_delay
    if initial_delay:
        for i in range(n):
            # region 0 (delay, before switch 0): pmus == 0
            # c[i, 0] == 0 -> pmus[i] = 0
            model.addGenConstrIndicator(
                c[i, 0].item(), 0,
                pmus[i] == 0,
                name=f"{prefix}reg0_{i}"
            )
            # region 3 (exhalation delay, after switch 2): pmus == 0
            # c[i, 2] == 1 -> pmus[i] = 0
            model.addGenConstrIndicator(
                c[i, 2].item(), 1,
                pmus[i] == 0,
                name=f"{prefix}reg3_{i}"
            )

    # monotonicity regions: c[i, s] activates the shape rule on pmus
    for i in range(n - 1):
        if initial_delay:
            # region 1 (decreasing ramp, between switch 0 and 1)
            # need aux binary because activation is a difference of binaries
            b_dec = model.addVar(vtype=GRB.BINARY, name=f"{prefix}b_dec_{i}")
            model.addConstr(b_dec == c[i, 0] - c[i, 1])
            model.addGenConstrIndicator(
                b_dec, 1,
                pmus[i + 1] + epsilon <= pmus[i],
                name=f"{prefix}reg1_{i}"
            )

            # region 2 (increasing recovery, between switch 1 and 2)
            b_inc = model.addVar(vtype=GRB.BINARY, name=f"{prefix}b_inc_{i}")
            model.addConstr(b_inc == c[i, 1] - c[i, 2])
            model.addGenConstrIndicator(
                b_inc, 1,
                pmus[i] + epsilon <= pmus[i + 1],
                name=f"{prefix}reg2_{i}"
            )
        else:
            # region 1 (before switch 0): pmus decreasing
            # c[i, 0] == 0 then pmus[i+1] + epsilon <= pmus[i]
            model.addGenConstrIndicator(
                c[i, 0].item(), 0,
                pmus[i + 1] + epsilon <= pmus[i],
                name=f"{prefix}reg1_{i}"
            )

            # region 2 (between switches): pmus increasing
            # need an aux binary b_mid = c[i, 0] - c[i, 1] because the activation
            # condition is a difference of binaries, not a single Var.
            b_mid = model.addVar(vtype=GRB.BINARY, name=f"{prefix}b_mid_{i}")
            model.addConstr(b_mid == c[i, 0] - c[i, 1])
            model.addGenConstrIndicator(
                b_mid, 1,
                pmus[i] + epsilon <= pmus[i + 1],
                name=f"{prefix}reg2_{i}"
            )

            # region 3 (after switch 1): pmus constant
            # c[i, 1] == 1 -> pmus[i] == pmus[i+1]
            model.addGenConstrIndicator(
                c[i, 1].item(), 1,
                pmus[i] == pmus[i + 1],
                name=f"{prefix}reg3_{i}"
            )

    return pmus, tik


# R is expected in (cmH2O.s/mL)
# E is expected in cmH2O/mL
def pmus_miqp_fixed(
    cycle: Cycle, R: float, E: float,
    tau_soe: int = 50,
    epsilon: float = 1e-3,
    l2_reg: bool = True,
    initial_delay_length: int = 0,
    threads: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:

    flow_ml_s = cycle.flow * 1000.0 / 60.0
    volume = cycle.volume
    pressure = cycle.pressure
    insexp = cycle.insexp

    if initial_delay_length > 0:
        # prepend a flat-zero region to all waveforms
        pad = np.zeros(initial_delay_length)
        flow_ml_s = np.concatenate([pad, flow_ml_s])
        volume = np.concatenate([pad, volume])
        pressure = np.concatenate([pad, pressure])
        insexp = np.concatenate([pad, insexp])

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()
    model = gp.Model(env=env)
    model.Params.TimeLimit = 60.0
    model.Params.Threads = threads

    # gp.MVar, not numpy arrays
    pmus, tik = define_constraints(
        model, insexp, tau_soe, epsilon,
        initial_delay = (initial_delay_length > 0),
    )

    residual = pressure - pmus - R * flow_ml_s - E * volume
    cost = residual @ residual
    if l2_reg:
        cost = cost + 1e-3 * (pmus @ pmus)
    model.setObjective(cost, GRB.MINIMIZE)
    #model.write("debug_miqp.lp")
    model.optimize()

    pmus_hat = np.asarray(pmus.X).ravel()
    switching_times = (np.arange(pmus_hat.size) @ tik.X).astype(int)
    solver_time = model.Runtime
    model.dispose()
    env.dispose()
    return pmus_hat, switching_times, solver_time


def pmus_miqp_full(
    cycle: Cycle,
    tau_soe: int = 50,
    epsilon: float = 1e-3,
    l2_reg: bool = True,
    initial_delay_length: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:

    flow_ml_s = cycle.flow * 1000.0 / 60.0
    volume = cycle.volume
    pressure = cycle.pressure
    insexp = cycle.insexp

    if initial_delay_length > 0:
        # prepend a flat-zero region to all waveforms; the helper will
        # add the matching pre-inspiratory region constraint
        pad = np.zeros(initial_delay_length)
        flow_ml_s = np.concatenate([pad, flow_ml_s])
        volume = np.concatenate([pad, volume])
        pressure = np.concatenate([pad, pressure])
        insexp = np.concatenate([pad, insexp])

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()
    model = gp.Model(env=env)
    model.Params.TimeLimit = 60.0

    # gp.MVar, not numpy arrays
    pmus, tik = define_constraints(
        model, insexp, tau_soe, epsilon,
        initial_delay=(initial_delay_length > 0),
    )

    # R, E are MVar, jointly optimized with pmus and tik
    # ranges R: [0, 100], C: [1, 200]
    RE = model.addMVar(2, lb=[0.0, 0.005], ub=[0.1, 1.0], name="RE")

    # matrix form
    A = np.column_stack([flow_ml_s, volume])  # (n, 2)
    residual = pressure - pmus - A @ RE
    cost = residual @ residual
    if l2_reg:
        cost = cost + 1e-3 * (pmus @ pmus)
    model.setObjective(cost, GRB.MINIMIZE)
    model.optimize()

    pmus_hat = np.asarray(pmus.X).ravel()
    switching_times = (np.arange(pmus_hat.size) @ tik.X).astype(int)
    R_hat, E_hat = float(RE.X[0]), float(RE.X[1])
    solver_time = model.Runtime
    model.dispose()
    env.dispose()
    return pmus_hat, R_hat, E_hat, switching_times, solver_time


# joint solve over two cycles sharing one (R, E)
def pmus_miqp_dual(
    cycle_first: Cycle,
    cycle_second: Cycle,
    tau_soe: int = 50,
    epsilon: float = 1e-3,
    l2_reg: bool = True,
    initial_delay_length: int = 0,
    threads: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, float]:

    if initial_delay_length > 0:
        zeros = np.zeros(initial_delay_length)
        def padded(cycle: Cycle) -> Cycle:
            return Cycle(
                time=np.concatenate([zeros, cycle.time]),
                pressure=np.concatenate([zeros, cycle.pressure]),
                flow=np.concatenate([zeros, cycle.flow]),
                volume=np.concatenate([zeros, cycle.volume]),
                pmus=np.concatenate([zeros, cycle.pmus]),
                pmus_mag=np.concatenate([zeros, cycle.pmus_mag]),
                insexp=np.concatenate([zeros, cycle.insexp]),
            )
        cycle_first = padded(cycle_first)
        cycle_second = padded(cycle_second)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()
    model = gp.Model(env=env)
    model.Params.TimeLimit = 60.0
    model.Params.Threads = threads

    pmus_first, tik_first = define_constraints(
        model, cycle_first.insexp, tau_soe, epsilon,
        initial_delay=(initial_delay_length > 0),
        prefix="first_",
    )
    pmus_second, tik_second = define_constraints(
        model, cycle_second.insexp, tau_soe, epsilon,
        initial_delay=(initial_delay_length > 0),
        prefix="second_",
    )

    # shared mechanics across both cycles
    RE = model.addMVar(2, lb=[0.0, 0.005], ub=[0.1, 1.0], name="RE")

    A_first = np.column_stack([cycle_first.flow * 1000.0 / 60.0, cycle_first.volume])
    A_second = np.column_stack([cycle_second.flow * 1000.0 / 60.0, cycle_second.volume])
    residual_first = cycle_first.pressure - pmus_first - A_first @ RE
    residual_second = cycle_second.pressure - pmus_second - A_second @ RE
    cost = residual_first @ residual_first + residual_second @ residual_second
    if l2_reg:
        cost = cost + 1e-3 * (pmus_first @ pmus_first + pmus_second @ pmus_second)
    model.setObjective(cost, GRB.MINIMIZE)
    model.optimize()

    pmus_hat_first = np.asarray(pmus_first.X).ravel()
    pmus_hat_second = np.asarray(pmus_second.X).ravel()
    switching_times_first = (np.arange(pmus_hat_first.size) @ tik_first.X).astype(int)
    switching_times_second = (np.arange(pmus_hat_second.size) @ tik_second.X).astype(int)
    R_hat, E_hat = float(RE.X[0]), float(RE.X[1])
    solver_time = model.Runtime
    model.dispose()
    env.dispose()
    return (
        pmus_hat_first,
        pmus_hat_second,
        R_hat, E_hat,
        switching_times_first,
        switching_times_second,
        solver_time,
    )


if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    from utils import load_recording, retrieve_parity_marks, extract_single_cycle

    CYCLE_IDX = 345
    CYCLE_IDX = 2420 # reverse trigger / early cycling?
    #CYCLE_IDX = 8099
    #CYCLE_IDX = 13775 # late cycling
    TAU_SOE = 50

    data, fs = load_recording(Path(__file__).parent / "data" / "ASL_spont_01.npz")
    ins_marks, exp_marks = retrieve_parity_marks(data["volume"].to_numpy() * 10)
    cycle = extract_single_cycle(
        df=data, fs=fs,
        ins_mark=int(ins_marks[CYCLE_IDX]),
        next_ins_mark=int(ins_marks[CYCLE_IDX+1]),
        exp_mark=int(exp_marks[CYCLE_IDX]),
        peep=5.0, offset=50,
    )

    print(f"n = {cycle.pressure.size}")
    # applying least squares in real pmus waveform
    flow_ml_s = cycle.flow / 60.0 * 1000.0
    A = np.column_stack([flow_ml_s, cycle.volume])
    b = cycle.pressure - cycle.pmus
    (R_lse, E_lse), *_ = np.linalg.lstsq(A, b, rcond=None)
    print(f"LSE true (external): R = {R_lse * 1000:.2f}, C = {1 / (E_lse):.2f}")

    pmus_qp = pmus_qp_fixed(cycle, R_lse, E_lse)
    pmus_fixed, _, solver_time_fixed = pmus_miqp_fixed(cycle, R_lse, E_lse)
    cost_fixed = np.linalg.norm(
        cycle.pressure - pmus_fixed - R_lse * flow_ml_s - E_lse * cycle.volume
    )

    delay = 0
    pmus_miqp, R_hat, E_hat, switches, solver_time_miqp = pmus_miqp_full(cycle, initial_delay_length=delay, tau_soe=TAU_SOE)
    pmus_miqp = pmus_miqp[delay:]
    switches = switches - delay
    cost_miqp = np.linalg.norm(
        cycle.pressure - pmus_miqp - R_hat * flow_ml_s - E_hat * cycle.volume
    )
    print(f"MIQP: R = {R_hat * 1000:.2f}, C = {1 / E_hat:.2f}")
    print(f"J (fixed) = {cost_fixed:.4f} ({solver_time_fixed:.2f}s)")
    print(f"J (full)  = {cost_miqp:.4f} ({solver_time_miqp:.2f}s)")

    t = cycle.time - cycle.time[0]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

    paw_est = pmus_miqp + R_hat * flow_ml_s + E_hat * cycle.volume
    axes[0].plot(t, cycle.pressure, "k", label="paw")
    axes[0].plot(t, paw_est, "tab:orange", label="paw_est (MIQP)")
    axes[0].set_ylabel("paw [cmH2O]"); axes[0].grid(True)
    axes[0].legend(loc="upper right", fontsize=10)

    axes[1].plot(t, cycle.flow, "k")
    axes[1].set_ylabel("flow [L/min]"); axes[1].grid(True)

    axes[2].plot(t, cycle.pmus, "k", label="pmus_true")
    axes[2].plot(t, cycle.pmus_mag, "tab:purple", label="pmus_mag_AI")
    axes[2].plot(t, pmus_qp, "tab:green", label="pmus_qp (QP solver)")
    axes[2].plot(t, pmus_fixed, "k", label="pmus_miqp_fixed")
    axes[2].plot(t, pmus_miqp, "tab:orange", label="pmus_miqp")
    axes[2].set_ylabel("pmus [cmH2O]"); axes[2].grid(True)
    axes[2].set_xlabel("time [s]")

    for ax in axes:
        for s in switches:
            ax.axvline(t[s], color="tab:red", linestyle="--", linewidth=1.0)
    axes[2].plot([], [], color="tab:red", label="binary switches (MIQP)")
    axes[2].legend(loc="lower right", fontsize=10)

    fig.suptitle(
        f"MIQP R = {R_hat*1000:.2f}, C = {1/E_hat:.2f}, J = {cost_miqp:.2f}"
    )
    fig.tight_layout()
    plt.show()