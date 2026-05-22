from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import filtfilt, firwin

@dataclass
class Cycle:
    time: np.ndarray
    pressure: np.ndarray
    flow: np.ndarray
    volume: np.ndarray
    pmus: np.ndarray
    pmus_mag: np.ndarray
    insexp: np.ndarray # 1 during inspiration, 0 during expiration

def load_recording(npz_path: str | Path) -> tuple[pd.DataFrame, float]:
    npz_object = np.load(npz_path)
    fs = float(npz_object["fs"])
    arrays = {k: npz_object[k] for k in npz_object.files if npz_object[k].ndim > 0}
    df = pd.DataFrame(arrays)
    df["time"] = np.arange(len(df)) / fs
    return df, fs


def retrieve_parity_marks(volume_x10: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ins_marks = []
    exp_marks = []
    parity = int(volume_x10[0]) % 2
    for i in range(1, volume_x10.size):
        v = volume_x10[i]
        if v == 0.0:
            continue
        new_parity = int(v) % 2
        if new_parity != parity:
            if parity == 0:
                ins_marks.append(i)
            else:
                exp_marks.append(i)
        parity = new_parity
    return np.asarray(ins_marks, dtype=np.int64), np.asarray(exp_marks, dtype=np.int64)


def retrieve_flow_marks(
    flow: np.ndarray,
    fs: float,
    flow_threshold: float = 10.0,
    smoothing_cutoff: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    departure_threshold = flow_threshold * 0.5
    filtered = fir_filter(8, smoothing_cutoff, fs, flow)
    ins_marks: list[int] = []
    exp_marks: list[int] = []
    state = 0  # +1 after entering inspiration, -1 after entering expiration
    for i in range(1, filtered.size):
        if state != 1 and filtered[i] >= flow_threshold:
            j = i
            while j > 0 and filtered[j - 1] > departure_threshold:
                j -= 1
            ins_marks.append(j)
            state = 1
        elif state != -1 and filtered[i] <= -flow_threshold:
            j = i
            while j > 0 and filtered[j - 1] < -departure_threshold:
                j -= 1
            exp_marks.append(j)
            state = -1
    return np.asarray(ins_marks, dtype=np.int64), np.asarray(exp_marks, dtype=np.int64)


def fir_filter(order: int, cutoff: float, fs: float, x: np.ndarray) -> np.ndarray:
    wn = cutoff / (fs / 2.0)
    taps = firwin(order + 1, wn, window="hann")
    return filtfilt(taps, [1.0], x)

# important! removes peep from pressure
def extract_single_cycle(
    df: pd.DataFrame,
    fs: float,
    ins_mark: int,
    next_ins_mark: int,
    exp_mark: int,
    peep: float,
    offset: int = 30,
) -> Cycle:
    start = ins_mark - offset
    stop = next_ins_mark - offset
    sliced = df.iloc[start:stop]

    flow = sliced["flow"].to_numpy()
    pressure = sliced["pressure"].to_numpy()
    volume = sliced["volume"].to_numpy()

    flow = fir_filter(8, 0.2, fs, sliced["flow"].to_numpy())
    pressure = fir_filter(8, 0.2, fs, sliced["pressure"].to_numpy())
    volume = fir_filter(8, 0.2, fs, sliced["volume"].to_numpy())

    pressure = pressure - peep
    volume = volume - volume[offset]

    exp_start = exp_mark - ins_mark + offset
    insexp = np.ones(pressure.size)
    insexp[exp_start:] = 0

    return Cycle(
        time=sliced["time"].to_numpy(),
        pressure=pressure,
        flow=flow,
        volume=volume,
        pmus=sliced["pmus"].to_numpy(),
        pmus_mag=sliced["pmus_mag"].to_numpy(),
        insexp=insexp,
    )
