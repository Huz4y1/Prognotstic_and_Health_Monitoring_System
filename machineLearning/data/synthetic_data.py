import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_TOTAL = 10_000
FAILURE_RATIO = 0.30

rng = np.random.default_rng(RANDOM_SEED)


def _clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def generate_normal(n: int) -> pd.DataFrame:
    temp      = _clip(rng.normal(65,    12,   n), 20,    130)
    vibration = _clip(rng.normal(0.3,   0.15, n), 0.0,   1.4)
    current   = _clip(rng.normal(1.8,   0.4,  n), 0.2,   3.0)
    rpm       = _clip(rng.normal(24000, 4000, n), 5000,  38000)
    roll      = rng.uniform(-10, 10, n)
    pitch     = rng.uniform(-5,  5,  n)
    yaw       = rng.uniform(-8,  8,  n)
    label     = np.zeros(n, dtype=np.int32)
    return pd.DataFrame(dict(temp=temp, vibration=vibration, current=current,
                             rpm=rpm, roll=roll, pitch=pitch, yaw=yaw, label=label))


def generate_thermal_vibration(n: int) -> pd.DataFrame:
    temp      = _clip(rng.normal(155, 15,  n), 130,   200)
    vibration = _clip(rng.normal(3.2, 0.7,  n), 2.5,   5.0)
    current   = _clip(rng.normal(2.4, 0.5,  n), 1.0,   4.5)
    rpm       = _clip(rng.normal(28000, 5000, n), 10000, 46000)
    roll      = rng.uniform(-20, 20, n)
    pitch     = rng.uniform(-15, 15, n)
    yaw       = rng.uniform(-15, 15, n)
    label     = np.ones(n, dtype=np.int32)
    return pd.DataFrame(dict(temp=temp, vibration=vibration, current=current,
                             rpm=rpm, roll=roll, pitch=pitch, yaw=yaw, label=label))


def generate_overspeed(n: int) -> pd.DataFrame:
    temp      = _clip(rng.normal(90, 18, n), 55,    160)
    vibration = _clip(rng.normal(1.1, 0.4, n), 0.3,   3.5)
    current   = _clip(rng.normal(4.0, 0.3, n), 3.2,   5.0)
    rpm       = _clip(rng.normal(46000, 1500, n), 44000, 50000)
    roll      = rng.uniform(-12, 12, n)
    pitch     = rng.uniform(-8,  8,  n)
    yaw       = rng.uniform(-10, 10, n)
    label     = np.ones(n, dtype=np.int32)
    return pd.DataFrame(dict(temp=temp, vibration=vibration, current=current,
                             rpm=rpm, roll=roll, pitch=pitch, yaw=yaw, label=label))


def generate_electrical(n: int) -> pd.DataFrame:
    temp      = _clip(rng.normal(70, 15, n), 30,   130)
    vibration = _clip(rng.normal(0.5, 0.3, n), 0.0,  2.0)
    current   = _clip(rng.normal(4.5, 0.25, n), 4.2,  5.0)
    rpm       = _clip(rng.normal(22000, 5000, n), 5000, 38000)
    roll      = rng.uniform(-8,  8,  n)
    pitch     = rng.uniform(-5,  5,  n)
    yaw       = rng.uniform(-6,  6,  n)
    label     = np.ones(n, dtype=np.int32)
    return pd.DataFrame(dict(temp=temp, vibration=vibration, current=current,
                             rpm=rpm, roll=roll, pitch=pitch, yaw=yaw, label=label))


def main() -> None:
    n_failure = int(N_TOTAL * FAILURE_RATIO)
    n_normal  = N_TOTAL - n_failure

    n_mode1 = n_failure // 3
    n_mode2 = n_failure // 3
    n_mode3 = n_failure - n_mode1 - n_mode2

    df_normal = generate_normal(n_normal)
    df_mode1  = generate_thermal_vibration(n_mode1)
    df_mode2  = generate_overspeed(n_mode2)
    df_mode3  = generate_electrical(n_mode3)

    df = pd.concat([df_normal, df_mode1, df_mode2, df_mode3], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Explicitly cast to native dtypes — fixes NumPy _ArrayMemoryError
    # on Python 3.12+ where pandas may produce non-standard NumPy string types
    # when calling to_csv() on certain array dtypes.
    for col in ["temp", "vibration", "current", "rpm", "roll", "pitch", "yaw"]:
        df[col] = df[col].astype("float64")
    df["label"] = df["label"].astype("int32")

    print(f"Total rows    : {len(df)}")
    print(f"Normal  (0)   : {(df['label'] == 0).sum()}")
    print(f"Failure (1)   : {(df['label'] == 1).sum()}")
    print(df.describe().round(3))

    out_dir  = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "data.csv")
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
