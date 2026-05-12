import json
from pathlib import Path

import gdown

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

RANK1_SCALE = np.array([0.85, 0.70, 0.85, 0.85])
TX_SCALE_1 = np.array([0.85, 0.65, 0.85, 0.85])
TX_SCALE_2 = np.array([0.85, 0.65, 0.85, 0.85])
TX_SCALE_3 = 0.5

# Download the dataset
url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
downloaded_file = "challenge.mat"
if not Path(downloaded_file).exists():
    gdown.download(url, downloaded_file, quiet=False, fuzzy=True)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def rank1_approx(band_matrix):
    cov = band_matrix.conj().T @ band_matrix
    _, vecs = np.linalg.eigh(cov)
    v = vecs[:, -1]
    return (band_matrix @ v)[:, None] * v.conj()[None, :]


def your_canceller(tx_n, rx):
    band_rx = np.column_stack(
        [helpers["score_filter"](rx[:, ch]) for ch in range(rx.shape[1])]
    )
    # rank-1 component subtraction
    rx_hat = rx - RANK1_SCALE[None, :] * rank1_approx(band_rx)

    # TX-component subtraction (first pass)
    tx_pred = helpers["fit_tx_prediction"](rx_hat)
    rx_hat = rx_hat - TX_SCALE_1[None, :] * tx_pred

    # TX-component subtraction (second pass)
    tx_pred = helpers["fit_tx_prediction"](rx_hat)
    rx_hat = rx_hat - TX_SCALE_2[None, :] * tx_pred

    # TX-component subtraction (third pass)
    tx_pred = helpers["fit_tx_prediction"](rx_hat)
    return rx_hat - TX_SCALE_3 * tx_pred


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
