from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / 'analysis_outputs' / 'stress_test_2026-10-27'
SOURCE = ROOT / "tier2_pairwise_sensitivity.csv"
OUTPUT = ROOT / "tier2_pairwise_sharpe_heatmaps.png"


def label(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:g}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def main() -> None:
    data = pd.read_csv(SOURCE)
    pairs = list(data[["parameter_x", "parameter_y"]].drop_duplicates().itertuples(index=False, name=None))
    columns = 3
    rows = math.ceil(len(pairs) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(18, 4.2 * rows), constrained_layout=True)
    axes = axes.ravel()
    for axis, (left, right) in zip(axes, pairs):
        subset = data.loc[data["parameter_x"].eq(left) & data["parameter_y"].eq(right)]
        matrix = subset.pivot(index="value_y", columns="value_x", values="sharpe").sort_index(ascending=False)
        matrix.index = [label(float(value)) for value in matrix.index]
        matrix.columns = [label(float(value)) for value in matrix.columns]
        values = matrix.to_numpy(dtype=float)
        image = axis.imshow(values, cmap="RdYlGn", vmin=min(0.0, float(np.nanmin(values))), vmax=max(6.0, float(np.nanmax(values))), aspect="auto")
        axis.set_xticks(range(len(matrix.columns)), matrix.columns)
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=8)
        axis.set_title(f"{left} vs {right}")
        axis.set_xlabel(left)
        axis.set_ylabel(right)
    for axis in axes[len(pairs):]:
        axis.axis("off")
    figure.suptitle("Annualized daily Sharpe across pairwise parameter stresses", fontsize=16)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
