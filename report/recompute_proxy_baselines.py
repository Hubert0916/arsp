"""Recompute the Empirical Mean Brier curve from historical public proxy rows.

The proxy artifact stores the smoothed term and its budget-zero prior. Since
p_smooth(n) = (s_n + 4 p_0)/(n + 4), the integer acquired success count s_n
can be recovered exactly (up to floating-point roundoff), allowing the
unsmoothed Empirical Mean to be evaluated on the identical 5,680 rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)


def main() -> None:
    rows = json.loads((Path(__file__).parent.parent / "analysis" / "analysis_source_topic.json").read_text())
    empirical_curve = []
    for n in BUDGETS:
        y = np.asarray([r[f"y_{n}"] for r in rows], dtype=float)
        smooth = np.asarray([r[f"smoothed_{n}"] for r in rows], dtype=float)
        if n == 0:
            empirical = np.full(len(rows), 0.5)
        else:
            p0 = np.asarray([r["smoothed_0"] for r in rows], dtype=float)
            successes = np.rint((n + 4.0) * smooth - 4.0 * p0)
            empirical = successes / n
        empirical_curve.append(float(np.mean((empirical - y) ** 2)))
    alc = float(np.dot(WEIGHTS, empirical_curve))
    print("empirical_mean", [round(v, 10) for v in empirical_curve], round(alc, 10))


if __name__ == "__main__":
    main()
