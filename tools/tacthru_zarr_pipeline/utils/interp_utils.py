from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def get_interp1d(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return interp1d(x, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1]))


def get_gripper_calibration_interpolator(aruco_measured_width, aruco_actual_width):
    return get_interp1d(np.asarray(aruco_measured_width), np.asarray(aruco_actual_width))
