from functools import cache

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree


def marker_lexsort(keypoints, tolerance: float = 5.0):
    """Perform an indirect stable sort on the keypoints so that the keypoints are arranged in rows.

    Args:
        keypoints (np.ndarray): _description_
        tolerance (float, optional): Tolerence in marker positions (px). Defaults to 5.0.
    """
    x_for_sort = np.round(keypoints[:, 0] / tolerance)
    y_for_sort = np.round(keypoints[:, 1] / tolerance)
    sort_indices = np.lexsort((x_for_sort, y_for_sort))
    return keypoints[sort_indices]


def _parse_grid_shape(grid_shape: tuple[int, int] | list[int]) -> tuple[int, int]:
    """解析 marker 网格尺寸，统一成 [rows, cols]。"""
    if len(grid_shape) != 2:
        raise ValueError(f"grid_shape must be [rows, cols], got {grid_shape!r}")
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"grid_shape values must be positive, got {grid_shape!r}")
    return rows, cols


def make_tacthru_blob_detector(min_area: float = 30, max_area: float = 400, blob_color: int = 0):
    """构建 TacThru marker blob 检测器，便于不同硬件复用。"""
    det_params = cv2.SimpleBlobDetector_Params()
    det_params.filterByConvexity = False
    det_params.filterByColor = True
    det_params.blobColor = int(blob_color)
    det_params.filterByArea = True
    det_params.minArea = float(min_area)
    det_params.maxArea = float(max_area)
    det_params.minDistBetweenBlobs = 0.5
    return cv2.SimpleBlobDetector_create(det_params)


def preprocess_tacthru_blob_image(img: np.ndarray, threshold: float = 150) -> np.ndarray:
    """对 TacThru 图像做简单阈值增强，提升 marker 圆点检测稳定性。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thres = cv2.threshold(gray, threshold, 255, cv2.THRESH_TOZERO)
    return cv2.normalize(thres, None, 0, 255, cv2.NORM_MINMAX)


def detect_marker_grid(frame: np.ndarray, grid_shape: tuple[int, int] | list[int], blob_det: callable = None) -> dict:
    """在参考帧中直接检测规则圆点网格，适配 5x7 等新款 TacThru。"""
    rows, cols = _parse_grid_shape(grid_shape)
    blob_detector = blob_det if hasattr(blob_det, "detect") else make_tacthru_blob_detector(min_area=20, max_area=5000)
    thres = preprocess_tacthru_blob_image(frame)
    blobs = blob_detector.detect(thres)
    all_kpts = np.asarray(cv2.KeyPoint_convert(blobs), dtype=np.float32).reshape(-1, 2)
    res = {"thres": thres, "all_kpts": all_kpts}

    flags = cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
    ok, centers = cv2.findCirclesGrid(thres, (cols, rows), flags=flags, blobDetector=blob_detector)
    res["marker_grid"] = centers.reshape(-1, 2).astype(np.float32, copy=False) if ok else None
    return res


class FrameProcessor:
    def __init__(self):
        pass

    def process(self, frame: np.ndarray, ts: float = None) -> dict:
        return {}

    def reset(self, **kwargs) -> None:
        pass

    # @torch.no_grad()
    def __call__(self, frame: np.ndarray) -> dict:
        return self.process(frame)

    @cache
    def get_output_dims(self) -> dict[str, int]:
        res = self.process(np.zeros((400, 400, 3), dtype=np.uint8))
        return {k: v.shape[-1] for k, v in res.items()}


def get_default_tacthru_tracker():
    filter_double_det = make_tacthru_blob_detector()

    def double_det_fn(img: np.ndarray):
        res = {}
        thres = preprocess_tacthru_blob_image(img)

        res["thres"] = thres
        res["blobs"] = filter_double_det.detect(thres)
        return res

    return double_det_fn


class KeypointsKFProcessor(FrameProcessor):
    n_effective_markers: int = 0

    def __init__(self, ref_marker_pos, blob_det: callable = None):
        self.blob_det: callable = get_default_tacthru_tracker() if blob_det is None else blob_det
        ref_marker_pos = marker_lexsort(ref_marker_pos, tolerance=25)
        self.update_ref_markers(ref_marker_pos)

    def update_ref_markers(self, kpts_ref: np.ndarray):
        self.kpts_ref = kpts_ref
        self.kpts_ref_tree = KDTree(self.kpts_ref, metric="minkowski", p=2)
        self.kpts_buffer = np.zeros_like(self.kpts_ref)
        self.kpts_dist_buffer = np.zeros(len(self.kpts_ref))
        self.kpts_buffer_valid = np.zeros(len(self.kpts_ref), dtype=bool)
        self.marker_id = np.arange(len(self.kpts_ref))

        self.reset()

    def reset(self):
        self.n_kpts = 0
        self.max_n_kpts = len(self.kpts_ref)
        self.marker_arange = np.arange(self.max_n_kpts)

        self.x_max_dist = 15

        self.state_noise_cov = np.diag([0.105**2, 0.105**2])
        self.obs_noise_cov = np.diag([0.421**2, 0.421**2])
        self.kf_x = self.kpts_ref.copy()

        # Pre-allocate buffers for KF
        self.kf_z = self.kpts_ref.copy()
        self.kf_pred = self.kpts_ref.copy()

        self.kf_cov = np.zeros([len(self.kpts_ref), 2, 2], dtype=np.float32)

        self.kf_gain = np.zeros([len(self.kpts_ref), 2, 2], dtype=np.float32)
        self.kf_update = np.zeros_like(self.kf_x)
        self.kf_update_cov = np.zeros([len(self.kpts_ref), 2, 2], dtype=np.float32)

        self.S = np.zeros_like(self.kf_cov)
        self.S_inv = np.zeros_like(self.kf_cov)
        self.innovation = np.zeros_like(self.kf_x)
        self.identity = np.eye(2, dtype=np.float32)[None, :, :]
        self.tmp_cov = np.zeros_like(self.kf_cov)
        self.tmp_update = np.zeros((len(self.kpts_ref), 2, 1), dtype=np.float32)

    def process(self, frame):
        res = self.blob_det(frame)
        blobs = res.pop("blobs")
        all_kpts = np.asarray(cv2.KeyPoint_convert(blobs), dtype=np.float32)
        if all_kpts.ndim != 2:
            all_kpts = np.zeros((0, 2), dtype=np.float32)

        # 新硬件在大形变或反光时可能出现短时漏检；此时保持上一帧 KF 状态继续运行。
        if len(all_kpts) == 0:
            self.n_effective_markers = 0
            res.update({"marker": self.kf_x.copy(), "marker_ref": self.kpts_ref, "all_kpts": all_kpts})
            return res

        kpts_to_x_cdist = cdist(self.kf_x, all_kpts)
        indices = np.argmin(kpts_to_x_cdist, axis=1)
        valid_mask = kpts_to_x_cdist[self.marker_arange, indices] < self.x_max_dist

        self.n_effective_markers = valid_mask.sum()

        # Prepare measurements
        matched_kpts = all_kpts[indices]
        np.copyto(self.kf_z, self.kf_x)
        self.kf_z[valid_mask] = matched_kpts[valid_mask]

        # Prediction: x_pred = x, P_pred = P + Q
        np.copyto(self.kf_pred, self.kf_x)
        self.kf_cov += self.state_noise_cov

        # S = P_pred + R
        np.add(self.kf_cov, self.obs_noise_cov, out=self.S)

        # Clean inversion
        det = self.S[:, 0, 0] * self.S[:, 1, 1] - self.S[:, 0, 1] * self.S[:, 1, 0]
        # Avoid zero div
        det = np.where(np.abs(det) < 1e-9, 1e-9, det)

        self.S_inv[:, 0, 0] = self.S[:, 1, 1] / det
        self.S_inv[:, 1, 1] = self.S[:, 0, 0] / det
        self.S_inv[:, 0, 1] = -self.S[:, 0, 1] / det
        self.S_inv[:, 1, 0] = -self.S[:, 1, 0] / det

        # K = P_pred @ S_inv
        np.matmul(self.kf_cov, self.S_inv, out=self.kf_gain)

        # for x_update calculation
        np.subtract(self.kf_z, self.kf_pred, out=self.innovation)
        np.matmul(self.kf_gain, self.innovation[..., None], out=self.tmp_update)
        np.add(self.kf_pred, self.tmp_update.squeeze(-1), out=self.kf_update)

        # P_upd = (I - K) P_pred
        np.subtract(self.identity, self.kf_gain, out=self.tmp_cov)
        np.matmul(self.tmp_cov, self.kf_cov, out=self.kf_update_cov)

        # Apply updates
        np.copyto(self.kf_x, self.kf_update)

        # For valid markers, use updated covariance.
        # For invalid markers, keep predicted covariance (which is already in self.kf_cov)
        self.kf_cov[valid_mask] = self.kf_update_cov[valid_mask]

        res.update({"marker": self.kf_x.copy(), "marker_ref": self.kpts_ref, "all_kpts": all_kpts})

        return res
