"""Configured TacThru marker processors with a common output contract."""

from __future__ import annotations

import cv2
import numpy as np

from utils.proc_utils import KeypointsKFProcessor
from utils.robust_marker_tracker import HighReliabilityGridMarkerTracker


HIGH_RELIABILITY_ALGORITHMS = {
    "high-reliability",
    "high_reliability",
    "robust-v2",
}
LEGACY_ALGORITHMS = {"legacy", "keypoints-kf", "keypoints_kf"}
MARKER_QUALITY_KEYS = (
    "marker_valid",
    "marker_fallback",
    "marker_estimated",
    "marker_confidence",
)


def tracking_algorithm(tracking_cfg) -> str:
    algorithm = str(tracking_cfg.get("algorithm", "legacy")).strip().lower()
    if algorithm in HIGH_RELIABILITY_ALGORITHMS:
        return "high-reliability"
    if algorithm in LEGACY_ALGORITHMS:
        return "legacy"
    raise ValueError(f"Unsupported TacThru tracking algorithm: {algorithm!r}")


class HighReliabilityMarkerProcessor:
    """Adapt high-reliability tracking to the existing FrameProcessor dict API."""

    n_effective_markers: int = 0

    def __init__(
        self,
        reference_points: np.ndarray,
        *,
        color_order: str,
        tracking_cfg,
    ) -> None:
        tracker_kwargs = {}
        configurable_keys = (
            "template_half_size",
            "template_search_radius",
            "weak_template_score",
            "strong_template_score",
            "domain_agreement_distance",
            "cue_agreement_distance",
            "primary_template_score",
            "minimum_position_gain",
            "maximum_position_gain",
            "verified_template_weight",
            "minimum_output_confidence",
            "max_fallback_frames",
            "minimum_measured_fraction_for_fallback",
            "template_warmup_frames",
            "detector_validation_interval",
            "max_step_distance",
            "max_reference_displacement",
            "reacquire_after",
            "reacquire_radius",
            "minimum_quality",
            "quality_cost",
            "velocity_gain",
            "optical_flow_window",
            "flow_fb_max_error",
        )
        for key in configurable_keys:
            value = tracking_cfg.get(key)
            if value is not None:
                tracker_kwargs[key] = value
        self.tracker = HighReliabilityGridMarkerTracker(
            reference_points,
            color_order=color_order,
            **tracker_kwargs,
        )

    def reset(self) -> None:
        self.tracker.reset()

    def process(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        result = self.tracker(frame)
        valid = np.asarray(result.valid_mask, dtype=bool)
        fallback = (
            np.zeros_like(valid)
            if result.fallback_mask is None
            else np.asarray(result.fallback_mask, dtype=bool)
        )
        estimated = (
            self.tracker.position.copy()
            if result.estimated_marker is None
            else np.asarray(result.estimated_marker, dtype=np.float32)
        )
        marker = np.asarray(result.marker, dtype=np.float32).copy()
        use_estimate = ~valid | ~np.isfinite(marker).all(axis=1)
        marker[use_estimate] = estimated[use_estimate]
        if not np.isfinite(marker).all():
            raise RuntimeError("High-reliability tracker produced non-finite marker output")

        self.n_effective_markers = int(np.count_nonzero(valid & ~fallback))
        return {
            "marker": marker,
            "marker_ref": np.asarray(result.marker_ref, dtype=np.float32),
            "all_kpts": np.asarray(result.candidates, dtype=np.float32),
            "thres": np.asarray(result.threshold_image, dtype=np.uint8),
            "marker_valid": valid,
            "marker_fallback": fallback,
            "marker_estimated": ~valid,
            "marker_confidence": np.asarray(result.confidence, dtype=np.float32),
            "tracking_warmup_remaining": np.int32(
                result.template_warmup_remaining
            ),
        }

    __call__ = process


def create_tactile_marker_processor(
    reference_points: np.ndarray,
    tracking_cfg,
    *,
    color_order: str,
):
    algorithm = tracking_algorithm(tracking_cfg)
    if algorithm == "legacy":
        return KeypointsKFProcessor(reference_points)
    opencv_threads = int(tracking_cfg.get("opencv_threads", 12))
    if opencv_threads <= 0:
        raise ValueError("tracking.opencv_threads must be positive")
    cv2.setNumThreads(opencv_threads)
    return HighReliabilityMarkerProcessor(
        reference_points,
        color_order=color_order,
        tracking_cfg=tracking_cfg,
    )


__all__ = [
    "HighReliabilityMarkerProcessor",
    "MARKER_QUALITY_KEYS",
    "create_tactile_marker_processor",
    "tracking_algorithm",
]
