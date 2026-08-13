"""Illumination-robust tracking for TacThru ring markers.

The tracker is shared by the viewer, calibration, configured live clients, and
offline dataset processing. Legacy sensors remain selectable through config.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class MarkerDetection:
    points: np.ndarray
    quality: np.ndarray
    threshold_image: np.ndarray
    gray: np.ndarray


@dataclass(frozen=True)
class MarkerTrackingResult:
    marker: np.ndarray
    marker_ref: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray
    match_distance: np.ndarray
    candidates: np.ndarray
    candidate_quality: np.ndarray
    threshold_image: np.ndarray
    template_confidence: np.ndarray | None = None
    flow_valid_mask: np.ndarray | None = None
    cue_count: np.ndarray | None = None
    rejection_code: np.ndarray | None = None
    template_warmup_remaining: int = 0
    fallback_mask: np.ndarray | None = None
    estimated_marker: np.ndarray | None = None


def _to_gray(frame: np.ndarray, color_order: str) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxW or HxWx3 image, got {image.shape}")
    order = color_order.upper()
    if order == "RGB":
        code = cv2.COLOR_RGB2GRAY
    elif order == "BGR":
        code = cv2.COLOR_BGR2GRAY
    else:
        raise ValueError(f"color_order must be RGB or BGR, got {color_order!r}")
    return cv2.cvtColor(image, code)


def _make_local_blob_detector(
    *,
    min_area: float,
    max_area: float,
) -> cv2.SimpleBlobDetector:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)
    params.filterByCircularity = True
    params.minCircularity = 0.45
    params.filterByConvexity = True
    params.minConvexity = 0.60
    params.filterByInertia = True
    params.minInertiaRatio = 0.25
    params.minDistBetweenBlobs = 6.0
    return cv2.SimpleBlobDetector_create(params)


class AdaptiveRingMarkerDetector:
    """Detect dark marker centers relative to their local bright annulus.

    The detector works on a local-darkness image rather than an absolute gray
    threshold.  This makes uniform brightness, exposure, and slow illumination
    gradients largely cancel before blob detection.
    """

    def __init__(
        self,
        *,
        background_sigma: float = 8.0,
        response_percentile: float = 95.0,
        min_response: float = 8.0,
        max_response: float = 80.0,
        min_area: float = 25.0,
        max_area: float = 500.0,
        refine_radius: int = 7,
        center_radius: int = 4,
        ring_inner_radius: int = 8,
        ring_outer_radius: int = 13,
        merge_radius: float = 3.5,
    ) -> None:
        if not 0.0 < response_percentile < 100.0:
            raise ValueError("response_percentile must be between 0 and 100")
        if not 0 < center_radius < ring_inner_radius < ring_outer_radius:
            raise ValueError("Marker center/ring radii must be strictly increasing")
        self.background_sigma = float(background_sigma)
        self.response_percentile = float(response_percentile)
        self.min_response = float(min_response)
        self.max_response = float(max_response)
        self.refine_radius = int(refine_radius)
        self.center_radius = int(center_radius)
        self.ring_inner_radius = int(ring_inner_radius)
        self.ring_outer_radius = int(ring_outer_radius)
        self.merge_radius = float(merge_radius)
        self._blob_detector = _make_local_blob_detector(
            min_area=min_area,
            max_area=max_area,
        )

    def detect(self, frame: np.ndarray, *, color_order: str = "BGR") -> MarkerDetection:
        gray = _to_gray(frame, color_order)
        background = cv2.GaussianBlur(
            gray,
            (0, 0),
            sigmaX=self.background_sigma,
            sigmaY=self.background_sigma,
        )
        darkness = cv2.subtract(background, gray)
        threshold = float(
            np.clip(
                np.percentile(darkness, self.response_percentile),
                self.min_response,
                self.max_response,
            )
        )
        mask = np.where(darkness >= threshold, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
        blobs = self._blob_detector.detect(mask)
        points = np.asarray(cv2.KeyPoint_convert(blobs), dtype=np.float32).reshape(-1, 2)
        if len(points) == 0:
            return MarkerDetection(
                points=points,
                quality=np.zeros((0,), dtype=np.float32),
                threshold_image=mask,
                gray=gray,
            )

        refined, quality = self._refine_and_score(gray, darkness, points, threshold)
        refined, quality = self._deduplicate(refined, quality)
        return MarkerDetection(
            points=refined,
            quality=quality,
            threshold_image=mask,
            gray=gray,
        )

    def _refine_and_score(
        self,
        gray: np.ndarray,
        darkness: np.ndarray,
        points: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = gray.shape
        refined = points.astype(np.float32, copy=True)
        quality = np.zeros((len(points),), dtype=np.float32)
        outer = self.ring_outer_radius

        for index, (point_x, point_y) in enumerate(points):
            x0 = max(0, int(np.floor(point_x)) - outer)
            y0 = max(0, int(np.floor(point_y)) - outer)
            x1 = min(width, int(np.floor(point_x)) + outer + 1)
            y1 = min(height, int(np.floor(point_y)) + outer + 1)
            if x1 <= x0 or y1 <= y0:
                continue

            yy, xx = np.ogrid[y0:y1, x0:x1]
            distance_sq = (xx - point_x) ** 2 + (yy - point_y) ** 2
            refine_mask = distance_sq <= self.refine_radius**2
            center_mask = distance_sq <= self.center_radius**2
            ring_mask = (distance_sq >= self.ring_inner_radius**2) & (
                distance_sq <= self.ring_outer_radius**2
            )
            if not np.any(center_mask) or not np.any(ring_mask):
                continue

            response_patch = darkness[y0:y1, x0:x1].astype(np.float32)
            weights = np.where(
                refine_mask,
                np.maximum(response_patch - 0.25 * threshold, 0.0),
                0.0,
            )
            weight_sum = float(weights.sum())
            if weight_sum > 1e-6:
                refined[index, 0] = float((weights * xx).sum() / weight_sum)
                refined[index, 1] = float((weights * yy).sum() / weight_sum)

            gray_patch = gray[y0:y1, x0:x1].astype(np.float32)
            center_mean = float(gray_patch[center_mask].mean())
            ring_mean = float(gray_patch[ring_mask].mean())
            annular_contrast = np.clip(
                (ring_mean - center_mean) / max(ring_mean, 16.0),
                0.0,
                1.0,
            )
            response_strength = np.clip(
                float(response_patch[center_mask].mean()) / max(2.0 * threshold, 1.0),
                0.0,
                1.0,
            )
            quality[index] = 0.75 * annular_contrast + 0.25 * response_strength

        return refined, quality

    def _deduplicate(
        self,
        points: np.ndarray,
        quality: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(points) < 2:
            return points, quality
        pairwise_distance = cdist(points, points)
        np.fill_diagonal(pairwise_distance, np.inf)
        if float(pairwise_distance.min()) >= self.merge_radius:
            return points, quality
        keep: list[int] = []
        for index in np.argsort(-quality):
            if all(
                np.linalg.norm(points[index] - points[kept]) >= self.merge_radius
                for kept in keep
            ):
                keep.append(int(index))
        keep_array = np.asarray(keep, dtype=np.int64)
        return points[keep_array], quality[keep_array]


class RobustGridMarkerTracker:
    """Track a calibrated marker grid with unique, validity-aware identities."""

    def __init__(
        self,
        reference_points: np.ndarray,
        *,
        color_order: str = "BGR",
        detector: AdaptiveRingMarkerDetector | None = None,
        max_step_distance: float = 16.0,
        max_reference_displacement: float | None = None,
        reacquire_after: int = 2,
        reacquire_radius: float = 10.0,
        minimum_quality: float = 0.12,
        quality_cost: float = 4.0,
        position_gain: float = 0.72,
        velocity_gain: float = 0.18,
        optical_flow_window: int = 21,
        flow_fb_max_error: float = 1.5,
    ) -> None:
        reference = np.asarray(reference_points, dtype=np.float32)
        if reference.ndim != 2 or reference.shape[1] != 2 or len(reference) == 0:
            raise ValueError(f"reference_points must have shape [N,2], got {reference.shape}")
        if not np.isfinite(reference).all():
            raise ValueError("reference_points must be finite")
        self.reference = reference.copy()
        self.color_order = color_order.upper()
        _to_gray(np.zeros((1, 1, 3), dtype=np.uint8), self.color_order)
        self.detector = detector or AdaptiveRingMarkerDetector()
        self.max_step_distance = float(max_step_distance)
        self.reacquire_after = int(reacquire_after)
        self.reacquire_radius = float(reacquire_radius)
        self.minimum_quality = float(minimum_quality)
        self.quality_cost = float(quality_cost)
        self.position_gain = float(position_gain)
        self.velocity_gain = float(velocity_gain)
        self.optical_flow_window = int(optical_flow_window)
        self.flow_fb_max_error = float(flow_fb_max_error)

        if len(reference) > 1:
            ref_distances = cdist(reference, reference)
            np.fill_diagonal(ref_distances, np.inf)
            typical_spacing = float(np.median(ref_distances.min(axis=1)))
        else:
            typical_spacing = 80.0
        if max_reference_displacement is None:
            max_reference_displacement = np.clip(0.42 * typical_spacing, 18.0, 40.0)
        self.max_reference_displacement = float(max_reference_displacement)
        self.reset()

    def reset(self) -> None:
        self.position = self.reference.copy()
        self.velocity = np.zeros_like(self.reference)
        self.missed_frames = np.zeros((len(self.reference),), dtype=np.int32)
        self.previous_gray: np.ndarray | None = None

    def process(self, frame: np.ndarray) -> MarkerTrackingResult:
        detection = self.detector.detect(frame, color_order=self.color_order)
        prediction, flow_valid = self._predict_with_optical_flow(detection.gray)
        matched_indices, valid, match_distance = self._assign_unique(
            prediction,
            detection.points,
            detection.quality,
        )

        previous_position = self.position.copy()
        for marker_index in range(len(self.reference)):
            if valid[marker_index]:
                candidate = detection.points[matched_indices[marker_index]]
                innovation = candidate - prediction[marker_index]
                self.position[marker_index] = (
                    prediction[marker_index] + self.position_gain * innovation
                )
                measured_delta = self.position[marker_index] - previous_position[marker_index]
                self.velocity[marker_index] = (
                    0.55 * self.velocity[marker_index]
                    + self.velocity_gain * measured_delta
                )
                self.missed_frames[marker_index] = 0
            else:
                if flow_valid[marker_index]:
                    self.position[marker_index] = prediction[marker_index]
                self.velocity[marker_index] *= 0.5
                self.missed_frames[marker_index] += 1

        self.previous_gray = detection.gray.copy()
        marker = self.position.copy()
        marker[~valid] = np.nan
        confidence = np.zeros((len(self.reference),), dtype=np.float32)
        assigned = matched_indices >= 0
        confidence[assigned] = detection.quality[matched_indices[assigned]]
        return MarkerTrackingResult(
            marker=marker,
            marker_ref=self.reference.copy(),
            valid_mask=valid,
            confidence=confidence,
            match_distance=match_distance,
            candidates=detection.points,
            candidate_quality=detection.quality,
            threshold_image=detection.threshold_image,
        )

    __call__ = process

    def _predict_with_optical_flow(
        self,
        gray: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.position + self.velocity
        valid = np.zeros((len(self.reference),), dtype=bool)
        if self.previous_gray is None or self.previous_gray.shape != gray.shape:
            return prediction, valid

        previous_points = self.position.reshape(-1, 1, 2).astype(np.float32)
        forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            previous_points,
            None,
            winSize=(self.optical_flow_window, self.optical_flow_window),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )
        if forward is None or forward_status is None:
            return prediction, valid
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.previous_gray,
            forward,
            None,
            winSize=(self.optical_flow_window, self.optical_flow_window),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )
        if backward is None or backward_status is None:
            return prediction, valid

        forward = forward.reshape(-1, 2)
        backward = backward.reshape(-1, 2)
        forward_backward_error = np.linalg.norm(
            backward - previous_points.reshape(-1, 2),
            axis=1,
        )
        step_distance = np.linalg.norm(forward - self.position, axis=1)
        reference_distance = np.linalg.norm(forward - self.reference, axis=1)
        valid = forward_status.reshape(-1).astype(bool)
        valid &= backward_status.reshape(-1).astype(bool)
        valid &= np.isfinite(forward).all(axis=1)
        valid &= forward_backward_error <= self.flow_fb_max_error
        valid &= step_distance <= 1.5 * self.max_step_distance
        valid &= reference_distance <= self.max_reference_displacement
        prediction[valid] = forward[valid]
        return prediction, valid

    def _assign_unique(
        self,
        prediction: np.ndarray,
        candidates: np.ndarray,
        candidate_quality: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        marker_count = len(self.reference)
        matched = np.full((marker_count,), -1, dtype=np.int64)
        valid = np.zeros((marker_count,), dtype=bool)
        match_distance = np.full((marker_count,), np.inf, dtype=np.float32)
        if len(candidates) == 0:
            return matched, valid, match_distance

        predicted_distance = cdist(prediction, candidates)
        reference_distance = cdist(self.reference, candidates)
        normal_match = predicted_distance <= self.max_step_distance
        reacquire = (
            (self.missed_frames >= self.reacquire_after)[:, None]
            & (reference_distance <= self.reacquire_radius)
        )
        inside_grid_cell = reference_distance <= self.max_reference_displacement
        eligible = inside_grid_cell & (normal_match | reacquire)

        association_distance = predicted_distance.copy()
        association_distance[reacquire] = np.minimum(
            association_distance[reacquire],
            reference_distance[reacquire] + 1.0,
        )
        quality_penalty = self.quality_cost * (
            1.0 - np.clip(candidate_quality, 0.0, 1.0)
        )
        cost = association_distance + quality_penalty[None, :]
        cost[~eligible] = 1e6

        dummy_cost = self.max_step_distance + self.quality_cost + 1.0
        dummy = np.full((marker_count, marker_count), dummy_cost, dtype=np.float64)
        rows, columns = linear_sum_assignment(np.concatenate([cost, dummy], axis=1))
        for row, column in zip(rows, columns):
            if column >= len(candidates) or not eligible[row, column]:
                continue
            if candidate_quality[column] < self.minimum_quality:
                continue
            matched[row] = int(column)
            valid[row] = True
            match_distance[row] = float(predicted_distance[row, column])
        return matched, valid, match_distance


class HighReliabilityGridMarkerTracker(RobustGridMarkerTracker):
    """Fuse static templates, optical flow, and ring detections.

    Static templates are captured on the first frame after construction or
    reset. They provide an illumination-invariant anchor while optical flow
    preserves fast real motion and the ring detector provides independent
    geometric validation and recovery.
    """

    def __init__(
        self,
        reference_points: np.ndarray,
        *,
        template_half_size: int = 14,
        template_search_radius: int = 16,
        weak_template_score: float = 0.52,
        strong_template_score: float = 0.78,
        domain_agreement_distance: float = 2.0,
        cue_agreement_distance: float = 3.0,
        primary_template_score: float = 0.95,
        minimum_position_gain: float = 0.74,
        maximum_position_gain: float = 0.98,
        verified_template_weight: float = 0.90,
        minimum_output_confidence: float = 0.70,
        max_fallback_frames: int = 2,
        minimum_measured_fraction_for_fallback: float = 0.75,
        template_warmup_frames: int = 30,
        detector_validation_interval: int = 300,
        **kwargs,
    ) -> None:
        if template_half_size < 4:
            raise ValueError("template_half_size must be at least 4")
        if template_search_radius < 2:
            raise ValueError("template_search_radius must be at least 2")
        self.template_half_size = int(template_half_size)
        self.template_search_radius = int(template_search_radius)
        self.weak_template_score = float(weak_template_score)
        self.strong_template_score = float(strong_template_score)
        self.domain_agreement_distance = float(domain_agreement_distance)
        self.cue_agreement_distance = float(cue_agreement_distance)
        self.primary_template_score = float(primary_template_score)
        self.minimum_position_gain = float(minimum_position_gain)
        self.maximum_position_gain = float(maximum_position_gain)
        self.verified_template_weight = float(verified_template_weight)
        self.minimum_output_confidence = float(minimum_output_confidence)
        self.max_fallback_frames = int(max_fallback_frames)
        if self.max_fallback_frames < 0:
            raise ValueError("max_fallback_frames must be non-negative")
        self.minimum_measured_fraction_for_fallback = float(
            minimum_measured_fraction_for_fallback
        )
        if not 0.0 <= self.minimum_measured_fraction_for_fallback <= 1.0:
            raise ValueError(
                "minimum_measured_fraction_for_fallback must be in [0,1]"
            )
        if not 0.5 <= self.verified_template_weight <= 1.0:
            raise ValueError("verified_template_weight must be in [0.5, 1.0]")
        self.template_warmup_frames = max(1, int(template_warmup_frames))
        self.detector_validation_interval = max(0, int(detector_validation_interval))
        template_size = 2 * self.template_half_size + 1
        self._template_mask = np.zeros((template_size, template_size), dtype=np.uint8)
        cv2.circle(
            self._template_mask,
            (self.template_half_size, self.template_half_size),
            self.template_half_size,
            255,
            -1,
        )
        kwargs.setdefault("max_step_distance", 22.0)
        kwargs.setdefault("reacquire_radius", 18.0)
        kwargs.setdefault("optical_flow_window", 41)
        kwargs.setdefault("flow_fb_max_error", 1.25)
        if kwargs.get("max_reference_displacement") is None:
            reference = np.asarray(reference_points, dtype=np.float32)
            if len(reference) > 1:
                reference_distance = cdist(reference, reference)
                np.fill_diagonal(reference_distance, np.inf)
                typical_spacing = float(
                    np.median(reference_distance.min(axis=1))
                )
            else:
                typical_spacing = 80.0
            kwargs["max_reference_displacement"] = float(
                np.clip(0.48 * typical_spacing, 20.0, 40.0)
            )
        super().__init__(reference_points, **kwargs)
        neighbor_distance = cdist(self.reference, self.reference)
        np.fill_diagonal(neighbor_distance, np.inf)
        self._neighbor_order = np.argsort(neighbor_distance, axis=1)

    def reset(self) -> None:
        super().reset()
        self._gray_templates: list[np.ndarray | None] | None = None
        self._dark_templates: list[np.ndarray | None] | None = None
        self._template_center_offsets: np.ndarray | None = None
        self._gray_template_samples: list[list[np.ndarray]] | None = None
        self._dark_template_samples: list[list[np.ndarray]] | None = None
        self._template_warmup_count = 0
        self._frames_since_detector = 0
        self._last_threshold_image: np.ndarray | None = None

    def process(self, frame: np.ndarray) -> MarkerTrackingResult:
        if self._gray_templates is None:
            initial = super().process(frame)
            initial_centers = self.position.copy()
            initial_matches, initial_valid, initial_distance = self._assign_unique(
                self.reference,
                initial.candidates,
                initial.candidate_quality,
            )
            for marker_index in np.flatnonzero(initial_valid):
                initial_centers[marker_index] = initial.candidates[
                    initial_matches[marker_index]
                ]
            self.position[initial_valid] = initial_centers[initial_valid]
            self.velocity.fill(0.0)
            self.missed_frames[initial_valid] = 0
            self._initialize_templates(frame, initial_centers)
            self._last_threshold_image = initial.threshold_image.copy()
            marker_count = len(self.reference)
            initial_confidence = np.zeros((marker_count,), dtype=np.float32)
            assigned = initial_matches >= 0
            initial_confidence[assigned] = initial.candidate_quality[
                initial_matches[assigned]
            ]
            output_valid, fallback_mask = self._bounded_output_masks(initial_valid)
            self._interpolate_unmeasured_positions(~initial_valid, initial_valid)
            initial_marker = self.position.copy()
            initial_marker[~output_valid] = np.nan
            initial_rejection = np.where(initial_valid, 0, 1).astype(np.int8)
            initial_rejection[fallback_mask] = 4
            return MarkerTrackingResult(
                marker=initial_marker,
                marker_ref=initial.marker_ref,
                valid_mask=output_valid,
                confidence=initial_confidence,
                match_distance=initial_distance,
                candidates=initial.candidates,
                candidate_quality=initial.candidate_quality,
                threshold_image=initial.threshold_image,
                template_confidence=np.zeros((marker_count,), dtype=np.float32),
                flow_valid_mask=np.zeros((marker_count,), dtype=bool),
                cue_count=initial_valid.astype(np.int8),
                rejection_code=initial_rejection,
                template_warmup_remaining=max(
                    self.template_warmup_frames - self._template_warmup_count,
                    0,
                ),
                fallback_mask=fallback_mask,
                estimated_marker=self.position.copy(),
            )

        gray = _to_gray(frame, self.color_order)
        prediction, flow_valid = self._predict_with_optical_flow(gray)
        template_points, template_confidence, template_valid = self._template_proposals(
            gray,
            prediction,
        )

        scheduled_validation = bool(
            self.detector_validation_interval > 0
            and self._frames_since_detector >= self.detector_validation_interval
        )
        need_detector = bool(
            scheduled_validation
            or not template_valid.all()
            or np.any(template_confidence < self.strong_template_score)
        )
        if need_detector:
            detection = self.detector.detect(frame, color_order=self.color_order)
            self._last_threshold_image = detection.threshold_image.copy()
            self._frames_since_detector = 0
        else:
            threshold_image = (
                self._last_threshold_image
                if self._last_threshold_image is not None
                else np.zeros_like(gray)
            )
            detection = MarkerDetection(
                points=np.zeros((0, 2), dtype=np.float32),
                quality=np.zeros((0,), dtype=np.float32),
                threshold_image=threshold_image,
                gray=gray,
            )
            self._frames_since_detector += 1

        association_target = prediction.copy()
        association_target[template_valid] = template_points[template_valid]
        matched_indices, candidate_valid, match_distance = self._assign_unique(
            association_target,
            detection.points,
            detection.quality,
        )

        marker_count = len(self.reference)
        measurement = np.zeros_like(self.position)
        measurement_valid = np.zeros((marker_count,), dtype=bool)
        confidence = np.zeros((marker_count,), dtype=np.float32)
        cue_count = np.zeros((marker_count,), dtype=np.int8)
        rejection_code = np.ones((marker_count,), dtype=np.int8)

        for marker_index in range(marker_count):
            candidate_index = matched_indices[marker_index]
            has_candidate = bool(candidate_valid[marker_index] and candidate_index >= 0)
            candidate = (
                detection.points[candidate_index]
                if has_candidate
                else np.asarray([np.nan, np.nan], dtype=np.float32)
            )
            candidate_quality = (
                float(detection.quality[candidate_index]) if has_candidate else 0.0
            )
            has_template = bool(template_valid[marker_index])
            template_score = float(template_confidence[marker_index])
            template = template_points[marker_index]

            flow_template_agree = bool(
                has_template
                and flow_valid[marker_index]
                and np.linalg.norm(template - prediction[marker_index])
                <= self.cue_agreement_distance
            )
            candidate_template_agree = bool(
                has_template
                and has_candidate
                and np.linalg.norm(candidate - template) <= self.cue_agreement_distance
            )
            candidate_flow_agree = bool(
                has_candidate
                and flow_valid[marker_index]
                and np.linalg.norm(candidate - prediction[marker_index])
                <= self.cue_agreement_distance
            )
            cue_count[marker_index] = int(has_template) + int(has_candidate) + int(
                flow_valid[marker_index]
            )

            accept_template = has_template and (
                template_score >= self.strong_template_score
                or (
                    template_score >= self.weak_template_score
                    and (flow_template_agree or candidate_template_agree)
                )
            )
            if accept_template and has_candidate and candidate_template_agree:
                template_weight = self.verified_template_weight
                candidate_weight = 1.0 - template_weight
                measurement[marker_index] = (
                    template_weight * template + candidate_weight * candidate
                )
                confidence[marker_index] = np.clip(
                    0.65 * template_score + 0.35 * candidate_quality,
                    0.0,
                    1.0,
                )
                measurement_valid[marker_index] = True
                rejection_code[marker_index] = 0
            elif accept_template:
                measurement[marker_index] = template
                confidence[marker_index] = template_score
                measurement_valid[marker_index] = True
                rejection_code[marker_index] = 0
            elif has_candidate and (
                candidate_flow_agree or candidate_quality >= 0.55
            ):
                measurement[marker_index] = candidate
                confidence[marker_index] = np.clip(
                    0.8 * candidate_quality + 0.2 * float(candidate_flow_agree),
                    0.0,
                    1.0,
                )
                measurement_valid[marker_index] = True
                rejection_code[marker_index] = 0
            elif has_template and has_candidate and not candidate_template_agree:
                rejection_code[marker_index] = 2
            elif has_template or has_candidate or flow_valid[marker_index]:
                rejection_code[marker_index] = 3

        low_confidence = measurement_valid & (
            confidence < self.minimum_output_confidence
        )
        measurement_valid[low_confidence] = False
        rejection_code[low_confidence] = 3

        previous_position = self.position.copy()
        for marker_index in range(marker_count):
            if measurement_valid[marker_index]:
                adaptive_gain = self.minimum_position_gain + (
                    self.maximum_position_gain - self.minimum_position_gain
                ) * confidence[marker_index]
                innovation = measurement[marker_index] - prediction[marker_index]
                self.position[marker_index] = (
                    prediction[marker_index] + adaptive_gain * innovation
                )
                measured_delta = self.position[marker_index] - previous_position[marker_index]
                self.velocity[marker_index] = (
                    0.45 * self.velocity[marker_index]
                    + self.velocity_gain * measured_delta
                )
                self.missed_frames[marker_index] = 0
            else:
                if flow_valid[marker_index]:
                    self.position[marker_index] = prediction[marker_index]
                self.velocity[marker_index] *= 0.5
                self.missed_frames[marker_index] += 1

        self.previous_gray = detection.gray.copy()
        self._update_template_warmup(
            detection.gray,
            self.position,
            measurement_valid,
            confidence,
        )
        output_valid, fallback_mask = self._bounded_output_masks(measurement_valid)
        self._interpolate_unmeasured_positions(
            ~measurement_valid,
            measurement_valid,
        )
        rejection_code[fallback_mask] = 4
        marker = self.position.copy()
        marker[~output_valid] = np.nan
        return MarkerTrackingResult(
            marker=marker,
            marker_ref=self.reference.copy(),
            valid_mask=output_valid,
            confidence=confidence,
            match_distance=match_distance,
            candidates=detection.points,
            candidate_quality=detection.quality,
            threshold_image=detection.threshold_image,
            template_confidence=template_confidence,
            flow_valid_mask=flow_valid,
            cue_count=cue_count,
            rejection_code=rejection_code,
            template_warmup_remaining=max(
                self.template_warmup_frames - self._template_warmup_count,
                0,
            ),
            fallback_mask=fallback_mask,
            estimated_marker=self.position.copy(),
        )

    __call__ = process

    def _bounded_output_masks(
        self,
        measured_valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fill short per-marker gaps while rejecting systemic tracking failure."""

        output_valid = measured_valid.astype(bool, copy=True)
        fallback = np.zeros_like(output_valid)
        if self.max_fallback_frames == 0 or len(output_valid) == 0:
            return output_valid, fallback
        measured_fraction = float(np.count_nonzero(measured_valid)) / len(
            measured_valid
        )
        if measured_fraction < self.minimum_measured_fraction_for_fallback:
            return output_valid, fallback
        eligible = (
            ~output_valid
            & (self.missed_frames <= self.max_fallback_frames)
            & np.isfinite(self.position).all(axis=1)
        )
        output_valid[eligible] = True
        fallback[eligible] = True
        return output_valid, fallback

    def _interpolate_unmeasured_positions(
        self,
        unmeasured_mask: np.ndarray,
        measured_valid: np.ndarray,
    ) -> None:
        """Keep unmeasured marker estimates continuous using local topology."""

        for marker_index in np.flatnonzero(unmeasured_mask):
            neighbors = self._neighbor_order[marker_index]
            neighbors = neighbors[measured_valid[neighbors]][:4]
            if len(neighbors) < 2:
                continue
            distance = np.linalg.norm(
                self.reference[neighbors] - self.reference[marker_index],
                axis=1,
            )
            weights = 1.0 / np.maximum(distance, 1e-6)
            weights /= weights.sum()
            displacement = self.position[neighbors] - self.reference[neighbors]
            self.position[marker_index] = self.reference[marker_index] + np.sum(
                weights[:, None] * displacement,
                axis=0,
            )
            self.velocity[marker_index] = np.sum(
                weights[:, None] * self.velocity[neighbors],
                axis=0,
            )

    def _initialize_templates(
        self,
        frame: np.ndarray,
        centers: np.ndarray,
    ) -> None:
        gray = _to_gray(frame, self.color_order)
        darkness = self._local_darkness(gray)
        gray_templates: list[np.ndarray | None] = []
        dark_templates: list[np.ndarray | None] = []
        offsets = np.zeros_like(centers, dtype=np.float32)
        gray_samples: list[list[np.ndarray]] = []
        dark_samples: list[list[np.ndarray]] = []
        for center in centers:
            gray_template = self._extract_template(gray, center)
            dark_template = self._extract_template(darkness, center)
            gray_templates.append(gray_template)
            dark_templates.append(dark_template)
            gray_samples.append([] if gray_template is None else [gray_template])
            dark_samples.append([] if dark_template is None else [dark_template])
        self._gray_templates = gray_templates
        self._dark_templates = dark_templates
        self._template_center_offsets = offsets
        self._gray_template_samples = gray_samples
        self._dark_template_samples = dark_samples
        self._template_warmup_count = 1

    def _extract_template(
        self,
        image: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray | None:
        half = self.template_half_size
        center_x, center_y = (float(center[0]), float(center[1]))
        if (
            center_x - half < 0
            or center_y - half < 0
            or center_x + half >= image.shape[1]
            or center_y + half >= image.shape[0]
        ):
            return None
        size = (2 * half + 1, 2 * half + 1)
        return cv2.getRectSubPix(image, size, (center_x, center_y))

    def _update_template_warmup(
        self,
        gray: np.ndarray,
        centers: np.ndarray,
        valid: np.ndarray,
        confidence: np.ndarray,
    ) -> None:
        if self._template_warmup_count >= self.template_warmup_frames:
            return
        assert self._gray_template_samples is not None
        assert self._dark_template_samples is not None
        darkness = self._local_darkness(gray)
        for marker_index, center in enumerate(centers):
            if not valid[marker_index] or confidence[marker_index] < 0.70:
                continue
            gray_patch = self._extract_template(gray, center)
            dark_patch = self._extract_template(darkness, center)
            if gray_patch is None or dark_patch is None:
                continue
            self._gray_template_samples[marker_index].append(gray_patch)
            self._dark_template_samples[marker_index].append(dark_patch)
        self._template_warmup_count += 1
        if self._template_warmup_count < self.template_warmup_frames:
            return
        assert self._gray_templates is not None
        assert self._dark_templates is not None
        for marker_index in range(len(self.reference)):
            gray_samples = self._gray_template_samples[marker_index]
            dark_samples = self._dark_template_samples[marker_index]
            if gray_samples:
                self._gray_templates[marker_index] = np.median(
                    np.stack(gray_samples, axis=0),
                    axis=0,
                ).astype(np.uint8)
            if dark_samples:
                self._dark_templates[marker_index] = np.median(
                    np.stack(dark_samples, axis=0),
                    axis=0,
                ).astype(np.uint8)
        self._gray_template_samples = None
        self._dark_template_samples = None

    def _local_darkness(self, gray: np.ndarray) -> np.ndarray:
        sigma = float(getattr(self.detector, "background_sigma", 8.0))
        background = cv2.GaussianBlur(
            gray,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        return cv2.subtract(background, gray)

    def _template_proposals(
        self,
        gray: np.ndarray,
        prediction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._gray_templates is not None
        assert self._dark_templates is not None
        assert self._template_center_offsets is not None
        darkness = self._local_darkness(gray)
        points = prediction.astype(np.float32, copy=True)
        confidence = np.zeros((len(self.reference),), dtype=np.float32)
        valid = np.zeros((len(self.reference),), dtype=bool)

        for marker_index, predicted_center in enumerate(prediction):
            gray_template = self._gray_templates[marker_index]
            dark_template = self._dark_templates[marker_index]
            if gray_template is None or dark_template is None:
                continue
            dark_match = self._match_template(
                darkness,
                dark_template,
                predicted_center,
                self._template_center_offsets[marker_index],
            )
            if dark_match is None:
                continue
            dark_point, dark_score = dark_match
            if (
                dark_score >= self.primary_template_score
                and np.linalg.norm(dark_point - predicted_center)
                <= self.cue_agreement_distance
            ):
                points[marker_index] = dark_point
                confidence[marker_index] = min(max(dark_score, 0.0), 1.0)
                valid[marker_index] = bool(
                    np.linalg.norm(dark_point - self.reference[marker_index])
                    <= self.max_reference_displacement
                )
                continue

            gray_match = self._match_template(
                gray,
                gray_template,
                predicted_center,
                self._template_center_offsets[marker_index],
            )
            if gray_match is None:
                continue
            gray_point, gray_score = gray_match
            domain_distance = float(np.linalg.norm(gray_point - dark_point))
            if domain_distance <= self.domain_agreement_distance:
                gray_weight = max(gray_score - 0.2, 1e-3)
                dark_weight = max(dark_score - 0.2, 1e-3)
                points[marker_index] = (
                    gray_weight * gray_point + dark_weight * dark_point
                ) / (gray_weight + dark_weight)
                confidence[marker_index] = np.clip(
                    0.55 * min(gray_score, dark_score)
                    + 0.45 * max(gray_score, dark_score),
                    0.0,
                    1.0,
                )
            elif gray_score >= dark_score:
                points[marker_index] = gray_point
                confidence[marker_index] = np.clip(0.65 * gray_score, 0.0, 1.0)
            else:
                points[marker_index] = dark_point
                confidence[marker_index] = np.clip(0.65 * dark_score, 0.0, 1.0)

            valid[marker_index] = bool(
                confidence[marker_index] >= self.weak_template_score
                and np.linalg.norm(points[marker_index] - self.reference[marker_index])
                <= self.max_reference_displacement
            )
        return points, confidence, valid

    def _match_template(
        self,
        image: np.ndarray,
        template: np.ndarray,
        predicted_center: np.ndarray,
        center_offset: np.ndarray,
    ) -> tuple[np.ndarray, float] | None:
        half = self.template_half_size
        search = self.template_search_radius
        center_x, center_y = np.rint(predicted_center).astype(np.int32)
        x0 = max(0, int(center_x - half - search))
        y0 = max(0, int(center_y - half - search))
        x1 = min(image.shape[1], int(center_x + half + search + 1))
        y1 = min(image.shape[0], int(center_y + half + search + 1))
        search_image = image[y0:y1, x0:x1]
        if (
            search_image.shape[0] < template.shape[0]
            or search_image.shape[1] < template.shape[1]
        ):
            return None
        response = cv2.matchTemplate(
            search_image,
            template,
            cv2.TM_CCOEFF_NORMED,
            mask=self._template_mask,
        )
        _, score, _, max_location = cv2.minMaxLoc(response)
        peak_x, peak_y = max_location
        sidelobe_mask = np.ones(response.shape, dtype=bool)
        sidelobe_mask[
            max(0, peak_y - 2) : peak_y + 3,
            max(0, peak_x - 2) : peak_x + 3,
        ] = False
        if np.any(sidelobe_mask):
            second_peak = float(response[sidelobe_mask].max())
            peak_margin = float(score - second_peak)
            ambiguity_factor = min(max((peak_margin - 0.03) / 0.12, 0.0), 1.0)
            score *= ambiguity_factor
        subpixel_offset = self._quadratic_peak_offset(response, max_location)
        point = np.asarray(
            [
                x0 + max_location[0] + half + subpixel_offset[0],
                y0 + max_location[1] + half + subpixel_offset[1],
            ],
            dtype=np.float32,
        )
        point += center_offset
        return point, float(min(max(score, -1.0), 1.0))

    def _assign_unique(
        self,
        prediction: np.ndarray,
        candidates: np.ndarray,
        candidate_quality: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exclude low-quality candidates before global assignment."""

        eligible_indices = np.flatnonzero(candidate_quality >= self.minimum_quality)
        if len(eligible_indices) == 0:
            marker_count = len(self.reference)
            return (
                np.full((marker_count,), -1, dtype=np.int64),
                np.zeros((marker_count,), dtype=bool),
                np.full((marker_count,), np.inf, dtype=np.float32),
            )
        matched, valid, distance = super()._assign_unique(
            prediction,
            candidates[eligible_indices],
            candidate_quality[eligible_indices],
        )
        assigned = matched >= 0
        matched[assigned] = eligible_indices[matched[assigned]]
        return matched, valid, distance

    @staticmethod
    def _quadratic_peak_offset(
        response: np.ndarray,
        max_location: tuple[int, int],
    ) -> np.ndarray:
        peak_x, peak_y = max_location
        center = float(response[peak_y, peak_x])
        offset = np.zeros((2,), dtype=np.float32)
        if 0 < peak_x < response.shape[1] - 1:
            left = float(response[peak_y, peak_x - 1])
            right = float(response[peak_y, peak_x + 1])
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-6:
                offset[0] = np.clip(0.5 * (left - right) / denominator, -1.0, 1.0)
        if 0 < peak_y < response.shape[0] - 1:
            top = float(response[peak_y - 1, peak_x])
            bottom = float(response[peak_y + 1, peak_x])
            denominator = top - 2.0 * center + bottom
            if abs(denominator) > 1e-6:
                offset[1] = np.clip(0.5 * (top - bottom) / denominator, -1.0, 1.0)
        return offset


__all__ = [
    "AdaptiveRingMarkerDetector",
    "HighReliabilityGridMarkerTracker",
    "MarkerDetection",
    "MarkerTrackingResult",
    "RobustGridMarkerTracker",
]
