import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import config

@dataclass
class CoarseClusterResult:
    roi_points: np.ndarray          # (M, 5): points inside ROI
    counts: np.ndarray              # (y_grid_num,): point count per X-bin
    coarse_x: Optional[Tuple[float, float]] = None # (min_x, max_x)
    cluster_span: float = 0.0       # max_x - min_x in meters
    status: str = "no_cluster"

class ShortEdgeDetector:
    """
    TM16 Short Edge Detection Algorithm Prototype (Phase 1: ROI & Coarse Clustering)
    Faithfully reproduces C++ detectShortEdges() from short_edge_detector.hpp
    Supports dynamic parameter tuning at runtime.
    """
    def __init__(self,
                 roi_config: Optional[dict] = None,
                 ground_z_threshold: Optional[float] = None,
                 y_grid_num: Optional[int] = None,
                 cluster_min_points: Optional[int] = None,
                 cluster_min_span: Optional[int] = None,
                 cluster_max_gap: Optional[int] = None):
        if ground_z_threshold is None:
            ground_z_threshold = config.ROI_CONFIG.get('ground_z_threshold', 0.20)
        c_cfg = getattr(config, 'COARSE_CLUSTER_CONFIG', {})
        if y_grid_num is None:
            y_grid_num = c_cfg.get('y_grid_num', 200)
        if cluster_min_points is None:
            cluster_min_points = c_cfg.get('cluster_min_points', 3)
        if cluster_min_span is None:
            cluster_min_span = c_cfg.get('cluster_min_span', 6)
        if cluster_max_gap is None:
            cluster_max_gap = c_cfg.get('cluster_max_gap', 3)
        self.roi = dict(roi_config if roi_config is not None else config.ROI_CONFIG)
        self.ground_z_threshold = ground_z_threshold
        self.y_grid_num = y_grid_num
        self.cluster_min_points = cluster_min_points
        self.cluster_min_span = cluster_min_span
        self.cluster_max_gap = cluster_max_gap

    def filter_roi(self, points: np.ndarray) -> np.ndarray:
        """
        Filters points strictly adhering to shortEdgeRoi():
        std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
        p.x > c.roi_x_min && p.x < c.roi_x_max &&
        p.y > c.roi_y_min && p.y < c.roi_y_max &&
        p.z > c.roi_z_min && p.z < c.roi_z_max &&
        p.z > c.ground_z_threshold
        """
        if len(points) == 0:
            return np.empty((0, 5), dtype=np.float32)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        valid_mask = (
            np.isfinite(x) & np.isfinite(y) & np.isfinite(z) &
            (x > self.roi['x_min']) & (x < self.roi['x_max']) &
            (y > self.roi['y_min']) & (y < self.roi['y_max']) &
            (z > self.roi['z_min']) & (z < self.roi['z_max']) &
            (z > self.ground_z_threshold)
        )
        return points[valid_mask]

    def detect_coarse(self, points: np.ndarray) -> CoarseClusterResult:
        """
        Runs Step 1: ROI filtering and Step 2: X-axis coarse grid clustering
        """
        roi_pts = self.filter_roi(points)
        num_bins = max(1, int(self.y_grid_num))
        counts = np.zeros(num_bins, dtype=int)

        if len(roi_pts) == 0:
            return CoarseClusterResult(
                roi_points=roi_pts,
                counts=counts,
                coarse_x=None,
                cluster_span=0.0,
                status="no_roi_points"
            )

        x_span = self.roi['x_max'] - self.roi['x_min']
        if x_span <= 0:
            return CoarseClusterResult(
                roi_points=roi_pts,
                counts=counts,
                coarse_x=None,
                cluster_span=0.0,
                status="invalid_x_range"
            )

        coarse_step = x_span / num_bins

        # Compute bin for each point
        bins = np.floor((roi_pts[:, 0] - self.roi['x_min']) / coarse_step).astype(int)
        valid_bins = (bins >= 0) & (bins < num_bins)
        np.add.at(counts, bins[valid_bins], 1)

        # 1D clustering along X-axis
        clusters: List[Tuple[int, int]] = []
        start = -1
        last = -1
        gap = 0

        for i in range(num_bins):
            if counts[i] >= self.cluster_min_points:
                if start < 0:
                    start = i
                last = i
                gap = 0
            elif start >= 0:
                gap += 1
                if gap > self.cluster_max_gap:
                    if (last - start + 1) >= self.cluster_min_span:
                        clusters.append((start, last))
                    start = -1
                    last = -1
                    gap = 0

        if start >= 0 and (last - start + 1) >= self.cluster_min_span:
            clusters.append((start, last))

        if not clusters:
            return CoarseClusterResult(
                roi_points=roi_pts,
                counts=counts,
                coarse_x=None,
                cluster_span=0.0,
                status="no_clusters"
            )

        # Choose cluster with the widest span
        best_cluster = max(clusters, key=lambda c: c[1] - c[0])
        coarse_min_x = self.roi['x_min'] + best_cluster[0] * coarse_step
        coarse_max_x = self.roi['x_min'] + (best_cluster[1] + 1) * coarse_step
        span_m = coarse_max_x - coarse_min_x

        return CoarseClusterResult(
            roi_points=roi_pts,
            counts=counts,
            coarse_x=(coarse_min_x, coarse_max_x),
            cluster_span=span_m,
            status="ok"
        )
