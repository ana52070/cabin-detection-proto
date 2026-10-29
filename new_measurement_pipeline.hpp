#pragma once
#include "lidar/tm16/measurement_config.hpp"
#include "lidar/tm16/cabin_kalman.hpp"

namespace lidar::tm16_measurement {
struct Measurement {
    verify::ShortEdgeResult edges;
    verify::FilteredHeight filtered;
    std::optional<double> final_height;
    std::string final_status;
};

class Pipeline {
public:
    explicit Pipeline(const Json& json)
        : base_(loadAlgorithmConfig(json)), short_(loadShortEdgeConfig(json, base_)),
          filter_config_(loadHeightFilterConfig(json)), filter_(filter_config_),
          pos_kf_(makeKalmanConfig(filter_config_)) {}

    Measurement process(const lidar::UnifiedFrame& cloud,
                        const verify::FrameQuality& quality, double time_ms) {
        Measurement out;
        // 步骤 1：空间几何解算 —— 识别短边前后挡板与原始车高
        out.edges = verify::detectShortEdges(cloud, base_, short_);

        // 步骤 2：位置卡尔曼滤波与短时外推保护（优化点云缺失帧小漏检）
        if (filter_config_.enable_kalman) {
            if (out.edges.position) {
                pos_kf_.update(out.edges.position->min_x, out.edges.position->max_x, time_ms);
            } else {
                auto coasted = pos_kf_.coast(time_ms);
                if (coasted) {
                    out.edges.position = lidar::XRange{coasted->x_rear, coasted->x_front};
                    out.edges.length = coasted->length;
                    out.edges.status = "kalman_coasting";

                    // 局部点云高度提取（若当前帧有点云）
                    std::vector<double> pf_zs, pr_zs;
                    for (const auto& p : cloud.points) {
                        if (p.z > base_.ground_z_threshold && p.z >= base_.roi_z_min && p.z <= base_.roi_z_max) {
                            if (std::abs(p.x - coasted->x_front) <= 0.25) pf_zs.push_back(p.z);
                            if (std::abs(p.x - coasted->x_rear) <= 0.25) pr_zs.push_back(p.z);
                        }
                    }
                    if (pf_zs.size() >= 5) {
                        out.edges.high_top = verify::edgePercentile(pf_zs, 0.98);
                    }
                    if (pr_zs.size() >= 5) {
                        out.edges.low_top = verify::edgePercentile(pr_zs, 0.98);
                    }
                }
            }
        }

        // 步骤 3：时序高度滤波（实测推进或卡尔曼平滑外推）
        out.filtered = filter_.feed(out.edges, quality, time_ms);

        // 步骤 4：综合一致性决策 —— 检查两端高度差是否超标，输出最终车帮高
        out.final_height = verify::finalFilteredHeight(out.filtered, short_.max_top_difference);
        out.final_status = out.final_height ? "ready" :
            (!out.filtered.low.filtered || !out.filtered.high.filtered) ? "missing_end" : "inconsistent";
        return out;
    }

    void reset() {
        filter_ = verify::HeightQualityFilter(filter_config_);
        pos_kf_.reset();
    }
    unsigned window() const { return filter_config_.window; }

private:
    static verify::KalmanConfig makeKalmanConfig(const verify::HeightFilterConfig& c) {
        verify::KalmanConfig kc;
        kc.enabled = c.enable_kalman;
        kc.max_coast_frames = c.max_coast_frames;
        kc.max_coast_time_ms = c.max_coast_time_ms;
        kc.pos_process_noise_q = c.pos_process_noise_q;
        kc.pos_measure_noise_r = c.pos_measure_noise_r;
        kc.height_process_noise_q = c.height_process_noise_q;
        kc.height_measure_noise_r = c.height_measure_noise_r;
        kc.height_init_cov = c.height_init_cov;
        kc.height_gate_limit = c.height_gate_limit;
        return kc;
    }

    AlgorithmConfig base_;
    verify::ShortEdgeConfig short_;
    verify::HeightFilterConfig filter_config_;
    verify::HeightQualityFilter filter_;
    verify::PositionKalmanFilter pos_kf_;
};
} // namespace lidar::tm16_measurement
