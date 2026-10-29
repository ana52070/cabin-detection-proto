#pragma once
#include "frame_quality.hpp"
#include "short_edge_detector.hpp"
#include "cabin_kalman.hpp"
#include <deque>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace verify {

struct HeightFilterConfig {
    unsigned window = 5;
    double reset_gap_ms = 1000.0, max_window_ms = 2000.0;
    double max_angle_step_deg = 5.0, min_angular_coverage = 0.80;
    size_t min_wall_points = 5, min_top_votes = 3, min_vote_margin = 1;
    double min_vote_fraction = 0.25;
    double position_jump_base_m = 1.0, max_position_speed_mps = 10.0;
    double z_min_valid = 4.0, z_max_valid = 8.5; // 与 Python 端一致的真实车帮高度区间

    // 卡尔曼滤波开关制与参数可调
    bool enable_kalman = true;
    size_t max_coast_frames = 5;
    double max_coast_time_ms = 600.0;
    double pos_process_noise_q = 0.05;
    double pos_measure_noise_r = 0.15;
    double height_process_noise_q = 0.001;
    double height_measure_noise_r = 0.03;
    double height_init_cov = 0.10;
    double height_gate_limit = 0.25;
};

struct FilteredEnd {
    bool qualified = false;
    std::optional<double> filtered;
    std::string reason = "unavailable", filter_status = "unavailable";
    size_t window_points = 0;
};

struct FilteredHeight { FilteredEnd low, high; };

// 最终综合高度裁决逻辑：两端均有滤波值且高度差绝对值 <= max_difference 时取均值
inline std::optional<double> finalFilteredHeight(const FilteredHeight& ends, double max_difference) {
    if (!std::isfinite(max_difference) || max_difference < 0)
        throw std::runtime_error("最终高度一致性阈值无效");

    if (!ends.low.filtered || !ends.high.filtered)
        return {};

    const double low = *ends.low.filtered;
    const double high = *ends.high.filtered;

    if (!std::isfinite(low) || !std::isfinite(high) || std::abs(low - high) > max_difference)
        return {};

    return (low + high) * 0.5;
}

// 鲁棒的时序高度滤波器（支持因果滑动窗口与卡尔曼平滑/外推双模式切换）
class HeightQualityFilter {
public:
    explicit HeightQualityFilter(HeightFilterConfig config)
        : config_(config),
          kf_low_(makeKalmanConfig(config), config.z_min_valid, config.z_max_valid),
          kf_high_(makeKalmanConfig(config), config.z_min_valid, config.z_max_valid) {}

    FilteredHeight feed(const ShortEdgeResult& r, const FrameQuality& q, double time_ms) {
        if (!std::isfinite(time_ms) || time_ms < last_time_)
            throw std::runtime_error("滤波时间必须有限且递增");

        if (last_time_ > 0 && (time_ms - last_time_ > config_.reset_gap_ms)) {
            kf_low_.reset();
            kf_high_.reset();
        }
        last_time_ = time_ms;

        return {
            processEnd(r, q, time_ms, true, low_, kf_low_),
            processEnd(r, q, time_ms, false, high_, kf_high_)
        };
    }

private:
    static KalmanConfig makeKalmanConfig(const HeightFilterConfig& c) {
        KalmanConfig kc;
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

    struct State {
        std::deque<std::pair<double, double>> values; // (timestamp_ms, height_z)
        std::optional<double> position, last_good;
    } low_, high_;

    HeightFilterConfig config_;
    double last_time_ = -std::numeric_limits<double>::infinity();
    SingleEndHeightKalman kf_low_, kf_high_;

    FilteredEnd processEnd(const ShortEdgeResult& r, const FrameQuality& q, double now,
                           bool low, State& state, SingleEndHeightKalman& kf) {
        (void)q;
        FilteredEnd out;

        const auto reject = [&](const std::string& reason) {
            out.reason = reason;
            out.filter_status = "rejected";
            return out;
        };

        const auto top = low ? r.low_top : r.high_top;
        const bool has_meas = r.position.has_value() && top.has_value() && std::isfinite(*top) &&
                              (*top >= config_.z_min_valid && *top <= config_.z_max_valid);

        if (config_.enable_kalman) {
            // 卡尔曼模式：实测更新或缺失外推
            std::optional<double> kf_res;
            if (has_meas) {
                kf_res = kf.update(*top, now);
                out.reason = "accepted";
                out.filter_status = "ready";
                out.qualified = true;
            } else {
                kf_res = kf.coast(now);
                if (kf_res) {
                    out.reason = "kalman_coasting";
                    out.filter_status = "ready";
                    out.qualified = true;
                } else {
                    return reject("unavailable");
                }
            }
            out.filtered = kf_res;
            out.window_points = 5;
            return out;
        }

        // 基线中值滤波模式
        if (state.last_good && (now - *state.last_good > config_.reset_gap_ms)) {
            state.values.clear();
        }
        while (!state.values.empty() && (now - state.values.front().first > config_.max_window_ms)) {
            state.values.pop_front();
        }

        if (!r.position) return reject("no_pair");
        const int index = low ? r.low_index : r.high_index;
        if (index < 0 || static_cast<size_t>(index) >= r.candidates.size()) return reject("no_pair");
        if (!top || !std::isfinite(*top)) return reject("top_missing");
        if (*top < config_.z_min_valid || *top > config_.z_max_valid) return reject("height_out_of_range");

        state.last_good = now;
        state.values.emplace_back(now, *top);
        while (state.values.size() > config_.window) {
            state.values.pop_front();
        }

        out.qualified = true;
        out.reason = "accepted";
        out.window_points = state.values.size();

        std::vector<double> vals;
        vals.reserve(state.values.size());
        for (const auto& v : state.values) vals.push_back(v.second);
        std::sort(vals.begin(), vals.end());

        if (state.values.size() >= 3) {
            const size_t mid = vals.size() / 2;
            out.filtered = (vals.size() % 2 == 1) ? vals[mid] : (vals[mid - 1] + vals[mid]) * 0.5;
            out.filter_status = "ready";
        } else {
            double sum = 0.0;
            for (double v : vals) sum += v;
            out.filtered = sum / vals.size();
            out.filter_status = "warming_up";
        }
        return out;
    }
};

} // namespace verify
