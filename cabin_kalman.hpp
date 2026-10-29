#pragma once

#include <cmath>
#include <optional>
#include <tuple>
#include <algorithm>
#include <stdexcept>

namespace verify {

struct KalmanConfig {
    bool enabled = true;
    size_t max_coast_frames = 5;
    double max_coast_time_ms = 600.0;
    double pos_process_noise_q = 0.05;
    double pos_measure_noise_r = 0.15;
    double height_process_noise_q = 0.001;
    double height_measure_noise_r = 0.03;
    double height_init_cov = 0.10;
    double height_gate_limit = 0.25; // 超过 0.25m 突变视作击中底板噪点，拒绝粗差更新
};

// 1D 车厢位置恒速卡尔曼滤波
// 状态 X = [x_center, v_center, length]
class PositionKalmanFilter {
public:
    explicit PositionKalmanFilter(KalmanConfig config = KalmanConfig()) : cfg_(config) {
        reset();
    }

    void reset() {
        is_initialized_ = false;
        last_time_ms_ = 0.0;
        last_meas_time_ms_ = 0.0;
        coast_frames_ = 0;
        x_center_ = 0.0;
        v_ = 0.0;
        length_ = 0.0;
        p00_ = 0.1; p01_ = 0.0; p02_ = 0.0;
        p10_ = 0.0; p11_ = 1.0; p12_ = 0.0;
        p20_ = 0.0; p21_ = 0.0; p22_ = 0.1;
    }

    bool isInitialized() const { return is_initialized_; }

    void update(double x_front, double x_rear, double timestamp_ms) {
        const double xc_meas = 0.5 * (x_front + x_rear);
        const double len_meas = std::abs(x_front - x_rear);

        if (!is_initialized_) {
            x_center_ = xc_meas;
            v_ = 0.0;
            length_ = len_meas;
            last_time_ms_ = timestamp_ms;
            last_meas_time_ms_ = timestamp_ms;
            coast_frames_ = 0;
            is_initialized_ = true;
            return;
        }

        const double dt = std::max(1e-4, (timestamp_ms - last_time_ms_) / 1000.0);
        predictStep(dt);
        last_time_ms_ = timestamp_ms;

        // 测量更新: H = [[1, 0, 0], [0, 0, 1]]
        // 测量噪声 R
        const double r = cfg_.pos_measure_noise_r;
        const double r0 = r * r;
        const double r1 = (r * 1.5) * (r * 1.5);

        // y = z - H * x
        const double y0 = xc_meas - x_center_;
        const double y1 = len_meas - length_;

        // S = H * P * H^T + R
        const double s00 = p00_ + r0;
        const double s01 = p02_;
        const double s10 = p20_;
        const double s11 = p22_ + r1;
        const double detS = s00 * s11 - s01 * s10;
        if (detS > 1e-9) {
            const double invS00 = s11 / detS;
            const double invS01 = -s01 / detS;
            const double invS10 = -s10 / detS;
            const double invS11 = s00 / detS;

            // K = P * H^T * invS
            const double k00 = p00_ * invS00 + p02_ * invS10;
            const double k01 = p00_ * invS01 + p02_ * invS11;
            const double k10 = p10_ * invS00 + p12_ * invS10;
            const double k11 = p10_ * invS01 + p12_ * invS11;
            const double k20 = p20_ * invS00 + p22_ * invS10;
            const double k21 = p20_ * invS01 + p22_ * invS11;

            x_center_ += k00 * y0 + k01 * y1;
            v_        += k10 * y0 + k11 * y1;
            length_   += k20 * y0 + k21 * y1;

            // P = (I - K * H) * P
            const double new_p00 = (1 - k00) * p00_ - k01 * p20_;
            const double new_p01 = (1 - k00) * p01_ - k01 * p21_;
            const double new_p02 = (1 - k00) * p02_ - k01 * p22_;

            const double new_p10 = -k10 * p00_ + p10_ - k11 * p20_;
            const double new_p11 = -k10 * p01_ + p11_ - k11 * p21_;
            const double new_p12 = -k10 * p02_ + p12_ - k11 * p22_;

            const double new_p20 = -k20 * p00_ + (1 - k21) * p20_;
            const double new_p21 = -k20 * p01_ + (1 - k21) * p21_;
            const double new_p22 = -k20 * p02_ + (1 - k21) * p22_;

            p00_ = new_p00; p01_ = new_p01; p02_ = new_p02;
            p10_ = new_p10; p11_ = new_p11; p12_ = new_p12;
            p20_ = new_p20; p21_ = new_p21; p22_ = new_p22;
        }

        last_meas_time_ms_ = timestamp_ms;
        coast_frames_ = 0;
    }

    struct CoastResult {
        double x_front;
        double x_rear;
        double length;
    };

    std::optional<CoastResult> coast(double timestamp_ms) {
        if (!is_initialized_) return std::nullopt;

        coast_frames_++;
        const double elapsed_ms = timestamp_ms - last_meas_time_ms_;
        if (coast_frames_ > cfg_.max_coast_frames || elapsed_ms > cfg_.max_coast_time_ms) {
            reset();
            return std::nullopt;
        }

        const double dt = std::max(1e-4, (timestamp_ms - last_time_ms_) / 1000.0);
        predictStep(dt);
        last_time_ms_ = timestamp_ms;

        CoastResult res;
        res.x_front = x_center_ + 0.5 * length_;
        res.x_rear  = x_center_ - 0.5 * length_;
        res.length  = length_;
        return res;
    }

private:
    void predictStep(double dt) {
        x_center_ += v_ * dt;
        const double fp00 = p00_ + dt * p10_;
        const double fp01 = p01_ + dt * p11_;
        const double fp02 = p02_ + dt * p12_;
        const double fp10 = p10_;
        const double fp11 = p11_;
        const double fp12 = p12_;
        const double fp20 = p20_;
        const double fp21 = p21_;
        const double fp22 = p22_;

        const double q_pos = cfg_.pos_process_noise_q;
        const double q00 = 0.25 * (dt * dt * dt * dt) * q_pos;
        const double q01 = 0.5  * (dt * dt * dt) * q_pos;
        const double q11 = (dt * dt) * q_pos;
        const double q22 = 1e-4 * dt;

        p00_ = fp00 + fp01 * dt + q00;
        p01_ = fp01 + q01;
        p02_ = fp02;
        p10_ = fp10 + fp11 * dt + q01;
        p11_ = fp11 + q11;
        p12_ = fp12;
        p20_ = fp20 + fp21 * dt;
        p21_ = fp21;
        p22_ = fp22 + q22;
    }

    KalmanConfig cfg_;
    bool is_initialized_ = false;
    double last_time_ms_ = 0.0;
    double last_meas_time_ms_ = 0.0;
    size_t coast_frames_ = 0;
    double x_center_ = 0.0, v_ = 0.0, length_ = 0.0;
    double p00_, p01_, p02_;
    double p10_, p11_, p12_;
    double p20_, p21_, p22_;
};

// 单端车帮高度标量卡尔曼滤波器（带残差门控）
class SingleEndHeightKalman {
public:
    SingleEndHeightKalman(KalmanConfig config = KalmanConfig(), double z_min = 4.0, double z_max = 7.5)
        : cfg_(config), z_min_(z_min), z_max_(z_max) {
        reset();
    }

    void reset() {
        is_initialized_ = false;
        z_ = 0.0;
        p_ = cfg_.height_init_cov;
        last_time_ms_ = 0.0;
        last_meas_time_ms_ = 0.0;
        coast_frames_ = 0;
        outlier_count_ = 0;
    }

    bool isInitialized() const { return is_initialized_; }

    void predict(double timestamp_ms) {
        if (!is_initialized_) return;
        const double dt = std::max(1e-4, (timestamp_ms - (last_time_ms_ > 0 ? last_time_ms_ : timestamp_ms)) / 1000.0);
        p_ += cfg_.height_process_noise_q * dt;
        last_time_ms_ = timestamp_ms;
    }

    std::optional<double> update(double z_meas, double timestamp_ms) {
        if (z_meas < z_min_ || z_meas > z_max_ || !std::isfinite(z_meas)) {
            return coast(timestamp_ms);
        }

        if (!is_initialized_) {
            z_ = z_meas;
            p_ = cfg_.height_init_cov;
            is_initialized_ = true;
            last_time_ms_ = timestamp_ms;
            last_meas_time_ms_ = timestamp_ms;
            coast_frames_ = 0;
            outlier_count_ = 0;
            return z_;
        }

        predict(timestamp_ms);

        // 残差门控（Innovation Gating）：防止击中车内底板的突变噪点
        const double innovation = std::abs(z_meas - z_);
        if (innovation > cfg_.height_gate_limit) {
            outlier_count_++;
            if (outlier_count_ < 3) {
                // 离群噪点：使用先验预测值外推，拒绝粗差
                return coast(timestamp_ms);
            } else {
                // 连续3帧突变，重新建档对齐
                z_ = z_meas;
                p_ = cfg_.height_init_cov;
                outlier_count_ = 0;
                last_meas_time_ms_ = timestamp_ms;
                coast_frames_ = 0;
                return z_;
            }
        }

        outlier_count_ = 0;
        const double r = cfg_.height_measure_noise_r;
        const double R = r * r;
        const double K = p_ / (p_ + R);
        z_ = z_ + K * (z_meas - z_);
        p_ = (1.0 - K) * p_;

        last_meas_time_ms_ = timestamp_ms;
        coast_frames_ = 0;
        return z_;
    }

    std::optional<double> coast(double timestamp_ms) {
        if (!is_initialized_ || last_meas_time_ms_ <= 0.0) return std::nullopt;
        coast_frames_++;
        const double elapsed_ms = timestamp_ms - last_meas_time_ms_;
        if (coast_frames_ > cfg_.max_coast_frames || elapsed_ms > cfg_.max_coast_time_ms) {
            reset();
            return std::nullopt;
        }
        predict(timestamp_ms);
        return z_;
    }

private:
    KalmanConfig cfg_;
    double z_min_, z_max_;
    bool is_initialized_ = false;
    double z_ = 0.0;
    double p_ = 0.10;
    double last_time_ms_ = 0.0;
    double last_meas_time_ms_ = 0.0;
    size_t coast_frames_ = 0;
    size_t outlier_count_ = 0;
};

} // namespace verify
