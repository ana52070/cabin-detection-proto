#pragma once
#include "lidar/tm16/height_quality_filter.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <stdexcept>

namespace lidar::tm16_measurement {
using Json = nlohmann::json;
using AlgorithmConfig = lidar::TM16DetectionStrategy::Config;

inline Json object(const Json& j, const char* key) {
    if (!j.contains(key)) return Json::object();
    if (!j.at(key).is_object()) throw std::runtime_error(std::string(key) + " 必须是对象");
    return j.at(key);
}

inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

inline AlgorithmConfig loadAlgorithmConfig(const Json& j) {
    AlgorithmConfig c;
    const auto pre = object(j, "preprocessing"), roi = object(pre, "roi");
    c.ground_z_threshold = pre.value("ground_z_threshold", c.ground_z_threshold);
    c.roi_x_min = roi.value("x_min", c.roi_x_min); c.roi_x_max = roi.value("x_max", c.roi_x_max);
    c.roi_y_min = roi.value("y_min", c.roi_y_min); c.roi_y_max = roi.value("y_max", c.roi_y_max);
    c.roi_z_min = roi.value("z_min", c.roi_z_min); c.roi_z_max = roi.value("z_max", c.roi_z_max);
    const auto pos = object(j, "position_detector");
    c.y_grid_num = pos.value("y_grid_num", c.y_grid_num);
    c.cluster_min_points = pos.value("cluster_min_points", c.cluster_min_points);
    c.cluster_min_span = pos.value("cluster_min_span", c.cluster_min_span);
    c.cluster_max_gap = pos.value("cluster_max_gap", c.cluster_max_gap);
    const auto len = object(j, "length_detector"), center = object(len, "center_region");
    const auto floor = object(len, "floor_detection"), wall = object(len, "wall_detection");
    const auto range = object(len, "valid_range");
    c.length_center_x_min = center.value("x_min", (c.roi_x_min + c.roi_x_max) / 2 - 0.5);
    c.length_center_x_max = center.value("x_max", (c.roi_x_min + c.roi_x_max) / 2 + 0.5);
    c.length_center_y_min = center.value("y_min", (c.roi_y_min + c.roi_y_max) / 2 - 0.3);
    c.length_center_y_max = center.value("y_max", (c.roi_y_min + c.roi_y_max) / 2 + 0.3);
    c.length_center_min_points = center.value("min_points", 20);
    c.length_floor_z_std_threshold = floor.value("z_std_threshold", c.length_floor_z_std_threshold);
    c.length_floor_z_tolerance = floor.value("z_tolerance", c.length_floor_z_tolerance);
    c.length_search_step = wall.value("search_step", c.length_search_step);
    c.length_wall_min_points = wall.value("min_points", c.length_wall_min_points);
    c.length_wall_z_rise_threshold = wall.value("z_rise_threshold", c.length_wall_z_rise_threshold);
    c.length_min_valid = range.value("min", c.length_min_valid);
    c.length_max_valid = range.value("max", c.length_max_valid);
    const auto structure = object(j, "cabin_structure"), surface = object(j, "cabin_surface");
    const auto left = object(surface, "y_region_left"), right = object(surface, "y_region_right");
    c.cabin_y_left_min = left.value("min", c.cabin_y_left_min);
    c.cabin_y_left_max = left.value("max", c.cabin_y_left_max);
    c.cabin_y_right_min = right.value("min", c.cabin_y_right_min);
    c.cabin_y_right_max = right.value("max", c.cabin_y_right_max);
    c.structure_inner_margin = structure.value("inner_margin", 0.05);
    c.structure_min_points_per_band = structure.value("min_points_per_band", 5);
    c.structure_side_high_percentile = structure.value("side_high_percentile", 0.9);
    c.structure_inner_level_percentile = structure.value("inner_level_percentile", 0.6);
    c.structure_min_side_inner_drop = structure.value("min_side_inner_drop", 0.25);
    c.structure_min_span_bins = structure.value("min_span_bins", 8);
    c.structure_max_gap_bins = structure.value("max_gap_bins", 2);
    c.cabin_exclude_edge = surface.value("exclude_edge", c.cabin_exclude_edge);
    c.cabin_height_z_max = surface.value("z_max", c.cabin_height_z_max);
    c.cabin_height_bin_min_points = surface.value("height_bin_min_points", c.cabin_height_bin_min_points);
    c.cabin_height_min_valid_bins = surface.value("height_min_valid_bins", c.cabin_height_min_valid_bins);
    c.cabin_height_side_percentile = surface.value("height_side_percentile", c.cabin_height_side_percentile);
    c.cabin_height_mode_bin_size = surface.value("height_mode_bin_size", c.cabin_height_mode_bin_size);
    c.cabin_height_mode_min_votes = surface.value("height_mode_min_votes", c.cabin_height_mode_min_votes);
    require(c.roi_x_min < c.roi_x_max && c.roi_y_min < c.roi_y_max && c.roi_z_min < c.roi_z_max,
        "ROI 每个 min 必须小于 max");
    require(c.y_grid_num > 0 && c.y_grid_num <= 200000 && c.cluster_min_points > 0 &&
        c.cluster_min_span > 0 && c.cluster_max_gap >= 0, "位置网格参数无效");
    require(c.length_search_step > 0 && (c.roi_x_max - c.roi_x_min + c.roi_y_max - c.roi_y_min) /
        c.length_search_step <= 1000000, "搜索步长必须大于零，且搜索步数不能超过一百万");
    require(c.length_center_x_min < c.length_center_x_max && c.length_center_y_min < c.length_center_y_max &&
        c.length_center_min_points > 0 && c.length_wall_min_points > 0 &&
        c.length_min_valid > 0 && c.length_min_valid < c.length_max_valid, "长度参数无效");
    require(c.cabin_y_left_min < c.cabin_y_left_max && c.cabin_y_left_max < c.cabin_y_right_min &&
        c.cabin_y_right_min < c.cabin_y_right_max, "左右车帮 Y 区间无效");
    require(c.cabin_height_mode_bin_size >= 0.000001 && c.cabin_height_min_valid_bins > 0 &&
        c.cabin_height_bin_min_points > 0 && c.cabin_height_mode_min_votes > 0 &&
        c.structure_min_span_bins > 0 && c.structure_max_gap_bins >= 0 && c.cabin_exclude_edge >= 0,
        "车厢结构或高度参数无效");
    return c;
}

inline verify::ShortEdgeConfig loadShortEdgeConfig(const Json& json,const AlgorithmConfig& base) {
    verify::ShortEdgeConfig c;
    const auto j=object(json,"short_edge_detector");
    c.x_bin_size=j.value("x_bin_size",c.x_bin_size);
    c.y_bin_size=j.value("y_bin_size",c.y_bin_size);
    c.y_min=j.value("y_min",base.roi_y_min);c.y_max=j.value("y_max",base.roi_y_max);
    c.floor_hist_bin=j.value("floor_hist_bin",c.floor_hist_bin);
    c.floor_min_points=j.value("floor_min_points",c.floor_min_points);
    c.wall_min_points=j.value("wall_min_points",c.wall_min_points);
    c.wall_min_y_bins=j.value("wall_min_y_bins",c.wall_min_y_bins);
    c.wall_min_y_span=j.value("wall_min_y_span",c.wall_min_y_span);
    c.wall_min_z_span=j.value("wall_min_z_span",c.wall_min_z_span);
    c.wall_max_gap_bins=j.value("wall_max_gap_bins",c.wall_max_gap_bins);
    c.wall_max_x_span=j.value("wall_max_x_span",c.wall_max_x_span);
    c.end_search_fraction=j.value("end_search_fraction",c.end_search_fraction);
    c.min_pair_support_ratio=j.value("min_pair_support_ratio",c.min_pair_support_ratio);
    c.top_percentile=j.value("top_percentile",c.top_percentile);
    c.top_y_bin_size=j.value("top_y_bin_size",c.top_y_bin_size);
    c.top_band_min_points=j.value("top_band_min_points",c.top_band_min_points);
    c.top_layer_min_points=j.value("top_layer_min_points",c.top_layer_min_points);
    c.top_min_votes=j.value("top_min_votes",base.cabin_height_mode_min_votes);
    c.top_mode_bin=j.value("top_mode_bin",base.cabin_height_mode_bin_size);
    c.max_top_difference=j.value("max_top_difference",c.max_top_difference);
    c.min_board_height=j.value("min_board_height",c.min_board_height);
    c.max_empty_gap=j.value("max_empty_gap",c.max_empty_gap);
    c.max_base_jump=j.value("max_base_jump",c.max_base_jump);
    c.enable_z_uniformity=j.value("enable_z_uniformity",c.enable_z_uniformity);
    verify::validateShortEdgeConfig(c);
    return c;
}

inline verify::HeightFilterConfig loadHeightFilterConfig(const Json& json) {
    const auto j = object(json, "height_filter");
    verify::HeightFilterConfig c;
#define TM16_FILTER_VALUE(name) c.name = j.value(#name, c.name)
    TM16_FILTER_VALUE(window);
    TM16_FILTER_VALUE(reset_gap_ms);
    TM16_FILTER_VALUE(max_window_ms);
    TM16_FILTER_VALUE(max_angle_step_deg);
    TM16_FILTER_VALUE(min_angular_coverage);
    TM16_FILTER_VALUE(min_wall_points);
    TM16_FILTER_VALUE(min_top_votes);
    TM16_FILTER_VALUE(min_vote_margin);
    TM16_FILTER_VALUE(min_vote_fraction);
    TM16_FILTER_VALUE(position_jump_base_m);
    TM16_FILTER_VALUE(max_position_speed_mps);
    TM16_FILTER_VALUE(enable_kalman);
    TM16_FILTER_VALUE(max_coast_frames);
    TM16_FILTER_VALUE(max_coast_time_ms);
    TM16_FILTER_VALUE(pos_process_noise_q);
    TM16_FILTER_VALUE(pos_measure_noise_r);
    TM16_FILTER_VALUE(height_process_noise_q);
    TM16_FILTER_VALUE(height_measure_noise_r);
    TM16_FILTER_VALUE(height_init_cov);
    TM16_FILTER_VALUE(height_gate_limit);
#undef TM16_FILTER_VALUE
    return c;
}
} // namespace lidar::tm16_measurement
