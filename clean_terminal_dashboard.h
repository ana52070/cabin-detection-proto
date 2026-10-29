#pragma once

// 终端实时仪表盘：单屏覆盖式刷新，取代原本每帧滚动的零散日志。
// Snapshot 由 main 循环各填充点写入；读写双方共用调用方传入的同一把 mutex，
// 持锁只做字段拷贝/渲染，不碰其它锁。

#include <unistd.h>   // isatty, STDOUT_FILENO

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace dash {

// 单台雷达状态，字段对应 lidar::RadarManager::DiagnosticData 的并行数组
struct RadarRow {
    bool online = false;
    uint64_t frames = 0;
    double fps = 0.0;
    size_t points = 0;
    std::string info;
};

// 仪表盘快照：所有待显示数据（外部加锁保护）
struct Snapshot {
    // ---- 系统 ----
    std::chrono::steady_clock::time_point start_time{}; // main 启动时刻，uptime 据此计算
    std::string config_path;                            // config.json 路径

    // ---- 雷达 ----
    std::vector<RadarRow> radars;

    // ---- 车辆 ----
    bool vehicle_detected = false;
    double vehicle_min_x = 0.0;   // 车尾位置（小 X 端，m）
    double vehicle_max_x = 0.0;   // 车头位置（大 X 端，m）
    double vehicle_length = 0.0;  // 车长(m)
    double height = 0.0;          // 前车帮高(m)
    double rear_height = 0.0;     // 后车帮高(m)
    double level_z = 0.0;         // 物位计 Z(m)
    bool level_z_valid = false;   // feature_level_meter 开启且已取值

    // ---- PLC ----
    bool plc_connected = false;
    bool position_signed = true;  // 支持负数位置
    uint16_t reg_min_x_raw = 0;   // 最近一次写入的原始寄存器值
    uint16_t reg_max_x_raw = 0;
    uint16_t reg_height_raw = 0;
    uint16_t reg_rear_height_raw = 0; // 后车帮独立高度，毫米
    double reg_min_x_m = 0.0;     // 对应米值
    double reg_max_x_m = 0.0;
    double reg_height_m = 0.0;
    uint16_t heartbeat_raw = 0;   // 最近一次读取的心跳/指令寄存器原始值
    std::string command_text;     // 解析出的指令文本（无车/前移/后移/停车/驶离）
    uint64_t plc_write_count = 0; // PLC 写入累计成功次数
    uint64_t plc_write_fail = 0;  // PLC 写入累计失败次数

    // ---- 车牌相机 (LPR) ----
    bool lpr_enabled = false;                  // feature_lpr
    int lpr_mode = 0;                          // 0=FTP 1=HIK
    std::vector<std::pair<std::string, bool>> lpr_cams;  // (相机IP, 是否在线)
    std::string lpr_last_plate;                // 最近识别车牌（显示文本，无则为空）
    std::string lpr_last_cam;                  // 最近识别来源相机IP
    std::string lpr_last_time;                 // 最近识别时刻 HH:MM:SS
    uint64_t lpr_plate_count = 0;              // 累计识别次数（g_lpr_recv_count 拷贝）

    // ---- MySQL ----
    bool mysql_ok = false;
};

// 数值显示：无效数据统一显示 '-'
inline std::string fmtNum(double v, bool valid, const char* suffix = "") {
    if (!valid) return "-";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f%s", v, suffix);
    return buf;
}

// uptime 显示为 h:mm:ss
inline std::string fmtUptime(const Snapshot& s) {
    if (s.start_time.time_since_epoch().count() <= 0) return "-";
    long long secs = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - s.start_time).count());
    if (secs < 0) secs = 0;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%lld:%02lld:%02lld",
                  secs / 3600, (secs % 3600) / 60, secs % 60);
    return buf;
}

// 渲染整个界面（调用方需持有 mtx，纯内存操作，持锁时间极短）
inline std::string render(const Snapshot& s) {
    std::ostringstream os;

    os << "=================== mlscan 实时仪表盘 ===================\n";
    os << "系统   uptime : " << fmtUptime(s) << "\n";
    os << "       config : " << (s.config_path.empty() ? std::string("-") : s.config_path) << "\n\n";

    os << "------------------------- 车辆 --------------------------\n";
    os << "状态         : " << (s.vehicle_detected ? "检测到" : "-") << "\n";
    os << "车头 max_x（大X端）: " << fmtNum(s.vehicle_max_x, s.vehicle_detected, " m") << "\n";
    os << "车尾 min_x（小X端）: " << fmtNum(s.vehicle_min_x, s.vehicle_detected, " m") << "\n";
    os << "前车帮高       : " << fmtNum(s.height, s.vehicle_detected, " m") << "\n";
    os << "后车帮高      : " << fmtNum(s.rear_height, s.vehicle_detected, " m") << "\n";
    os << "车长         : " << fmtNum(s.vehicle_length, s.vehicle_detected, " m") << "\n";
    os << "物位计 Z     : " << fmtNum(s.level_z, s.level_z_valid, " m") << "\n\n";

    os << "------------------------- 雷达 --------------------------\n";
    if (s.radars.empty()) {
        os << "-\n";
    } else {
        for (size_t i = 0; i < s.radars.size(); ++i) {
            const RadarRow& r = s.radars[i];
            char fbuf[32];
            std::snprintf(fbuf, sizeof(fbuf), "%.1f", r.fps);
            os << "#" << i << "  " << (r.info.empty() ? std::string("-") : r.info)
               << "  " << (r.online ? "ONLINE " : "OFFLINE")
               << "  帧数=" << r.frames
               << "  fps=" << fbuf
               << "  点数=" << r.points << "\n";
        }
    }
    os << "\n";

    os << "-------------------------- PLC ---------------------------\n";
    os << "状态         : " << (s.plc_connected ? "已连接" : "未连接") << "\n";
    if (s.command_text.empty()) {
        os << "心跳/指令    : -\n";
    } else {
        char hbuf[32];
        std::snprintf(hbuf, sizeof(hbuf), "0x%04X", static_cast<unsigned>(s.heartbeat_raw));
        os << "心跳/指令    : " << hbuf << "  " << s.command_text << "\n";
    }
    os << "写入 车头位置 : " << static_cast<int>(static_cast<int16_t>(s.reg_min_x_raw)) << " mm\n";
    os << "写入 车尾位置 : " << static_cast<int>(static_cast<int16_t>(s.reg_max_x_raw)) << " mm\n";
    os << "写入 前车帮高 : " << s.reg_height_raw << " mm\n";
    os << "写入 后车帮高 : " << s.reg_rear_height_raw << " mm\n";
    os << "累计写入     : 成功 " << s.plc_write_count << "  失败 " << s.plc_write_fail << "\n\n";

    os << "-------------------- 车牌相机 (LPR) ---------------------\n";
    if (!s.lpr_enabled) {
        os << "-\n\n";
    } else {
        os << "模式         : " << (s.lpr_mode == 1 ? "HIK 海康直连" : "FTP 文件监控") << "\n";
        if (s.lpr_cams.empty()) {
            os << "相机         : -\n";
        } else {
            for (size_t i = 0; i < s.lpr_cams.size(); ++i) {
                const std::pair<std::string, bool>& cam = s.lpr_cams[i];
                os << "#" << i << "  " << (cam.first.empty() ? std::string("-") : cam.first)
                   << "  " << (cam.second ? "ONLINE " : "OFFLINE") << "\n";
            }
        }
        if (s.lpr_last_plate.empty()) {
            os << "最近车牌     : -\n";
        } else {
            os << "最近车牌     : " << s.lpr_last_plate
               << " @ " << (s.lpr_last_cam.empty() ? std::string("-") : s.lpr_last_cam)
               << " " << (s.lpr_last_time.empty() ? std::string("-") : s.lpr_last_time) << "\n";
        }
        os << "识别次数     : " << s.lpr_plate_count << "\n\n";
    }

    os << "------------------------- MySQL --------------------------\n";
    os << "状态         : " << (s.mysql_ok ? "可用" : "不可用") << "\n";

    return os.str();
}

// 仪表盘主循环：isatty 为真时每秒清屏重绘；为假直接返回
inline void runDashboard(Snapshot& snap, std::mutex& mtx) {
    // 服务模式/重定向到文件：不输出任何东西
    if (!isatty(STDOUT_FILENO)) return;

    std::string frame;
    while (true) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            frame = render(snap);
        }
        // 直接写 stdout，绕开 std::cout 的小时日志 tee，不把转义序列写进日志文件
        std::fputs("\033[2J\033[H", stdout);
        std::fwrite(frame.data(), 1, frame.size(), stdout);
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

} // namespace dash
