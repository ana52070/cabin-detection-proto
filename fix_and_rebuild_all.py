# -*- coding: utf-8 -*-
import paramiko

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

    patch_py = """
import re

# 1. 修复 plc_measurements.hpp：解耦位置与高度，只要有X坐标即输出车头车尾位置！
plc_meas_files = [
    '/container/LD_MLScan_common_tm16_test/src/lidar/tm16/plc_measurements.hpp',
    '/container/LD_MLScan_common/src/lidar/tm16/plc_measurements.hpp',
    '/container/LD_MLScan_common_tm16_test/tools/tm16_pcap_verify/plc_measurements.hpp',
    '/container/LD_MLScan_common/tools/tm16_pcap_verify/plc_measurements.hpp'
]

new_plc_meas_content = '''#pragma once
#include <optional>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace lidar::tm16_measurement {
struct PlcOutputConfig {double lidar_x_mm=1200;};
struct PlcInput {
    std::optional<double> low_x_m,high_x_m,low_top_z_m,high_top_z_m;
    bool low_qualified=false,high_qualified=false;
};
struct PlcMeasurements {
    std::optional<int32_t> front_height_mm,rear_height_mm,head_position_mm,tail_position_mm;
    unsigned validMask()const{return unsigned(bool(front_height_mm))|unsigned(bool(rear_height_mm))<<1|
        unsigned(bool(head_position_mm))<<2|unsigned(bool(tail_position_mm))<<3;}
};
// 毫米测量结果：只要车厢存在合法的 low_x 和 high_x，车头和车尾位置即始终输出！
// 绝对不因为单帧缺少水平顶面采样而将车尾位置清零。
inline PlcMeasurements makePlcMeasurements(const PlcInput& in,const PlcOutputConfig& config){
    if(!std::isfinite(config.lidar_x_mm))throw std::runtime_error("PLC的雷达X安装坐标必须有限");
    PlcMeasurements out;
    if(!in.low_x_m||!in.high_x_m||!std::isfinite(*in.low_x_m)||!std::isfinite(*in.high_x_m)||
       *in.low_x_m>=*in.high_x_m)return out;
    const auto millimeters=[](const std::optional<double>& meters,double offset)->std::optional<int32_t>{
        if(!meters||!std::isfinite(*meters))return {};
        const double rounded=std::round(*meters*1000.0+offset);
        if(!std::isfinite(rounded)||rounded<std::numeric_limits<int32_t>::min()||
           rounded>std::numeric_limits<int32_t>::max())return {};
        return static_cast<int32_t>(rounded);
    };

    // 车头位置 (大 X 端) 与车尾位置 (小 X 端) 始终加上雷达安装基准 1200mm
    out.head_position_mm = millimeters(in.high_x_m, config.lidar_x_mm);
    out.tail_position_mm = millimeters(in.low_x_m, config.lidar_x_mm);

    // 高度字段在有值时独立输出毫米值
    if(in.high_top_z_m && std::isfinite(*in.high_top_z_m)){
        out.front_height_mm = millimeters(in.high_top_z_m, 0.0);
    }
    if(in.low_top_z_m && std::isfinite(*in.low_top_z_m)){
        out.rear_height_mm = millimeters(in.low_top_z_m, 0.0);
    }
    return out;
}
}
'''

for p in plc_meas_files:
    try:
        with open(p, 'w', encoding='utf-8') as f:
            f.write(new_plc_meas_content)
        print(f"Updated {p}")
    except Exception as e:
        print(f"Error {p}: {e}")

# 2. 重写 terminal_dashboard.h 确保格式完全符合要求
dash_files = [
    '/container/LD_MLScan_common_tm16_test/src/common/terminal_dashboard.h',
    '/container/LD_MLScan_common/src/common/terminal_dashboard.h'
]

for p in dash_files:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 查找并在 Snapshot 中补全 rear_height 和 position_signed
        new_lines = []
        for line in lines:
            if 'double height = 0.0;' in line and 'rear_height' not in ''.join(lines):
                new_lines.append(line)
                new_lines.append('    double rear_height = 0.0;     // 后车帮高(m)\\n')
            elif 'bool plc_connected = false;' in line and 'bool position_signed' not in ''.join(lines):
                new_lines.append(line)
                new_lines.append('    bool position_signed = true;   // 支持负数位置\\n')
            else:
                new_lines.append(line)
        
        content = ''.join(new_lines)

        # 替换车辆部分渲染
        pattern_vehicle = r'os << "------------------------- 车辆 --------------------------\\\\n";.*?os << "物位计 Z     : " << fmtNum\(s\.level_z, s\.level_z_valid, " m"\) << "\\\\n\\\\n";'
        replacement_vehicle = '''os << "------------------------- 车辆 --------------------------\\\\n";
    os << "状态         : " << (s.vehicle_detected ? "检测到" : "-") << "\\\\n";
    os << "车头 max_x（大X端）: " << fmtNum(s.vehicle_max_x, s.vehicle_detected, " m") << "\\\\n";
    os << "车尾 min_x（小X端）: " << fmtNum(s.vehicle_min_x, s.vehicle_detected, " m") << "\\\\n";
    os << "前车帮高       : " << fmtNum(s.height, s.vehicle_detected, " m") << "\\\\n";
    os << "后车帮高      : " << fmtNum(s.rear_height, s.vehicle_detected, " m") << "\\\\n";
    os << "车长         : " << fmtNum(s.vehicle_length, s.vehicle_detected, " m") << "\\\\n";
    os << "物位计 Z     : " << fmtNum(s.level_z, s.level_z_valid, " m") << "\\\\n\\\\n";'''

        content = re.sub(pattern_vehicle, replacement_vehicle, content, flags=re.DOTALL)

        # 替换 PLC 写入部分渲染：有符号输出车头和车尾毫米值
        pattern_plc = r'os << "写入 车头位置 : " << s\.reg_min_x_raw << " mm\\\\n";\\s*os << "写入 车尾位置 : " << s\.reg_max_x_raw << " mm\\\\n";'
        replacement_plc = '''os << "写入 车头位置 : " << static_cast<int16_t>(s.reg_min_x_raw) << " mm\\\\n";
    os << "写入 车尾位置 : " << static_cast<int16_t>(s.reg_max_x_raw) << " mm\\\\n";'''
        content = re.sub(pattern_plc, replacement_plc, content)

        with open(p, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {p}")
    except Exception as e:
        print(f"Error {p}: {e}")

# 3. 修复 main.cpp
main_files = [
    '/container/LD_MLScan_common_tm16_test/src/main.cpp',
    '/container/LD_MLScan_common/src/main.cpp'
]

for p in main_files:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            c = f.read()

        if 'const double rear_height_val' not in c:
            c = c.replace(
                'const double height_val = millimeters.front_height_mm.value_or(0) / 1000.0;',
                'const double height_val = millimeters.front_height_mm.value_or(0) / 1000.0;\\n                const double rear_height_val = millimeters.rear_height_mm.value_or(0) / 1000.0;'
            )
        if 'g_snap.rear_height = rear_height_val;' not in c:
            c = c.replace(
                'g_snap.height = height_val;',
                'g_snap.height = height_val;\\n                    g_snap.rear_height = rear_height_val;\\n                    g_snap.position_signed = g_config.plc_position_signed;'
            )
        c = c.replace(
            'g_snap.vehicle_length = g_snap.height = 0.0;',
            'g_snap.vehicle_length = g_snap.height = g_snap.rear_height = 0.0;'
        )

        with open(p, 'w', encoding='utf-8') as f:
            f.write(c)
        print(f"Updated {p}")
    except Exception as e:
        print(f"Error {p}: {e}")
"""

    sftp = ssh.open_sftp()
    with sftp.file('/tmp/patch_fix_all.py', 'w') as f:
        f.write(patch_py)
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command('python3 /tmp/patch_fix_all.py')
    print("PATCH OUTPUT:", stdout.read().decode('utf-8'))

    # 清理并全量重新编译 mlscan
    print("Clean and full rebuild mlscan on Jetson...")
    build_cmd = """
cd /container/LD_MLScan_common_tm16_test/build
rm -f CMakeFiles/mlscan.dir/src/main.cpp.o CMakeFiles/mlscan.dir/src/lidar/*.o bin/mlscan
make -j4
"""
    stdin, stdout, stderr = ssh.exec_command(build_cmd)
    out = stdout.read().decode('utf-8')
    err = stderr.read().decode('utf-8')
    print("BUILD OUTPUT (last 15 lines):")
    print('\n'.join(out.splitlines()[-15:]))
    if "error:" in err.lower():
        print("BUILD ERROR:", err)

    # 检查编译产物中的字符串
    stdin, stdout, stderr = ssh.exec_command("strings /container/LD_MLScan_common_tm16_test/build/bin/mlscan | grep '前车帮高'")
    print("STRINGS CHECK (前车帮高):", stdout.read().decode('utf-8'))

    ssh.close()

if __name__ == '__main__':
    main()
