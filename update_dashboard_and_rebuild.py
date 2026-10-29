# -*- coding: utf-8 -*-
import paramiko

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

    patch_py = """
# 1. Patch terminal_dashboard.h
dash_paths = [
    '/container/LD_MLScan_common_tm16_test/src/common/terminal_dashboard.h',
    '/container/LD_MLScan_common/src/common/terminal_dashboard.h'
]

for p in dash_paths:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            c = f.read()
        
        # 增加 rear_height 和 position_signed 到 Snapshot
        if 'double rear_height = 0.0;' not in c:
            c = c.replace(
                'double height = 0.0;          // 车帮高(m)',
                'double height = 0.0;          // 前车帮高(m)\\n    double rear_height = 0.0;     // 后车帮高(m)'
            )
            c = c.replace(
                'bool plc_connected = false;',
                'bool plc_connected = false;\\n    bool position_signed = false;'
            )

        # 修改 【车辆】 显示顺序为用户指定格式
        old_vehicle_block = '''    os << "------------------------- 车辆 --------------------------\\n";
    os << "状态         : " << (s.vehicle_detected ? "检测到" : "-") << "\\n";
    os << "车尾 min_x（小X端）: " << fmtNum(s.vehicle_min_x, s.vehicle_detected, " m") << "\\n";
    os << "车头 max_x（大X端）: " << fmtNum(s.vehicle_max_x, s.vehicle_detected, " m") << "\\n";
    os << "车长         : " << fmtNum(s.vehicle_length, s.vehicle_detected, " m") << "\\n";
    os << "车帮高       : " << fmtNum(s.height, s.vehicle_detected, " m") << "\\n";
    os << "物位计 Z     : " << fmtNum(s.level_z, s.level_z_valid, " m") << "\\n\\n";'''

        new_vehicle_block = '''    os << "------------------------- 车辆 --------------------------\\n";
    os << "状态         : " << (s.vehicle_detected ? "检测到" : "-") << "\\n";
    os << "车头 max_x（大X端）: " << fmtNum(s.vehicle_max_x, s.vehicle_detected, " m") << "\\n";
    os << "车尾 min_x（小X端）: " << fmtNum(s.vehicle_min_x, s.vehicle_detected, " m") << "\\n";
    os << "前车帮高       : " << fmtNum(s.height, s.vehicle_detected, " m") << "\\n";
    os << "后车帮高      : " << fmtNum(s.rear_height, s.vehicle_detected, " m") << "\\n";
    os << "车长         : " << fmtNum(s.vehicle_length, s.vehicle_detected, " m") << "\\n";
    os << "物位计 Z     : " << fmtNum(s.level_z, s.level_z_valid, " m") << "\\n\\n";'''

        if old_vehicle_block in c:
            c = c.replace(old_vehicle_block, new_vehicle_block)

        # PLC 写入显示支持有符号负数显示
        old_plc_block = '''    os << "写入 车头位置 : " << s.reg_min_x_raw << " mm\\n";
    os << "写入 车尾位置 : " << s.reg_max_x_raw << " mm\\n";'''

        new_plc_block = '''    if (s.position_signed) {
        os << "写入 车头位置 : " << static_cast<int16_t>(s.reg_min_x_raw) << " mm\\n";
        os << "写入 车尾位置 : " << static_cast<int16_t>(s.reg_max_x_raw) << " mm\\n";
    } else {
        os << "写入 车头位置 : " << s.reg_min_x_raw << " mm\\n";
        os << "写入 车尾位置 : " << s.reg_max_x_raw << " mm\\n";
    }'''

        if old_plc_block in c:
            c = c.replace(old_plc_block, new_plc_block)

        with open(p, 'w', encoding='utf-8') as f:
            f.write(c)
        print(f"Patched {p}")
    except Exception as e:
        print(f"Error {p}: {e}")

# 2. Patch main.cpp 赋值与清零
main_paths = [
    '/container/LD_MLScan_common_tm16_test/src/main.cpp',
    '/container/LD_MLScan_common/src/main.cpp'
]

for p in main_paths:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            c = f.read()

        # 赋值
        target_val = 'const double height_val = millimeters.front_height_mm.value_or(0) / 1000.0;'
        add_rear = '\\n                const double rear_height_val = millimeters.rear_height_mm.value_or(0) / 1000.0;'
        if target_val in c and 'rear_height_val' not in c:
            c = c.replace(target_val, target_val + add_rear)

        target_snap = 'g_snap.height = height_val;'
        add_snap_rear = '\\n                    g_snap.rear_height = rear_height_val;\\n                    g_snap.position_signed = g_config.plc_position_signed;'
        if target_snap in c and 'g_snap.rear_height' not in c:
            c = c.replace(target_snap, target_snap + add_snap_rear)

        # 清零
        c = c.replace(
            'g_snap.vehicle_length = g_snap.height = 0.0;',
            'g_snap.vehicle_length = g_snap.height = g_snap.rear_height = 0.0;'
        )

        with open(p, 'w', encoding='utf-8') as f:
            f.write(c)
        print(f"Patched {p}")
    except Exception as e:
        print(f"Error {p}: {e}")

# 3. 开启 config_lingang_pcap.json 和 config_lingang.json 的 position_signed
import json
json_paths = [
    '/container/LD_MLScan_common_tm16_test/mlconfig/config_lingang_pcap.json',
    '/container/LD_MLScan_common_tm16_test/mlconfig/config_lingang.json',
    '/container/LD_MLScan_common/mlconfig/config_lingang_pcap.json',
    '/container/LD_MLScan_common/mlconfig/config_lingang.json'
]

for jp in json_paths:
    try:
        with open(jp, 'r', encoding='utf-8') as f:
            jc = json.load(f)
        if 'plc' in jc:
            jc['plc']['position_signed'] = True
        if 'plc_output' in jc:
            jc['plc_output']['position_signed'] = True
        with open(jp, 'w', encoding='utf-8') as f:
            json.dump(jc, f, indent=2, ensure_ascii=False)
        print(f"Enabled position_signed in {jp}")
    except Exception as e:
        print(f"Error {jp}: {e}")
"""

    sftp = ssh.open_sftp()
    with sftp.file('/tmp/patch_dashboard.py', 'w') as f:
        f.write(patch_py)
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command('python3 /tmp/patch_dashboard.py')
    print("PATCH OUTPUT:", stdout.read().decode('utf-8'))

    # 4. 重新编译 mlscan
    print("Recompiling mlscan on Jetson...")
    build_cmd = """
cd /container/LD_MLScan_common_tm16_test/build
rm -f CMakeFiles/mlscan.dir/src/main.cpp.o
make -j4
"""
    stdin, stdout, stderr = ssh.exec_command(build_cmd)
    out = stdout.read().decode('utf-8')
    err = stderr.read().decode('utf-8')
    print("BUILD OUTPUT (last 10 lines):")
    print('\n'.join(out.splitlines()[-10:]))
    if "error:" in err.lower():
        print("BUILD ERROR:", err)

    ssh.close()

if __name__ == '__main__':
    main()
