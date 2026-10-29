# -*- coding: utf-8 -*-
import paramiko

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

    sftp = ssh.open_sftp()

    # 1. 上传 clean_terminal_dashboard.h 到两套工程
    with open('clean_terminal_dashboard.h', 'rb') as f:
        data = f.read()

    for p in ['/container/LD_MLScan_common_tm16_test/src/common/terminal_dashboard.h',
              '/container/LD_MLScan_common/src/common/terminal_dashboard.h']:
        with sftp.file(p, 'wb') as f:
            f.write(data)
        print(f"Uploaded terminal_dashboard.h -> {p}")

    # 2. 更新 plc_registers.hpp 支持负数编码
    plc_reg_content = """#pragma once
#include "plc_measurements.hpp"
#include <array>
namespace lidar::tm16_measurement {
struct PlcRegisters {
    uint16_t front_height=0, rear_height=0, head_position=0, tail_position=0;
    unsigned valid_mask=0;
};
// 临港现场位置寄存器统一支持 16 位有符号整数 INT (-32768 ~ +32767 mm)
inline PlcRegisters makePlcRegisters(const PlcMeasurements& mm, bool signed_positions=true) {
    PlcRegisters out;
    auto encode=[&](std::optional<int32_t> v, bool signed_value, unsigned bit) -> uint16_t {
        if (!v || *v<(signed_value ? -32768 : 0) || *v>(signed_value ? 32767 : 65535)) return 0;
        out.valid_mask|=bit;
        return static_cast<uint16_t>(static_cast<int16_t>(*v));
    };
    out.front_height=encode(mm.front_height_mm,false,1);
    out.rear_height=encode(mm.rear_height_mm,false,2);
    out.head_position=encode(mm.head_position_mm,true,4);
    out.tail_position=encode(mm.tail_position_mm,true,8);
    return out;
}
// 四个地址依次为前高、后高、大 X、小 X；单项写失败也尝试清理其余字段。
template<class Writer>
bool writePlcRegisters(const PlcRegisters& f, const std::array<int,4>& addresses, Writer writer) {
    const std::array<uint16_t,4> values={f.front_height,f.rear_height,f.head_position,f.tail_position};
    bool ok=true;
    for (size_t i=0;i<values.size();++i) if (writer(addresses[i],values[i])==-1) ok=false;
    return ok;
}
}
"""
    for p in ['/container/LD_MLScan_common_tm16_test/src/lidar/tm16/plc_registers.hpp',
              '/container/LD_MLScan_common/src/lidar/tm16/plc_registers.hpp',
              '/container/LD_MLScan_common_tm16_test/tools/tm16_pcap_verify/plc_registers.hpp',
              '/container/LD_MLScan_common/tools/tm16_pcap_verify/plc_registers.hpp']:
        try:
            with sftp.file(p, 'w') as f:
                f.write(plc_reg_content)
            print(f"Uploaded plc_registers.hpp -> {p}")
        except Exception as e:
            print(f"Error {p}: {e}")

    sftp.close()

    # 3. 彻底编译 mlscan
    print("Rebuilding mlscan on Jetson...")
    build_cmd = """
cd /container/LD_MLScan_common_tm16_test/build
rm -f CMakeFiles/mlscan.dir/src/main.cpp.o bin/mlscan
make mlscan -j4
"""
    stdin, stdout, stderr = ssh.exec_command(build_cmd)
    out = stdout.read().decode('utf-8')
    err = stderr.read().decode('utf-8')
    print("BUILD OUTPUT (last 10 lines):")
    print('\n'.join(out.splitlines()[-10:]))
    if "error:" in err.lower():
        print("BUILD ERROR:", err)

    # 4. 字符串核验
    stdin, stdout, stderr = ssh.exec_command("strings /container/LD_MLScan_common_tm16_test/build/bin/mlscan | grep '前车帮高'")
    print("STRINGS VERIFICATION (前车帮高):", stdout.read().decode('utf-8').strip())

    ssh.close()

if __name__ == '__main__':
    main()
