# -*- coding: utf-8 -*-
import paramiko

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

    sftp = ssh.open_sftp()

    # 1. 上传新的 height_quality_filter.hpp
    with open('new_height_quality_filter.hpp', 'rb') as f:
        data = f.read()

    for p in ['/container/LD_MLScan_common_tm16_test/src/lidar/tm16/height_quality_filter.hpp',
              '/container/LD_MLScan_common/src/lidar/tm16/height_quality_filter.hpp',
              '/container/LD_MLScan_common_tm16_test/tools/tm16_pcap_verify/height_quality_filter.hpp',
              '/container/LD_MLScan_common/tools/tm16_pcap_verify/height_quality_filter.hpp']:
        with sftp.file(p, 'wb') as f:
            f.write(data)
        print(f"Uploaded height_quality_filter.hpp -> {p}")

    # 2. 改造 short_edge_detector.hpp 中的 measureShortEdgeTop 为稳健 98% 分位数提取
    patch_code = """
import re

target_files = [
    '/container/LD_MLScan_common_tm16_test/src/lidar/tm16/short_edge_detector.hpp',
    '/container/LD_MLScan_common/src/lidar/tm16/short_edge_detector.hpp',
    '/container/LD_MLScan_common_tm16_test/tools/tm16_pcap_verify/short_edge_detector.hpp',
    '/container/LD_MLScan_common/tools/tm16_pcap_verify/short_edge_detector.hpp'
]

new_measure_func = '''inline void measureShortEdgeTop(ShortEdgeCandidate& wall, const ShortEdgeConfig& s, double z_max) {
    std::vector<double> zs;
    zs.reserve(wall.points.points.size());
    for (const auto& p : wall.points.points) {
        if (std::isfinite(p.z) && p.z <= z_max && p.z >= 0.5) {
            zs.push_back(p.z);
        }
    }
    if (zs.size() < 3) return; // 至少 3 个点即可稳定测出立面顶部高度

    const double pct = (s.top_percentile > 0.0 && s.top_percentile <= 1.0) ? s.top_percentile : 0.98;
    const double top = edgePercentile(zs, pct);
    wall.top = top;
    wall.top_vote_count = static_cast<size_t>(zs.size());
    wall.top_samples.push_back(top);

    for (const auto& p : wall.points.points) {
        if (std::abs(p.z - top) <= 0.15) {
            wall.top_points.points.push_back(p);
        }
    }
}'''

for p in target_files:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            content = f.read()

        # 匹配 inline void measureShortEdgeTop(...) 直到下一个注释
        pattern = r'inline void measureShortEdgeTop\(ShortEdgeCandidate& wall, const ShortEdgeConfig& s, double z_max\) \{.*?\n\}'
        if re.search(pattern, content, flags=re.DOTALL):
            content = re.sub(pattern, new_measure_func, content, count=1, flags=re.DOTALL)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Patched measureShortEdgeTop in {p}")
        else:
            print(f"Pattern not found in {p}")
    except Exception as e:
        print(f"Error {p}: {e}")
"""
    with sftp.file('/tmp/patch_measure.py', 'w') as f:
        f.write(patch_code)
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command('python3 /tmp/patch_measure.py')
    print("PATCH OUTPUT:", stdout.read().decode('utf-8'))

    # 3. 清理并重新编译 mlscan 和 tm16_pcap_verify
    print("Rebuilding on Jetson...")
    build_cmd = """
cd /container/LD_MLScan_common_tm16_test/build
rm -f CMakeFiles/mlscan.dir/src/lidar/*.o CMakeFiles/mlscan.dir/src/main.cpp.o bin/mlscan
make mlscan -j4

cd /container/LD_MLScan_common_tm16_test/tools/tm16_pcap_verify/build
rm -f CMakeFiles/tm16_pcap_verify.dir/main.cpp.o tm16_pcap_verify
make tm16_pcap_verify -j4
"""
    stdin, stdout, stderr = ssh.exec_command(build_cmd)
    out = stdout.read().decode('utf-8')
    err = stderr.read().decode('utf-8')
    print("BUILD OUTPUT (last 15 lines):")
    print('\n'.join(out.splitlines()[-15:]))
    if "error:" in err.lower():
        print("BUILD ERROR:", err)

    ssh.close()

if __name__ == '__main__':
    main()
