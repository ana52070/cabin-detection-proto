# -*- coding: utf-8 -*-
import paramiko

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

    patch_py = """
paths = [
    '/container/LD_MLScan_common_tm16_test/src/lidar/tm16/short_edge_detector.hpp',
    '/container/LD_MLScan_common_tm16_test/tools/tm16_pcap_verify/short_edge_detector.hpp',
    '/container/LD_MLScan_common/src/lidar/tm16/short_edge_detector.hpp',
    '/container/LD_MLScan_common/tools/tm16_pcap_verify/short_edge_detector.hpp'
]

old_code = '''    r.status = (!r.length) ? "length_out_of_range" :
               (!r.low_top || !r.high_top) ? "top_unavailable" :
               (!r.height) ? "height_inconsistent" : "ok";'''

new_code = '''    if (!r.height_status.empty()) {
        // height_status already set to agree or different
    } else {
        r.height_status = "top_unavailable";
    }
    r.status = (r.length) ? "ok" : "length_out_of_range";'''

for p in paths:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            content = f.read()
        if old_code in content:
            content = content.replace(old_code, new_code)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Patched {p}")
        else:
            print(f"Pattern not found in {p}")
    except Exception as e:
        print(f"Error {p}: {e}")
"""

    sftp = ssh.open_sftp()
    with sftp.file('/tmp/patch_status.py', 'w') as f:
        f.write(patch_py)
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command('python3 /tmp/patch_status.py')
    print("PATCH OUTPUT:", stdout.read().decode('utf-8'))

    # 重新编译 mlscan 和 tm16_pcap_verify
    build_cmd = """
cd /container/LD_MLScan_common_tm16_test/build
rm -f CMakeFiles/mlscan.dir/src/lidar/*.o
make -j4

cd /container/LD_MLScan_common_tm16_test/tools/tm16_pcap_verify/build
rm -f CMakeFiles/tm16_pcap_verify.dir/main.cpp.o tm16_pcap_verify
make -j4
"""
    print("Rebuilding on Jetson...")
    stdin, stdout, stderr = ssh.exec_command(build_cmd)
    out = stdout.read().decode('utf-8')
    err = stderr.read().decode('utf-8')
    print("BUILD OUTPUT (last 10 lines):")
    print('\n'.join(out.splitlines()[-10:]))
    if "error:" in err.lower():
        print("BUILD ERROR:", err)

    # 重新跑 tm16_pcap_verify 并下载 csv
    verify_cmd = """
cd /container/LD_MLScan_common_tm16_test
rm -rf tm16_verify_output
./tools/tm16_pcap_verify/build/tm16_pcap_verify --pcap pcap/空车.pcap --config mlconfig/config_lingang_pcap.json --out-dir tm16_verify_output
"""
    print("Running tm16_pcap_verify...")
    stdin, stdout, stderr = ssh.exec_command(verify_cmd)
    stdout.read()

    # 下载新的 cpp_results.csv
    sftp = ssh.open_sftp()
    sftp.get('/container/LD_MLScan_common_tm16_test/tm16_verify_output/results.csv', 'cpp_results.csv')
    sftp.close()
    print("Updated cpp_results.csv downloaded!")

    ssh.close()

if __name__ == '__main__':
    main()
