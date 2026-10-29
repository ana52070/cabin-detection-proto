# -*- coding: utf-8 -*-
import paramiko

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

    # 1. 确保 tools/tm16_pcap_verify/main.cpp 里面加载了师傅限定值
    patch_code = """
import re

main_paths = [
    '/container/LD_MLScan_common_tm16_test/tools/tm16_pcap_verify/main.cpp',
    '/container/LD_MLScan_common/tools/tm16_pcap_verify/main.cpp'
]

for p in main_paths:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            content = f.read()
        target = 'c.max_top_difference=j.value("max_top_difference",c.max_top_difference);'
        addition = (
            '\\n    c.min_board_height=j.value("min_board_height",c.min_board_height);\\n'
            '    c.max_empty_gap=j.value("max_empty_gap",c.max_empty_gap);\\n'
            '    c.max_base_jump=j.value("max_base_jump",c.max_base_jump);\\n'
            '    c.enable_z_uniformity=j.value("enable_z_uniformity",c.enable_z_uniformity);'
        )
        if target in content and 'min_board_height' not in content:
            content = content.replace(target, target + addition)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Patched {p}")
        else:
            print(f"Already patched or target not found in {p}")
    except Exception as e:
        print(f"Error {p}: {e}")
"""

    sftp = ssh.open_sftp()
    with sftp.file('/tmp/patch_verify_main.py', 'w') as f:
        f.write(patch_code)
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command('python3 /tmp/patch_verify_main.py')
    print("PATCH OUTPUT:", stdout.read().decode('utf-8'))

    # 2. 清理并重新编译 tools/tm16_pcap_verify
    build_cmd = """
cd /container/LD_MLScan_common_tm16_test/tools/tm16_pcap_verify/build
rm -f CMakeFiles/tm16_pcap_verify.dir/main.cpp.o tm16_pcap_verify
make -j4
"""
    stdin, stdout, stderr = ssh.exec_command(build_cmd)
    print("BUILD OUTPUT:", stdout.read().decode('utf-8'))
    err = stderr.read().decode('utf-8')
    if err:
        print("BUILD STDERR:", err)

    # 3. 检查字符串
    stdin, stdout, stderr = ssh.exec_command('strings /container/LD_MLScan_common_tm16_test/tools/tm16_pcap_verify/build/tm16_pcap_verify | grep z_breakage_rejected')
    print("STRINGS CHECK:", stdout.read().decode('utf-8'))

    ssh.close()

if __name__ == '__main__':
    main()
