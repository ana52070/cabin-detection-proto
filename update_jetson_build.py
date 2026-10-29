# -*- coding: utf-8 -*-
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

# 1. Update run_pcap.sh
sftp = ssh.open_sftp()
script_content = """#!/bin/bash
cd /container/LD_MLScan_common_tm16_test
./build/bin/mlscan --config mlconfig/config_lingang_pcap.json "$@"
"""
with sftp.open('/container/LD_MLScan_common_tm16_test/run_pcap.sh', 'w') as f:
    f.write(script_content)
sftp.chmod('/container/LD_MLScan_common_tm16_test/run_pcap.sh', 0o775)
print('Updated run_pcap.sh with exact path!')

# 2. Force recompile lidar_processing.cpp and mlscan
cmd = "cd /container/LD_MLScan_common_tm16_test/build && rm -f CMakeFiles/mlscan.dir/src/lidar/lidar_processing.cpp.o bin/mlscan && make -j4"
stdin, stdout, stderr = ssh.exec_command(cmd)
print('Make stdout:', stdout.read().decode('utf-8', errors='replace'))
print('Make stderr:', stderr.read().decode('utf-8', errors='replace'))

# 3. Check strings in binary
stdin, stdout, stderr = ssh.exec_command("strings /container/LD_MLScan_common_tm16_test/build/bin/mlscan | grep -E 'short_edges_missing|z_breakage_rejected'")
print('Binary strings check:', stdout.read().decode('utf-8', errors='replace'))

sftp.close()
ssh.close()
