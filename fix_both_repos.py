# -*- coding: utf-8 -*-
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

commands = [
    # 1. Copy updated headers from tm16_test to common
    "cp -v /container/LD_MLScan_common_tm16_test/src/lidar/tm16/short_edge_detector.hpp /container/LD_MLScan_common/src/lidar/tm16/short_edge_detector.hpp",
    "cp -v /container/LD_MLScan_common_tm16_test/src/lidar/tm16/measurement_config.hpp /container/LD_MLScan_common/src/lidar/tm16/measurement_config.hpp",
    "cp -v /container/LD_MLScan_common_tm16_test/mlconfig/config_lingang_pcap.json /container/LD_MLScan_common/mlconfig/config_lingang_pcap.json",
    
    # 2. Fix run_pcap.sh in LD_MLScan_common_tm16_test to NOT cd to LD_MLScan_common
    "echo -e '#!/bin/bash\\ncd /container/LD_MLScan_common_tm16_test\\n./build/bin/mlscan --config mlconfig/config_lingang_pcap.json \"$@\"' > /container/LD_MLScan_common_tm16_test/run_pcap.sh && chmod +x /container/LD_MLScan_common_tm16_test/run_pcap.sh",
    
    # 3. Clean and reconfigure LD_MLScan_common_tm16_test/build so its source dir is tm16_test
    "cd /container/LD_MLScan_common_tm16_test/build && rm -rf CMakeCache.txt CMakeFiles bin/mlscan && cmake .. && make -j4",
    
    # 4. Also recompile LD_MLScan_common
    "cd /container/LD_MLScan_common/build && rm -rf CMakeFiles/mlscan.dir/src/lidar/lidar_processing.cpp.o bin/mlscan && make -j4",
    
    # 5. Check strings on both binaries
    "strings /container/LD_MLScan_common_tm16_test/build/bin/mlscan | grep -E 'short_edges_missing|z_breakage_rejected'",
    "strings /container/LD_MLScan_common/build/bin/mlscan | grep -E 'short_edges_missing|z_breakage_rejected'"
]

for cmd in commands:
    print('===> Executing:', cmd[:80], '...')
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out:
        print('Output:', out[-400:] if len(out) > 400 else out)
    if err:
        print('Stderr:', err[-400:] if len(err) > 400 else err)

ssh.close()
