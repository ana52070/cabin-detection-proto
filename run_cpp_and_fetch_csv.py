# -*- coding: utf-8 -*-
import paramiko
import os
import time

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

    # 1. 运行 tm16_pcap_verify 跑整段 空车.pcap
    cmd = (
        "cd /container/LD_MLScan_common_tm16_test && "
        "rm -rf tm16_verify_output && "
        "./tools/tm16_pcap_verify/build/tm16_pcap_verify "
        "--pcap pcap/空车.pcap "
        "--config mlconfig/config_lingang_pcap.json "
        "--out-dir tm16_verify_output"
    )
    print(f"Executing: {cmd}")
    t0 = time.time()
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    print(f"Finished in {time.time() - t0:.2f}s")
    print("OUTPUT (first 20 lines):")
    print('\n'.join(out.splitlines()[:20]))
    if err:
        print("STDERR (first 10 lines):")
        print('\n'.join(err.splitlines()[:10]))

    # 2. 从 Jetson 下载 tm16_verify_output/results.csv 到本地 cpp_results.csv
    remote_csv = "/container/LD_MLScan_common_tm16_test/tm16_verify_output/results.csv"
    local_csv = "cpp_results.csv"
    print(f"Downloading {remote_csv} -> {local_csv}")
    sftp = ssh.open_sftp()
    sftp.get(remote_csv, local_csv)
    sftp.close()
    print("Download completed successfully!")

    ssh.close()

if __name__ == '__main__':
    main()
