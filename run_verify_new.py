# -*- coding: utf-8 -*-
import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

# 测试运行 tm16_pcap_verify 跑整段 PCAP 并查看结果
cmd = (
    "cd /container/LD_MLScan_common_tm16_test && "
    "rm -rf tm16_verify_output && "
    "./tools/tm16_pcap_verify/build/tm16_pcap_verify "
    "--pcap pcap/空车.pcap "
    "--config mlconfig/config_lingang_pcap.json "
    "--out-dir tm16_verify_output"
)
print("Executing verification run...")
t0 = time.time()
stdin, stdout, stderr = ssh.exec_command(cmd)
out = stdout.read().decode('utf-8')
print(f"Done in {time.time()-t0:.2f}s")

# 下载最新的 results.csv
sftp = ssh.open_sftp()
sftp.get('/container/LD_MLScan_common_tm16_test/tm16_verify_output/results.csv', 'cpp_results_new.csv')
sftp.close()
ssh.close()
print("Downloaded cpp_results_new.csv")
