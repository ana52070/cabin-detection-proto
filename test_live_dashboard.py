# -*- coding: utf-8 -*-
import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

# 在 pty=True 下执行，这样 isatty() 成立，dashboard 才能渲染！
stdin, stdout, stderr = ssh.exec_command("cd /container/LD_MLScan_common_tm16_test && timeout 5 ./run_pcap.sh", get_pty=True)
time.sleep(4)
out = stdout.read().decode('utf-8', errors='replace')
print("--- OUTPUT CAPTURED ---")
for line in out.splitlines():
    if "仪表盘" in line or "车辆" in line or "车头" in line or "车尾" in line or "车长" in line or "车帮" in line or "写入" in line:
        print(line)

ssh.close()
