# -*- coding: utf-8 -*-
import paramiko
import json

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

sftp = ssh.open_sftp()
for p in ['/container/LD_MLScan_common_tm16_test/mlconfig/config_lingang.json',
          '/container/LD_MLScan_common_tm16_test/mlconfig/config_lingang_pcap.json']:
    try:
        with sftp.file(p, 'r') as f:
            c = json.load(f)
        print(f"File: {p}")
        print("  Keys:", list(c.keys()))
        if "plc" in c:
            print("  plc section:", c["plc"])
        else:
            print("  plc section: NOT PRESENT (using defaults)")
        if "plc_output" in c:
            print("  plc_output section:", c["plc_output"])
    except Exception as e:
        print(f"Error {p}: {e}")
sftp.close()
ssh.close()
