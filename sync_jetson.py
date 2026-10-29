# -*- coding: utf-8 -*-
import paramiko
import sys
import json

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

    sftp = ssh.open_sftp()
    remote_script = """import json

paths = [
    '/container/LD_MLScan_common_tm16_test/mlconfig/config_lingang_pcap.json',
    '/container/LD_MLScan_common/mlconfig/config_lingang_pcap.json'
]

for p in paths:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            c = json.load(f)
        c['lidar_extrinsics']['lidar_0']['transformation_matrix'][0] = [0.0, 1.0, 0.0, 0.0]
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(c, f, indent=2, ensure_ascii=False)
        print(f"Updated matrix in {p}")
    except Exception as e:
        print(f"Error {p}: {e}")
"""
    with sftp.file('/tmp/fix_matrix.py', 'w') as f:
        f.write(remote_script)
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command('python3 /tmp/fix_matrix.py')
    print(stdout.read().decode('utf-8'))
    print(stderr.read().decode('utf-8'))

    ssh.close()

if __name__ == '__main__':
    main()
