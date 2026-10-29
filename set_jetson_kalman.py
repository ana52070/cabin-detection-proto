# -*- coding: utf-8 -*-
import sys
import json
import paramiko

def set_jetson_kalman(enable: bool):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.0.232', 22, 'nvidia', 'nvidia', timeout=10)
    
    sftp = ssh.open_sftp()
    remote_path = '/container/LD_MLScan_lingang/mlconfig/config_lingang_pcap.json'
    with sftp.open(remote_path, 'r') as f:
        data = json.load(f)
    
    if 'height_filter' in data:
        data['height_filter']['enable_kalman'] = enable
    
    with sftp.open(remote_path, 'w') as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False))
    
    sftp.close()
    ssh.close()
    print(f"Successfully set Jetson config enable_kalman = {enable}")

if __name__ == '__main__':
    enable_arg = sys.argv[1].lower() in ('true', '1', 'on') if len(sys.argv) > 1 else False
    set_jetson_kalman(enable_arg)
