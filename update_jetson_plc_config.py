# -*- coding: utf-8 -*-
import paramiko
import json

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

    sftp = ssh.open_sftp()
    paths = [
        '/container/LD_MLScan_common_tm16_test/mlconfig/config_lingang_pcap.json',
        '/container/LD_MLScan_common/mlconfig/config_lingang_pcap.json'
    ]

    for p in paths:
        try:
            with sftp.file(p, 'r') as f:
                c = json.load(f)
            c['plc'] = {
                "ip": "192.168.11.100",
                "port": 502,
                "slave_id": 1,
                "timeout_ms": 500,
                "reconnect_ms": 2000,
                "offset_m": 1.2,
                "position_signed": False,
                "reg_direction": 0,
                "reg_height_1": 1,
                "reg_heartbeat": 2,
                "reg_lidar_alarm": 41,
                "reg_camera_alarm": 41,
                "reg_height_2": 7,
                "reg_plate_start": 25,
                "reg_target": 34,
                "reg_min_x": 43,
                "reg_max_x": 44,
                "reg_data_start": 45,
                "reg_data_count": 29,
                "reg_flag": 52
            }
            c['plc_output'] = {
                "lidar_x_mm": 1200
            }
            with sftp.file(p, 'w') as f:
                json.dump(c, f, indent=2, ensure_ascii=False)
            print(f"Added PLC config to {p}")
        except Exception as e:
            print(f"Error {p}: {e}")

    sftp.close()
    ssh.close()

if __name__ == '__main__':
    main()
