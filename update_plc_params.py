import paramiko

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.0.176', 22, 'nvidia', 'nvidia', timeout=10)

    patch_script = """
import json

# 1. 修复 plc_registers.hpp，使 signed_positions 真正控制头尾编码模式
paths = [
    '/container/LD_MLScan_common/src/lidar/tm16/plc_registers.hpp',
    '/container/LD_MLScan_common_tm16_test/src/lidar/tm16/plc_registers.hpp',
    '/container/LD_MLScan_common/tools/tm16_pcap_verify/plc_registers.hpp',
    '/container/LD_MLScan_common_tm16_test/tools/tm16_pcap_verify/plc_registers.hpp'
]
for p in paths:
    with open(p, 'r') as f:
        c = f.read()
    c = c.replace('out.head_position=encode(mm.head_position_mm,true,4);', 'out.head_position=encode(mm.head_position_mm,signed_positions,4);')
    c = c.replace('out.tail_position=encode(mm.tail_position_mm,true,8);', 'out.tail_position=encode(mm.tail_position_mm,signed_positions,8);')
    with open(p, 'w') as f:
        f.write(c)
    print("UPDATED plc_registers.hpp:", p)

# 2. 修改全部 config_lingang.json 和 config_lingang_pcap.json
# offset_m = 35.0, position_signed = false
config_paths = [
    '/container/LD_MLScan_common/mlconfig/config_lingang.json',
    '/container/LD_MLScan_common/mlconfig/config_lingang_pcap.json',
    '/container/LD_MLScan_common_tm16_test/mlconfig/config_lingang.json',
    '/container/LD_MLScan_common_tm16_test/mlconfig/config_lingang_pcap.json',
    '/container/mlscan_lingang_deploy/mlconfig/config_lingang.json',
    '/container/mlscan_lingang_deploy/mlconfig/config_lingang_pcap.json'
]

for cp in config_paths:
    with open(cp, 'r') as f:
        data = json.load(f)
    if 'plc' in data:
        data['plc']['offset_m'] = 35.0
        data['plc']['position_signed'] = False
    with open(cp, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("UPDATED CONFIG:", cp)
"""

    sftp = ssh.open_sftp()
    with sftp.file('/tmp/apply_plc_unsigned.py', 'w') as f:
        f.write(patch_script)
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command('python3 /tmp/apply_plc_unsigned.py')
    print("STDOUT:\n", stdout.read().decode())
    print("STDERR:\n", stderr.read().decode())

    ssh.close()

if __name__ == '__main__':
    main()
