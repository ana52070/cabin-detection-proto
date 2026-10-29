import paramiko

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

    print("Running tm16_pcap_verify with config_lingang_pcap.json on Jetson...")
    cmd_run_verify = """
    cd /container/LD_MLScan_common_tm16_test/tools/tm16_pcap_verify/build &&
    ./tm16_pcap_verify --pcap /container/LD_MLScan_common_tm16_test/pcap/空车.pcap --config /container/LD_MLScan_common_tm16_test/mlconfig/config_lingang_pcap.json --out-dir .
    """
    stdin, stdout, stderr = ssh.exec_command(cmd_run_verify)
    print("VERIFY STDOUT:\n", stdout.read().decode()[:1000])
    err = stderr.read().decode()
    if err:
        print("VERIFY STDERR:\n", err[:1000])

    # 检查结果
    remote_check = """
import csv

with open('/container/LD_MLScan_common_tm16_test/tools/tm16_pcap_verify/build/results.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print('Total rows:', len(rows))
for r in rows:
    if r['frame'] == '718':
        print('Frame 718:', r)
        break

valid_low = sum(1 for r in rows if float(r.get('z_low_m', 0) or 0) > 0)
valid_high = sum(1 for r in rows if float(r.get('z_high_m', 0) or 0) > 0)
valid_final = sum(1 for r in rows if float(r.get('final_top_z_m', 0) or 0) > 0)
print(f'Valid Low Top Frames: {valid_low}/{len(rows)}')
print(f'Valid High Top Frames: {valid_high}/{len(rows)}')
print(f'Valid Final Top Frames: {valid_final}/{len(rows)}')
"""
    sftp = ssh.open_sftp()
    with sftp.file('/tmp/check_verify.py', 'w') as f:
        f.write(remote_check)
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command('python3 /tmp/check_verify.py')
    print("CHECK STDOUT:\n", stdout.read().decode())
    print("CHECK STDERR:\n", stderr.read().decode())

    ssh.close()

if __name__ == '__main__':
    main()
