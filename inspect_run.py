import paramiko
import time

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

    print("Running mlscan for 50 seconds to cover vehicle detection range...")
    cmd = """
    cd /container/LD_MLScan_common_tm16_test &&
    timeout 50 ./run_pcap.sh > /tmp/mlscan_test.log 2>&1
    """
    stdin, stdout, stderr = ssh.exec_command(cmd)
    stdout.read()

    # 查看日志中出现的仪表盘
    check_script = """
with open('/tmp/mlscan_test.log', 'r') as f:
    lines = f.readlines()

dashboards = []
cur_dash = []
in_dash = False
for line in lines:
    if '实时仪表盘' in line:
        if cur_dash:
            dashboards.append(''.join(cur_dash))
            cur_dash = []
        in_dash = True
    if in_dash:
        cur_dash.append(line)
        if '累计写入' in line:
            dashboards.append(''.join(cur_dash))
            cur_dash = []
            in_dash = False

print(f'Total dashboards captured: {len(dashboards)}')
if dashboards:
    print('--- Sample Dashboard 1 ---')
    print(dashboards[len(dashboards)//2])
    print('--- Latest Dashboard ---')
    print(dashboards[-1])

# 查看几条 TM16测量 日志
tm16_lines = [l for l in lines if '[TM16测量]' in l and '状态=ok' in l]
print(f'Total TM16测量 ok lines: {len(tm16_lines)}')
if tm16_lines:
    print('--- Sample TM16 OK Lines ---')
    for l in tm16_lines[-5:]:
        print(l.strip())
"""
    sftp = ssh.open_sftp()
    with sftp.file('/tmp/parse_log.py', 'w') as f:
        f.write(check_script)
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command('python3 /tmp/parse_log.py')
    print(stdout.read().decode())
    print(stderr.read().decode())

    ssh.close()

if __name__ == '__main__':
    main()
