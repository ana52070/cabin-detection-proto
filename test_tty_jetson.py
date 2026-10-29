import paramiko

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.0.176', 22, 'nvidia', 'nvidia', timeout=10)

    test_script = """
import pty, os, time, subprocess, select

master, slave = pty.openpty()
p = subprocess.Popen(['./run_pcap.sh'], cwd='/container/mlscan_lingang_deploy', stdin=slave, stdout=slave, stderr=slave, close_fds=True)
os.close(slave)

out = b''
start = time.time()
while time.time() - start < 25:
    r, _, _ = select.select([master], [], [], 0.5)
    if r:
        try:
            data = os.read(master, 4096)
            if not data:
                break
            out += data
        except OSError:
            break
p.terminate()
os.close(master)

text = out.decode('utf-8', errors='ignore')
parts = text.split('=================== mlscan 实时仪表盘 ===================')
found = 0
for part in parts:
    if '写入 车头位置' in part and '写入 车头位置 : 0 mm' not in part:
        print('=== CAPTURED DASHBOARD ===')
        idx = part.find('-------------------- 车牌相机')
        print(part[:idx] if idx != -1 else part)
        found += 1
        if found >= 2:
            break
"""

    sftp = ssh.open_sftp()
    with sftp.file('/tmp/test_tty.py', 'w') as f:
        f.write(test_script)
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command('python3 /tmp/test_tty.py')
    print("STDOUT:\n", stdout.read().decode())
    print("STDERR:\n", stderr.read().decode())

    ssh.close()

if __name__ == '__main__':
    main()
