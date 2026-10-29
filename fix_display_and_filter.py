import paramiko

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.0.176', 22, 'nvidia', 'nvidia', timeout=10)

    patch_script = """
dash_headers = [
    '/container/LD_MLScan_common_tm16_test/src/common/terminal_dashboard.h',
    '/container/LD_MLScan_common/src/common/terminal_dashboard.h'
]
for dh in dash_headers:
    with open(dh, 'r') as f:
        c = f.read()
    c = c.replace('mm\\n\";', 'mm\\\\n\";')
    with open(dh, 'w') as f:
        f.write(c)
    print("SUCCESS CLEANED:", dh)
"""

    sftp = ssh.open_sftp()
    with sftp.file('/tmp/clean_dash.py', 'w') as f:
        f.write(patch_script)
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command('python3 /tmp/clean_dash.py')
    print("STDOUT:\n", stdout.read().decode())
    print("STDERR:\n", stderr.read().decode())

    stdin, stdout, stderr = ssh.exec_command("sed -n '140,146p' /container/LD_MLScan_common_tm16_test/src/common/terminal_dashboard.h")
    print("CHECK DASH:\n", stdout.read().decode())

    ssh.close()

if __name__ == '__main__':
    main()
