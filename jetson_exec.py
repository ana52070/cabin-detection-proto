# -*- coding: utf-8 -*-
import sys
import paramiko

def run_remote(cmd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    ssh.close()
    return out, err

if __name__ == '__main__':
    if len(sys.argv) > 1:
        cmd = ' '.join(sys.argv[1:])
        out, err = run_remote(cmd)
        print(out)
        if err:
            print('STDERR:', err, file=sys.stderr)
