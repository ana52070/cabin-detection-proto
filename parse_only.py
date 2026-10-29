import paramiko

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

    stdin, stdout, stderr = ssh.exec_command("sed -n '1145,1175p' /container/LD_MLScan_common_tm16_test/src/main.cpp")
    print(stdout.read().decode())

    ssh.close()

if __name__ == '__main__':
    main()
