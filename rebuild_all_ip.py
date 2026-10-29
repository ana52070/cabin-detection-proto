import paramiko

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.0.176', 22, 'nvidia', 'nvidia', timeout=10)

    # 1. 编译 LD_MLScan_common_tm16_test/mlscan
    print("=== Compiling LD_MLScan_common_tm16_test/mlscan ===")
    cmd1 = """
    cd /container/LD_MLScan_common_tm16_test/build &&
    rm -f CMakeFiles/mlscan.dir/src/main.cpp.o ../bin/mlscan &&
    make mlscan -j4
    """
    stdin, stdout, stderr = ssh.exec_command(cmd1)
    print("OUT1:", stdout.read().decode()[-300:])

    # 2. 编译 LD_MLScan_common/mlscan
    print("=== Compiling LD_MLScan_common/mlscan ===")
    cmd2 = """
    cd /container/LD_MLScan_common/build &&
    rm -f CMakeFiles/mlscan.dir/src/main.cpp.o ../bin/mlscan &&
    make mlscan -j4
    """
    stdin, stdout, stderr = ssh.exec_command(cmd2)
    print("OUT2:", stdout.read().decode()[-300:])

    # 3. 同步最新编译好的 bin/mlscan 到 /container/mlscan_lingang_deploy/ 并重新打包
    print("=== Updating deploy package ===")
    cmd3 = """
    cp /container/LD_MLScan_common/build/bin/mlscan /container/mlscan_lingang_deploy/ &&
    cd /container &&
    tar -czf mlscan_lingang_deploy.tar.gz mlscan_lingang_deploy &&
    echo 'DEPLOY TAR REPACKED' &&
    ls -lh /container/mlscan_lingang_deploy.tar.gz
    """
    stdin, stdout, stderr = ssh.exec_command(cmd3)
    print("OUT3:\n", stdout.read().decode())

    ssh.close()

if __name__ == '__main__':
    main()
