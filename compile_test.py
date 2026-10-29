import paramiko

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.4.222', 22, 'nvidia', 'nvidia', timeout=10)

    # 1. 编译 mlscan
    print("=== Compiling mlscan ===")
    cmd_compile_mlscan = """
    cd /container/LD_MLScan_common_tm16_test/build &&
    rm -f CMakeFiles/mlscan.dir/src/lidar/tm16_driver.cpp.o CMakeFiles/mlscan.dir/src/main.cpp.o ../bin/mlscan &&
    make mlscan -j4
    """
    stdin, stdout, stderr = ssh.exec_command(cmd_compile_mlscan)
    out = stdout.read().decode()
    err = stderr.read().decode()
    print("OUT:", out[-500:] if len(out) > 500 else out)
    if err:
        print("ERR:", err[-500:] if len(err) > 500 else err)

    # 2. 编译 tm16_pcap_verify
    print("\n=== Compiling tm16_pcap_verify ===")
    cmd_compile_verify = """
    cd /container/LD_MLScan_common_tm16_test/tools/tm16_pcap_verify/build &&
    rm -f CMakeFiles/tm16_pcap_verify.dir/main.cpp.o tm16_pcap_verify &&
    make tm16_pcap_verify -j4
    """
    stdin, stdout, stderr = ssh.exec_command(cmd_compile_verify)
    out = stdout.read().decode()
    err = stderr.read().decode()
    print("OUT:", out[-500:] if len(out) > 500 else out)
    if err:
        print("ERR:", err[-500:] if len(err) > 500 else err)

    ssh.close()

if __name__ == '__main__':
    main()
