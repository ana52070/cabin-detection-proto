import paramiko

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.0.176', 22, 'nvidia', 'nvidia', timeout=10)

    remote_pack_script = """#!/bin/bash
set -e

rm -rf /container/mlscan_lingang_deploy /container/mlscan_lingang_deploy.tar.gz
mkdir -p /container/mlscan_lingang_deploy/lib
mkdir -p /container/mlscan_lingang_deploy/mlconfig
mkdir -p /container/mlscan_lingang_deploy/logs
mkdir -p /container/mlscan_lingang_deploy/pcap

cp /container/LD_MLScan_common/build/bin/mlscan /container/mlscan_lingang_deploy/
cp -a /container/LD_MLScan_common/release_tm16_pcap_20261015/lib/* /container/mlscan_lingang_deploy/lib/
cp -a /container/LD_MLScan_common/mlconfig/* /container/mlscan_lingang_deploy/mlconfig/
[ -f /container/LD_MLScan_common/car.png ] && cp /container/LD_MLScan_common/car.png /container/mlscan_lingang_deploy/ || true
[ -f /container/LD_MLScan_common/ftpConfig.json ] && cp /container/LD_MLScan_common/ftpConfig.json /container/mlscan_lingang_deploy/ || true
[ -f /container/LD_MLScan_common/pcap/空车.pcap ] && cp /container/LD_MLScan_common/pcap/空车.pcap /container/mlscan_lingang_deploy/pcap/ || true

# 1. 现场直连雷达脚本
cat > /container/mlscan_lingang_deploy/run_online.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")"
export LD_LIBRARY_PATH="$PWD/lib:${LD_LIBRARY_PATH:-}"
exec ./mlscan --config mlconfig/config_lingang.json "$@"
EOF

# 2. 现场 PCAP 回放脚本 (同样连接现场 PLC)
cat > /container/mlscan_lingang_deploy/run_pcap.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")"
export LD_LIBRARY_PATH="$PWD/lib:${LD_LIBRARY_PATH:-}"
exec ./mlscan --config mlconfig/config_lingang_pcap.json "$@"
EOF

chmod +x /container/mlscan_lingang_deploy/run_online.sh
chmod +x /container/mlscan_lingang_deploy/run_pcap.sh
chmod +x /container/mlscan_lingang_deploy/mlscan

cd /container
tar -czf mlscan_lingang_deploy.tar.gz mlscan_lingang_deploy
echo "PACKAGE_SUCCESS"
ls -lh /container/mlscan_lingang_deploy.tar.gz
"""

    sftp = ssh.open_sftp()
    with sftp.file('/tmp/do_package.sh', 'w') as f:
        f.write(remote_pack_script)
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command('bash /tmp/do_package.sh')
    print("STDOUT:\n", stdout.read().decode())
    print("STDERR:\n", stderr.read().decode())

    ssh.close()

if __name__ == '__main__':
    main()
