# -*- coding: utf-8 -*-
import paramiko
import os
import sys
import json

def main():
    print("Connecting to Jetson 192.168.0.232...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.0.232', 22, 'nvidia', 'nvidia', timeout=10)
    sftp = ssh.open_sftp()

    lingang_dir = "/container/LD_MLScan_lingang"

    # 1. Upload header files
    uploads = [
        ("cabin_kalman.hpp", f"{lingang_dir}/src/lidar/tm16/cabin_kalman.hpp"),
        ("cabin_kalman.hpp", f"{lingang_dir}/tools/tm16_pcap_verify/cabin_kalman.hpp"),
        ("new_height_quality_filter.hpp", f"{lingang_dir}/src/lidar/tm16/height_quality_filter.hpp"),
        ("new_height_quality_filter.hpp", f"{lingang_dir}/tools/tm16_pcap_verify/height_quality_filter.hpp"),
        ("new_measurement_pipeline.hpp", f"{lingang_dir}/src/lidar/tm16/measurement_pipeline.hpp"),
        ("new_measurement_config.hpp", f"{lingang_dir}/src/lidar/tm16/measurement_config.hpp")
    ]

    for local_f, remote_f in uploads:
        print(f"Uploading {local_f} -> {remote_f}")
        sftp.put(local_f, remote_f)

    # 2. Update config_lingang_pcap.json
    cfg_remote = f"{lingang_dir}/mlconfig/config_lingang_pcap.json"
    with sftp.open(cfg_remote, "r") as f:
        cfg = json.load(f)

    if "height_filter" not in cfg:
        cfg["height_filter"] = {}

    cfg["height_filter"].update({
        "enable_kalman": True,
        "max_coast_frames": 5,
        "max_coast_time_ms": 600,
        "pos_process_noise_q": 0.05,
        "pos_measure_noise_r": 0.15,
        "height_process_noise_q": 0.001,
        "height_measure_noise_r": 0.03,
        "height_init_cov": 0.10,
        "height_gate_limit": 0.25
    })

    print(f"Updating {cfg_remote} with Kalman settings...")
    with sftp.open(cfg_remote, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # Also update tools config_lingang.json
    tools_cfg_remote = f"{lingang_dir}/tools/tm16_pcap_verify/config_lingang.json"
    try:
        with sftp.open(tools_cfg_remote, "r") as f:
            tcfg = json.load(f)
        if "height_filter" not in tcfg:
            tcfg["height_filter"] = {}
        tcfg["height_filter"].update(cfg["height_filter"])
        with sftp.open(tools_cfg_remote, "w") as f:
            json.dump(tcfg, f, indent=2, ensure_ascii=False)
        print(f"Updated {tools_cfg_remote}")
    except Exception as e:
        print(f"Note: {e}")

    # 3. Patch tools/tm16_pcap_verify/main.cpp
    main_cpp_path = f"{lingang_dir}/tools/tm16_pcap_verify/main.cpp"
    with sftp.open(main_cpp_path, "r") as f:
        content = f.read().decode('utf-8')

    # Add cabin_kalman.hpp include if not present
    if '#include "cabin_kalman.hpp"' not in content:
        content = '#include "cabin_kalman.hpp"\n' + content

    # Add FILTER_VALUE macro items
    old_macro = 'FILTER_VALUE(position_jump_base_m);FILTER_VALUE(max_position_speed_mps);'
    new_macro = (
        'FILTER_VALUE(position_jump_base_m);FILTER_VALUE(max_position_speed_mps);\n'
        '    FILTER_VALUE(enable_kalman);FILTER_VALUE(max_coast_frames);FILTER_VALUE(max_coast_time_ms);\n'
        '    FILTER_VALUE(pos_process_noise_q);FILTER_VALUE(pos_measure_noise_r);\n'
        '    FILTER_VALUE(height_process_noise_q);FILTER_VALUE(height_measure_noise_r);\n'
        '    FILTER_VALUE(height_init_cov);FILTER_VALUE(height_gate_limit);'
    )
    if old_macro in content and 'FILTER_VALUE(enable_kalman)' not in content:
        content = content.replace(old_macro, new_macro)

    # Add PositionKalmanFilter instantiation
    old_kf_pos = 'verify::HeightQualityFilter height_filter(filter_config);'
    new_kf_pos = (
        'verify::HeightQualityFilter height_filter(filter_config);\n'
        '    verify::KalmanConfig pos_kf_cfg;\n'
        '    pos_kf_cfg.enabled = filter_config.enable_kalman;\n'
        '    pos_kf_cfg.max_coast_frames = filter_config.max_coast_frames;\n'
        '    pos_kf_cfg.max_coast_time_ms = filter_config.max_coast_time_ms;\n'
        '    pos_kf_cfg.pos_process_noise_q = filter_config.pos_process_noise_q;\n'
        '    pos_kf_cfg.pos_measure_noise_r = filter_config.pos_measure_noise_r;\n'
        '    verify::PositionKalmanFilter pos_kf(pos_kf_cfg);'
    )
    if old_kf_pos in content and 'PositionKalmanFilter pos_kf' not in content:
        content = content.replace(old_kf_pos, new_kf_pos)

    # In loop: update position Kalman
    old_algo = 'const auto r = runAlgorithms(combined, strategy, algorithm_config,short_config,mode);'
    new_algo = (
        'auto r = runAlgorithms(combined, strategy, algorithm_config,short_config,mode);\n'
        '                if (filter_config.enable_kalman) {\n'
        '                    if (r.position) {\n'
        '                        pos_kf.update(std::min(r.position->min_x, r.position->max_x),\n'
        '                                      std::max(r.position->min_x, r.position->max_x),\n'
        '                                      newest - origin);\n'
        '                    } else {\n'
        '                        auto coasted = pos_kf.coast(newest - origin);\n'
        '                        if (coasted) {\n'
        '                            r.position = lidar::XRange{coasted->x_rear, coasted->x_front};\n'
        '                            r.length = coasted->length;\n'
        '                            if (r.short_edges) {\n'
        '                                r.short_edges->position = r.position;\n'
        '                                r.short_edges->length = r.length;\n'
        '                                r.short_edges->status = "kalman_coasting";\n'
        '                            }\n'
        '                        }\n'
        '                    }\n'
        '                }'
    )
    if old_algo in content and 'pos_kf.update' not in content:
        content = content.replace(old_algo, new_algo)

    with sftp.open(main_cpp_path, "w") as f:
        f.write(content.encode('utf-8'))
    print("Patched main.cpp in tm16_pcap_verify successfully.")

    sftp.close()
    ssh.close()
    print("Deployment completed.")

if __name__ == '__main__':
    main()
