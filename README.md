# TM16 激光雷达点云离线解析与算法验证平台 (Python 版)

本项目是针对 **TM16 激光雷达识别算法重构** 构建的 Python 基础验证工具，实现了离线 PCAP 抓包高效读取、DIFOP 标定动态解析、激光点云三维坐标解算、外参变换、算法第一阶段（ROI粗过滤与X轴粗聚类）以及交互式 3D GUI 可视化播放器。

### 2026-10-22：GPU 优先解析与 CPU 自动回退

继续双击 `start_gui.bat` 启动即可。默认优先使用 NVIDIA GPU 计算点云坐标和外参变换；PCAP 解包、包顺序和切帧由 CPU 处理。解析从逐个 32 点小块运算改为整帧批量运算，CPU 回退路径也使用批量处理。界面加载进度和加载完成信息显示实际后端及解析耗时。

GPU 依赖缺失、驱动/运行库初始化失败、计算失败或显存不足时，程序自动改用 CPU；计算中失败会用保留的原始输入重算当前帧，再由 CPU 处理剩余帧，避免漏帧或重复帧。该次加载中不反复尝试失败的 GPU；重新打开文件时会重新探测。取消加载仍按取消处理。CPU 本身失败、文件读取失败等异常仍正常报错。

`lidar_learn` 环境已安装 CuPy 14.2.0 及 CUDA 12.9 运行库。本机其他 Python 环境如未安装 GPU 依赖会自动使用 CPU。其他机器可在实际启动 GUI 的环境安装：

```powershell
conda run --name lidar_learn python -m pip install "cupy-cuda12x[ctk]>=14,<15"
```

安装方式来自 [CuPy 官方安装文档](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi)。内核编译缓存默认放在本程序的 `.gpu_cache` 目录，可通过 `CUPY_CACHE_DIR` 覆盖。首次 GPU 初始化和内核编译可能增加加载时间。

需要手动强制 CPU 对照时，在 PowerShell 中执行；关闭此终端即可结束该临时设置：

```powershell
$env:TM16_DECODE_BACKEND = 'cpu'
.\start_gui.bat
```

恢复默认 GPU 优先：

```powershell
Remove-Item Env:TM16_DECODE_BACKEND -ErrorAction SilentlyContinue
.\start_gui.bat
```

程序接口支持 `TM16Decoder(backend='auto')`、`backend='gpu'`、`backend='cpu'`；前两种都会在 GPU 失败时回退 CPU，显式参数优先于环境变量。可通过 `active_backend`、`fallback_reason` 和 `backend_status` 查看实际状态。

复测命令如下。第二条会对整份默认录包比较原版、批量 CPU、GPU 两次运行，检查全部点的坐标误差（上限 0.05 mm）及帧元数据/点顺序；CPU 要求输出逐值一致。耗时包含后端初始化，但不包含结果比较及 GUI 渲染。

```powershell
conda run --name lidar_learn python -m unittest test_gpu_decoder -v
conda run --name lidar_learn python verify_gpu_decode.py --out verification_gpu/metrics_full_truck.json
```

本次原始文件保存在 `verification_gpu/baseline_decoder.py` 和 `verification_gpu/baseline_visualizer_gui.py`。验证只针对 Python 离线程序，不代表现场雷达或 PLC 验收结果。

本机完整录包实测：2465 帧、34879892 个点，原版 48.09 秒，批量 CPU 12.00 秒，GPU 两次分别 12.91/14.05 秒（含初始化）。GPU 相对原版约快 3.4～3.7 倍；本次批量 CPU 更快。CPU 输出与原版逐值一致，GPU 全部点的最大坐标差异约 0.00474 mm。36 项选定测试通过，其中 11 项为解析专项；历史跟踪/恢复测试中有 21 项原有失败，已用原解析器复测确认失败名单相同。详见 [GPU 解析验证说明](verification_gpu/验证说明.md)。

### 2026-10-21：前后位置与车长的缺点修复（最新）

位置输出新增包数、端点位移、可靠跨度和ROI截断检查；中间缺点时尝试局部恢复两端，无法恢复时最多预测5帧且不超过0.6秒，超时失效。三图与主界面使用同一份序列结果，纯位置预测不标成PLC有效实测。位置过程噪声改为10.0以减少起停过冲。

第1727～2294帧内，前/后位置、车长超过1m的相邻跳变从6/4/9次降至0；卡尔曼模式551帧常规实测校正、13帧局部恢复、4帧预测。关闭时四个预测位置帧保留缺测。31项专项及卡尔曼测试通过；全程高度和旧测试仍有限制。完整证据及操作见 [位置缺点修复验证说明](verification_position_dropout/验证说明.md)。

### 2026-10-21：完整空车录包第1727～2294帧测高优化（上一轮）

默认加载“全流程空车.pcap”，测高半宽改为0.65m，默认开启卡尔曼。修复两种模式的端点映射、局部恢复测高来源及后台参数传递；时序窗口新增“前后车帮位置与车长（三图）”页签，与高度四图共用选帧。位置及长度仍为现有粗聚类边界/跨度。

重点区间卡尔曼小端相邻帧变化P95由45.00mm降至6.65mm，综合输出564/568帧，其中7帧包含单端预测。无实测真值，不能据此给出绝对准确率；全程覆盖率和旧测试仍有限制。完整结果、手动步骤及备份位置见 [验证说明](verification_focus_1727_2294/验证说明.md)。下文2026-10-19章节为旧录包和旧实现的历史结果。

---

## 一、目录结构

`
临港_调试/tm16_python_proto/
│
├── config.py              # 配置定义：ROI 范围、外参矩阵、默认 PCAP 路径、雷达协议常量
├── pcap_reader.py         # 高性能纯 Python 二进制 PCAP 读取器（支持经典/纳秒 PCAP、VLAN、IPv4/UDP 解包）
├── tm16_decoder.py        # TM16 雷达协议解析器（DIFOP 16线垂直仰角提取、12块方位角插值解算、切角组帧）
├── short_edge_detector.py # 算法模块：ROI 粗过滤（剔除地面与场外杂点）+ X 轴 200 网格大粗聚类圈车
├── visualizer_gui.py      # PySide6 + PyQtGraph 3D 点云可视化交互式播放器主界面（支持参数实时调节）
├── main.py                # 启动入口脚本
├── start_gui.bat          # Windows 一键启动脚本（标准 ASCII，自动激活 conda lidar_learn）
└── README.md              # 说明文档
`

---

## 二、运行方式

### 1. Windows 桌面/资源管理器双击启动（推荐）

直接双击运行目录下的：
`
start_gui.bat
`
*(软件启动时默认自动异步加载并渲染 空车.pcap)*

### 2026-10-19：809帧断档恢复优化（当前版本）

目标确认和高度预热现已并行：确认期间，只暂存每个候选链的实测高度，不发布该帧高度或位置。连续3帧确认后，把同一候选链之前的样本加入当前窗口；只有凑齐5个有效、限时且高度一致的实测样本才输出。已经处理过的确认前帧保持缺测，不回填、不读取未来帧。候选切换或缺测会丢弃未确认的预热样本。

超过 `reset_gap=1.0 s` 无可靠结果时进入重新确认，不立即丢弃目标参考；最多保留 `identity_memory=3.0 s`。恢复时同时检查：

- 最近最多6个可靠位置估计的车厢中心速度，以及两端相同平移的运动一致性。预测位置只用来搜索和关联本帧实际候选，不作为实测输出。
- 长度差不超过0.6 m，端点速度不超过配置上限5 m/s，预测位置余量为 `0.5 m + 0.5 × recovery_acceleration × 时间差²`，当前 `recovery_acceleration=1.0 m/s²`。
- 原有单帧立面/底板验证，并比较两端墙面P98顶部高度和P98-P5高度跨度；差异门限 `profile_tolerance=0.5 m`。

关联通过时保留目标代次并标记 `resumed`；不通过或参考过期时走新目标重新捕获，标记 `reacquired` 并增加目标代次。两种情况都需要连续3帧确认；长断档前的高度样本不会作为恢复后的预热样本。几何相似车辆仍可能存在身份歧义，以上是关联判定，不是物理身份真值保证。

主窗口会显示“候选确认中（高度同步预热）”“同一目标恢复”“目标重新捕获”或“持续跟踪”。在同一份PCAP下，重点查看809～815帧；CSV编号从0开始，界面顶部计数从1开始。

测试与本轮回放对比：

```powershell
Set-Location 'D:\imformation\a企业实习\项目任务\26年10月\临港_调试\tm16_python_proto'
conda run --name lidar_learn python -m unittest test_cabin_tracking test_cabin_recovery -v
$baseline = (Get-ChildItem backups\恢复优化前_*.zip | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
conda run --name lidar_learn python verify_cabin_recovery.py --baseline $baseline
```

本轮输出单独保存于 `verification_recovery`，不会覆盖上一轮证据。`before.csv` 对应767帧输出的上一版，`after.csv` 为当前版，`metrics.json` 包含809帧附近逐帧事件和恢复等待时间；`frames_data.json` 保存每次确认使用的预热样本及原始高度。`comparison.png` 为全程对比，`recovery_809.png` 为局部对比。候选预热的导出字段 `warmup_samples` 为JSON数组，确认前的正式高度字段仍为空。

同一份1572帧空车PCAP的本轮实测：

| 指标 | 上一版 | 当前版 |
|---|---:|---:|
| 809帧之后首个最终高度输出（从0编号） | 815帧 | 813帧 |
| 从809帧收到数据到首次最终输出 | 0.600 s | 0.420 s |
| 全程最终有效输出 | 767帧 | 773帧 |
| 两端连续有效帧间超过15 cm的跳变合计 | 0次 | 0次 |
| 融合高度最大连续帧变化 | 3.56 cm | 3.56 cm |

本次811帧判定为同一目标恢复，目标代次保持4，预热种子为809、810帧；813帧的窗口使用809～813这5个实际候选高度。809、810帧的正式结果仍为空，断档前的高度不进入这个窗口。恢复等待减少约0.180 s（30%），原始PCAP时间轴中的约2.11 s数据空档仍然保留，未用预测填补。

41项回归测试通过，离屏界面已检查恢复/确认状态、后台计算和调参缓存失效；尚未进行现场验收。统计表不能证明错误关联率或高度准确率为零，仍需有车辆身份和高度真值的其他录包进一步验证。

### 2026-10-19：第一轮车帮高度时序优化（历史结果）

该轮改动只作用于本目录的 Python 原型。单帧几何识别保持无状态，完整序列计算增加历史端点跟踪：

- 先以端点位移和车厢长度筛选候选，再应用原有立面及底板检查。位移上限为 `position_slack + max_speed × 时间差`，默认分别为 0.5 m 和 5 m/s，长度差门限为 0.6 m。
- 初次捕获、丢失后重新捕获，以及持续高票新候选替换旧目标，都需要连续 3 帧确认。旧目标仍可见时允许新候选持续挑战，避免初始误锁后无法纠正。缺测时不返回旧坐标。
- 高度滤波只使用当前帧和历史帧，必须有完整的 5 个有效样本。首轮锁定需要 3 帧，随后高度需要 5 个样本；约 10 Hz 的连续理想输入下，第一帧至首次输出约 0.6 s，稳定段中值对线性变化约滞后 0.2 s。
- 默认启用短缺测历史保留：历史最多 0.7 s，相邻帧时间断档超过 0.3 s 或目标代次改变时清空。缺测、被拒绝的当前帧始终不输出，也不插值。
- 窗口高度跨度不得超过 0.2 m。短时高度异常不进入可靠窗口，连续 5 个稳定的新高度样本可重新建立窗口。短缺测保留模式下，恢复帧必须有本帧实际测量并通过检查。
- 两端仍以高度差不超过 0.3 m 才融合；该条件不能证明两端识别一定正确，仍需人工点云及现场真值核对。

参数位于 `config.py` 的 `CABIN_TRACKING_CONFIG` 与 `HEIGHT_TIMESERIES_CONFIG`。将 `preserve_short_gaps` 改为 `False` 可切换到严格连续 5 帧模式，任何无效帧都清空对应端窗口。门限是本次空车回放的原型初值，尚不能视为其他车型和速度的现场验收值。

手动验证步骤：

1. 关闭旧进程，重新双击 `start_gui.bat`。
2. 打开“车帮顶部时序图”，点击“重新提取与计算”。首次打开也会自动计算。
3. 重点检查约 57.7、76.6、81.6、97 及 149～151 s。点击曲线可跳到对应点云；计算完成后，三维车厢框使用同一序列的端点跟踪结果。计算前显示单帧几何结果，状态提示中会注明。
4. 改变 ROI、车帮参数或切换 PCAP 后，旧曲线与跟踪结果会清除；再次点击“重新提取与计算”。后台任务使用参数快照。
5. 关闭窗口停止查看。程序没有新增任何真实 PLC 写入或远端部署操作；主窗口 PLC 数值仍按本帧原始 P98 高度计算，不等同于时序图的滤波融合高度。

回归测试（不连接设备）：

```powershell
Set-Location 'D:\imformation\a企业实习\项目任务\26年10月\临港_调试\tm16_python_proto'
conda run --name lidar_learn python -m unittest test_cabin_tracking -v
```

完整 PCAP 验证：

```powershell
$baseline = (Get-ChildItem backups\时序优化前_*.zip | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
conda run --name lidar_learn python verify_cabin_tracking.py --baseline $baseline
```

`--baseline` 必须指向本轮优化前保存的源码 ZIP。结果输出到 `verification_tracking`：`before.csv`、`after.csv` 为逐帧识别、滤波及有效性原因，`metrics.json` 为统计，`comparison.png` 为四图对比。运行指标中的跳变只统计连续有效帧，不能替代高度准确率；恢复延迟不包含未恢复成功的片段。源码原件保存在 `backups`，原始 PCAP 与原有 CSV 保持原状。

本次空车 PCAP 实测共 1572 帧，SHA-256 为 `2e82e18bbb86d5b4aeb90797eadc413b75f24b838724d55b9af55ea9e86f6e16`：

| 指标 | 修改前 | 修改后（短缺测保留模式） |
|---|---:|---:|
| 小X端连续有效帧间超过15 cm的跳变 | 2次 | 0次 |
| 大X端连续有效帧间超过15 cm的跳变 | 17次 | 0次 |
| 融合高度最大连续帧变化 | 7.09 cm | 3.56 cm |
| 最终有效输出 | 956帧 | 767帧 |
| 占全部回放帧比例 | 60.81% | 48.79% |

有效输出减少约19.8%，是历史跟踪确认、完整窗口及异常拒绝带来的实际代价。不能据此宣称识别准确率提高多少；当前没有人工标注真值，进出视野阶段的低高度及稳定偏差仍需进一步核验。26项回归测试通过，界面在离屏模式下验证了后台计算、跟踪框联动、调参后缓存失效、绘图和退出；尚未进行现场实时雷达验收。`optimized_timeseries.png` 为本次逐帧结果绘制的优化后四图。

### 2. 命令行直接启动（conda lidar_learn 环境）

`powershell
& conda run --live-stream --name lidar_learn python D:\imformation\a企业实习\项目任务\26年10月\临港_调试\tm16_python_proto\main.py
`

---

## 三、参数实时调节功能（边调边看）

在右侧控制面板中新增了 **参数实时调节面板**，任何参数微调均会**零延迟直接生效**：

### 1. ⚙️ ROI 空间范围与地面滤除
- **X Min (前向) / X Max (后向)**：截取沿车道长方向的有效范围（默认 [-15.0, 15.0] m）。调节时三维金黄 ROI 框即时伸缩，框外杂点实时剥离。
- **Y Min (左侧) / Y Max (右侧)**：截取车道左右宽度（默认 [-3.0, 3.0] m）。用于剔除车道两侧的立柱、护栏与杂物。
- **Z Min (下限) / Z Max (上限)**：截取空间垂直高度（默认 [0.2, 8.0] m）。用于过滤高空遮雨棚与悬挂结构。
- **Ground Z 门限**：地表水泥与道床剥离阈值（默认  .20 m）。调大可更彻底剔除水泥地面噪点，调太大会误削车轮和底梁。

### 2. 🚗 X 轴大网格粗聚类参数
- **X 轴划分网格数 (y_grid_num)**：默认 200 格。动态显示单格分辨率（默认  .150 m/格）。
- **单格有效点数 (cluster_min_points)**：默认 3 点。单网格内点数门限，低于此值视为空白格。
- **容忍最大空隙 (cluster_max_gap)**：默认 3 格。允许连续空格的数量，跨越车厢内部低反射暗区。
- **最小整车跨度 (cluster_min_span)**：默认 3 格。成簇的最小网格数，剔除行人等小体量误检。
- **🔄 恢复默认参数配置**：一键恢复为 config.py 的出厂默认值。
