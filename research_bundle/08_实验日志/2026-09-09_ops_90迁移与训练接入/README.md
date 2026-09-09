# 90迁移与训练接入（2026-09-09）

> **后续完成回执（9月9日10时核回）**：数据恢复于02:41:55完成、耗时806.68秒，Drone train/val17990/1469、LLVIP fit/dev9619/2406，完整解码与标签计数检查通过。回执存于[新复现条目 server_state](../2026-09-09_repro_RGBIR对比方法优先接入/server_state/data_receipt_success.json)。下列02:33的RUNNING是历史快照，不再代表当前数据状态；数据完成不等于训练准入。

**90接续已实际开始：两份原数据已找到，dataX隔离环境通过CPU模型加载，原协议train/dev恢复任务已后台启动；还没有新训练结果。教师/参考备份和90实际训练速度仍未解决。**

本次指令授权90上的检查与接入准备，94旧任务不自动恢复。所有原数据、历史工程与失败结果保留；不在已95%使用的`/mnt/dataY`写入新大数据/权重。训练需沿用资源余量纪律、数据/损失协议和实际耗时验证，不能直接套用4090测得速度。最终结果与路径在本条续记。

## 已完成的盘点与接入

| 项目 | 90实查结果 | 本次动作 |
|---|---|---|
| DroneVehicle | `/mnt/dataset/DroneVehicle`，原始train/val双模态可读 | 恢复旧640×512裁边/HBB转换，保持RGB与IR标签独立 |
| LLVIP | `/mnt/dataY/ydf/dataset/LLVIP.zip`，4,003,858,172字节 | 仅恢复official train对应图像/XML；固定fit9619/dev2406 |
| 新写盘 | `/mnt/dataX/ydf`，实际在`/dev/sdb /mnt`，约2.2TB余量 | 已新建`projects/RGBT_campaign_90`；无新大文件写dataY/系统盘 |
| 运行时 | Python3.11.15；torch2.10.0+cu128、Ultralytics8.4.115 | 新venv补齐numpy2.2.6/scipy1.15.3；CPU导入成功 |
| 通用初始化 | 90既有80类YOLO11n，2,624,080参数 | 复制到新`weights/pretrained/yolo11n.pt`；不是教师权重 |
| 当前教师/参考 | 本次自有常规项目、results、experiments、backups及本地索引中未找到94四份现用权重 | 已向用户询问可访问备份；未把旧June模型冒充现用教师 |

94环境Python为3.10.20，90为3.11.15；OpenCV等辅助包也未证明完全一致。复用版本与处理规则不等于整个运行环境、JPEG字节或随机轨迹已证明等价。若重建教师，将记录新身份并用90匹配基线比较。

## 数据恢复任务

`2026-09-09 02:28:28 +08`以`setsid nohup`启动PID1930758。入口`prepare_data90.py`，CPU及OpenCV单线程，CUDA隐藏；只写全新`data_attempt1`。运行日志：

`/mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/migration_20260909_attempt1/data_prepare_launch1.log`

- Drone验证17990/1469图像对、完整RGB源组名单、各模态标签数及旧转换receipt细项计数；所有输出图像另做可解码验证。
- LLVIP冻结完整fit/dev名单，ZIP只读取选中的36075个图像/XML成员；流式EOF校验CRC，不读取test图像/标签，不新计算文件摘要哈希。
- 最终生成双模态YAML、paired/shuffled train映射和弱模态dev映射；保留原`prepare_rgbt_experiment.py`接口。
- `receipt_success.json`出现前不能称完成。`PREPARED`仅表示数据恢复，不等于KD实际loader、梯度或训练准入通过。
- 所有失败attempt保留，入口拒绝复用现存输出目录。

02:33:49实查：Drone train RGB的17990张转换已完成并通过该模态旧receipt计数比较，IR转换到4500张；全流程仍RUNNING，进程RSS约126MiB、1线程。此快照不能当两数据集恢复完成，准确时间见`stage90_0236.json`内部UTC字段（文件名时间标签不准确，以内部时间为准）。

LLVIP映射解析到raw symlink目标是旧逻辑：实际双模态loader由privileged YAML建立teacher dataset，通过canonical image路径查索引，再从teacher dataset读取processed标签。静态链路已核，90真实loader仍需实际检查。

## 资源与耗时判断

02:15和02:24两次快照均为7张可见RTX3090且全部有计算负载，没有完全空卡。6号卡仅约1.1GiB显存，但GPU利用率100%，不能称空卡；其驱动报告PID在当前`/proc`不可见，不能认定是可停止的自有进程。未停止、清理或reset任何GPU任务。

02:24卡3占用约13.1GiB，其他0/1/2/4/5约20–22GiB；容量随共享负载变化。共享卡并非禁止，但必须实测显存余量、完整进程树RSS及吞吐后决定。90仅约503GiB主机内存，不照搬94的1TiB可用量。

所有后续GPU训练/评估/校准共用新项目唯一`runs/.project_resource_leases.json`。现有工程更严格的每卡两个任务、项目显存<70%、整卡≥2GiB余量及240GiB预约阈值保留；不移植94活跃租约或旧测得峰值作为90准入。

94的C1快版0.22136秒/批、E200纯训练6.92小时是4090短窗结果，不能承诺3090也小于10小时。90尚未做GPU性能短测。本轮短测只验速度/资源/数值，不用分钟级微调判定方法去留；完整实验仍保持E200、640、batch32及原人口。

## 接下来按什么顺序推进

**代码接入已完成的部分：**快版端口89文件已部署，90实际检查75个Python源码并导入C1/N算子及训练依赖，`code_cpu_smoke90.txt`为`CPU_IMPORT_PASSED_TEMPLATES_BLOCKED`、`cuda_initialized=false`。三配置因尚未绑定教师/数据/实测资源而技术性阻塞，不需要再次向用户申请“是否允许迁移”。原formal_native最小来源包也已部署，实际`--help`与CPU import通过（`native_help90.txt`、`native_import90.json`），不加载教师/参考。四份重建模板和说明见[NATIVE_REBUILD_PLAN.md](NATIVE_REBUILD_PLAN.md)。来源包只供现有入口接入，不能另建资源队列。

1. 先完成数据恢复和真实loader核对；已有CPU接入继续绑定实际数据、权重及配置。
2. 优先恢复原IR42教师/RGB42参考及父run的`args.yaml`。若备份不可用，原formal_native入口可在没有T/R的情况下训练：先重建所选数据集的RGB/IR seed42，明确新身份，不伪造原checkpoint或假教师。两数据集配置都准备，资源不一次铺开整个历史矩阵。
3. 先在90做实际训练路径的速度/显存短测，包含数据读取、优化器和验证保存开销；不调小batch、不换子集、不删原生损失来伪装加速。预计完整任务超过10小时则先定位本机瓶颈，不直接提交数十小时实验。
4. 恢复必要权重后继续方法工作：LLVIP优先解决有实证机会的定位目标内容；Drone继续针对类别混淆核实传递内容。现有C1快版可以复用工程，但不是已证明有效的新方法；严格L1几何阻塞不会因换服务器自动解除。
5. 首批只安排与一个已冻结候选直接相关的匹配基线/方法，随后按原三seed与四臂标准扩展；不默认重跑旧C1三seed、全部控制及27次矩阵。94故障前部分epoch不视为新端点，也不在90无状态验证地续训。

若四份权重都需重建，基于当前证据优先LLVIP的RGB/IR seed42组合，再恢复Drone组合；这是基础模型恢复优先级，不是定位方法已准入长训的结论。当前没有新lambda、门控或方法胜负结论。

## 证据与局限

- 本地原始盘点：`inventory90.json`、`weights90_search.json`、`weights90_extra_search.json`、`data90_archive.json`、`root_created90.txt`。
- 环境：`environment90_metadata_attempt2.json`、`environment_ready90.txt`；原metadata尝试遇缺SciPy后已修复新venv，旧环境保留。
- 数据：`prepare_data90.py`、`payload_data/`、`data_prepare_launch1.json`及稍后实际完成回执。
- 代码：`payload_code/CHANGES.md`及端口源码；仅来源归档不可作为新准入凭证。
- 结构回执：[90临时项目接入](../../99_整理回执/2026-09-09_90临时项目接入.md)。首次Ultralytics import自动迁移用户全局settings的副作用也已记录，后续缓存/配置均指向新项目。

服务器项目根`PROJECTS_INDEX_20260909.md`互指本条。未访问94、未启动或恢复GPU训练、未发布新AP、未修改历史结果。数据准备仍在后台运行时应按实际日志报告阶段，不能提前写成成功。
