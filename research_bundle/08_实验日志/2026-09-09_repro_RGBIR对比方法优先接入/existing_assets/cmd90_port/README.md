# CMDistill-corrected 最小 90 端口

**已准备三份未改源码、两份原测试和两份 90 模板；身份为 `PROTOCOL-ADAPTED`，真实 IR 教师 `TO_BIND`，尚未准入训练。** 本包只准备 CMD 一条路径，不建立新调度器或方法矩阵，不阻塞其他方法各自的复现工作。

来源为当前 `03_现行工程/SpaceNet6_OTD_official_reproduction`。复制记录见 [COPY_RECEIPT.json](COPY_RECEIPT.json)，静态检查见 [PREPARATION_CHECKS.json](PREPARATION_CHECKS.json)：五文件逐字节一致、AST 可解析、两模板原配方/增强未改、22 个原测试函数仍在。本轮未执行模型、测试函数或 runtime import；未 SSH/GPU/新 hash/安装环境。

## 最小来源和依赖

仅复制以下三份训练源码，不修改旧工程：

- [tools/train_rgbt_cmdistill.py](tools/train_rgbt_cmdistill.py)：原 `cmdistill_corrected` 路径、native loss 与实际 B 倍 KD、冻结 IR 教师、普通 RGB/visible 学生推理。
- [yolo_osssl/rgbt_cmdistill_kd.py](yolo_osssl/rgbt_cmdistill_kd.py)：PCCFD、SLRD、IBCLD 三项原算式与系数1。
- [yolo_osssl/rgbt_hnewa_pairing.py](yolo_osssl/rgbt_hnewa_pairing.py)：原同步增强配对 adapter。

原 [loss tests](tests/test_rgbt_cmdistill_kd.py) 和 [trainer tests](tests/test_train_rgbt_cmdistill.py) 共22个测试函数一起保留，不另造验收框架。

本包是**增量源包**，不是独立完整 Python package。其共享依赖已经在 [payload_native](../../../2026-09-09_ops_90迁移与训练接入/payload_native/README.md)：`paired_detection.py`，原 `yolo_osssl/__init__/manifest/model`，`tools/project_resource_guard.py`、`write_jstars_run_receipt.py`、`validate_jstars_data_contract.py` 及 exposure ledger。根代理应把 native payload 放入一个**新独立运行源目录**，再按本包相对路径放入上述三源码/测试/配置；不修改正在使用的旧源或共享 guard。不建议把两个分离目录简单加到 PYTHONPATH：已有普通 `yolo_osssl` package 的 `__init__` 会让另一目录的增量模块不可见。

## 两配置与真实初始化

| 数据集 | 模板 | 学生完整数据 | IR 教师 |
|---|---|---|---|
| DroneVehicle | [drone](configs/research/cmdistill_corrected_90_drone_seed42.template.yaml) | `ROOT/data_attempt1/prepared/dronevehicle/rgb.data.yaml` | `TO_BIND`，必须对应5类真实 IR detector |
| LLVIP | [llvip](configs/research/cmdistill_corrected_90_llvip_seed42.template.yaml) | `ROOT/data_attempt1/prepared/llvip/visible.data.yaml` | `TO_BIND`，必须对应1类真实 IR detector |

`ROOT=/mnt/dataX/ydf/projects/RGBT_campaign_90`。两者学生从现有 `ROOT/weights/pretrained/yolo11n.pt` 通用预训练初始化；它是学生初始化，**不是 IR 教师**。映射与 prepared receipt 均指向各数据集 `data_attempt1/prepared/` 原文件。保留 E200、640、B32、nbs64、workers8、SGD、lr0/lrf=.01、warmup3、普通 BN 和原弱增强。IR checkpoint 经根代理实际绑定后走原必填 CLI `--teacher-weights`；CMD 无 RGB 参考模型，不需要假的 T/R 占位模型。

模板 `port90` 只是明确的准备 metadata，原 trainer 不读取它的 readiness/teacher 字段。正式入口必须由主 root 统一租约先核真实教师/数据/版本；不能因模板文件存在就宣称已绑定。新 90 初始化、模型头和数据流须形成新证据，不能把旧94三 seed均值作为新90分母。

## 根代理可直接执行的 CPU 接线入口

在合并后的新运行源目录内，使用现有 90 环境，不做全局安装：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /mnt/dataX/ydf/projects/RGBT_campaign_90/environments/rgbir90/bin/python tools/train_rgbt_cmdistill.py --help
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /mnt/dataX/ydf/projects/RGBT_campaign_90/environments/rgbir90/bin/python -m pytest -q -p no:cacheprovider tests/test_rgbt_cmdistill_kd.py tests/test_train_rgbt_cmdistill.py
```

第二条仅在现有环境已具备 pytest 时执行；未安装时向根代理报告，不在本包引入依赖安装。`--help` 不加载教师或数据；tests 使用小合成张量，非两数据集实际训练复现。

日后由**主 root 唯一资源 lease** 调用原训练入口，参数形状为 `--config <本模板的已绑定副本> --arm cmdistill_corrected --teacher-weights <真实IR权重> --seed 42 --batch 32 --workers 8 --output <新目录> --device <租约卡>`。本包没有创建队列、资源池或启动任何任务，也不复用历史 shell 的无限重试和固定显存数字。

训练前技术短测只核实际接口/冻结教师/非零且有限 KD/显存与端到端速度，供单任务≤10h判断；不以短测 AP 做方法输赢裁决。原 `--max-steps` 计 optimizer 调用，不保证 AMP 成功更新；`--validate-only` 只做部分文件检查；原 runtime 没有本包新增硬限时。使用根代理既有测量/限时能力，不为补这些字段另写框架。完整评价仍需 accepted full-dev adapter；旧内部关闭 eval、last 独立评价的历史薄 receipt不能自动当新90完整正式比较。

原 corrected 适配、历史六端点与局限见 [资产审计](../README.md)。本包没有 author-exact、三 seed增益或当前90性能结论。

## 90 实际接线更新（2026-09-09 10:15）

根代理已在独立合并目录 source_attempt1 执行 help 与原测试：22 passed in 1.68s，两个进程 returncode=0。见 [CPU回执](cmd_cpu_receipt_attempt1.json)、[测试日志](cmd_tests_attempt1.log)、[合并回执](cmd_source_merged.json)。真实IR教师仍为 TO_BIND；前文未运行描述为打包时状态，不能解释为当前未做CPU测试。没有真实数据训练或GPU准入。
