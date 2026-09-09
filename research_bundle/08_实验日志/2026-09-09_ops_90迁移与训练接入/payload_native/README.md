# 原 formal_native 的最小来源包

**只含未修改来源与四个新模板；无 T/R checkpoint，无训练动作。** 复制范围和逐字节检查见 [COPY_RECEIPT.json](COPY_RECEIPT.json)。12 个 Python 文件均已在本地 AST 解析，无运行时导入；Windows 缺 Linux `fcntl`，不 mock 资源 guard 冒充 Linux import 通过。

保留文件：`tools/train_rgbt_kd_detector.py`、`project_resource_guard.py`、`write_jstars_run_receipt.py`、`validate_jstars_data_contract.py`；`yolo_osssl/{__init__,manifest,model,fp_selective_distillation,b0_distillation,raw_head_abi,stage2,paired_detection}.py`；`configs/research/jstars_dataset_exposure_v1.json`。`manifest/model` 是原包初始化的导入依赖，`stage2/paired_detection` 是原 trainer 的静态导入依赖；native 分支不执行教师或配对数据路径。只复制原 package，未改写空 `__init__`。

根代理将此目录放至 90 新目录后，在该目录中用已创建环境执行以下 **CPU 帮助/导入**，不加载预训练、数据或模型：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /mnt/dataX/ydf/projects/RGBT_campaign_90/environments/rgbir90/bin/python tools/train_rgbt_kd_detector.py --help
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /mnt/dataX/ydf/projects/RGBT_campaign_90/environments/rgbir90/bin/python -c "from tools import train_rgbt_kd_detector as m; import torch; assert 'native' in m.ARMS; assert 'native' not in m.CROSS_MODAL_ARMS; assert 'native' not in m.SAME_MODAL_ARMS; assert not torch.cuda.is_initialized(); print('NATIVE_IMPORT_CPU_PASS')"
```

这些命令不是 `--validate-only`：原 `--validate-only` 会读取 dataset/mapping 并加载通用预训练，需在 data preparation 完成后另行执行；它仍不读取假 T/R。真正训练调用形状为原 `--config configs/<模板> --arm native --seed 42 --batch 32 --output <新独立目录> --device 0`，**本包不授权/启动该调用**。

模板中的 `rebuild90` 是新增身份/预期参数 metadata，原 CLI 不检查其 readiness，须由根代理现有统一入口做准入。使用根代理已部署的共享 lease 环境，不使用本包默认的私有 ledger 路径，不另建队列。复制的历史 exposure ledger 只是旧函数的已知输入依赖，不表示新 90 暴露登记已成立。

保留原 E200 / B32 / workers8 / native loss 路径；完整任务 ≤10h 和短测只验性能的边界见 [NATIVE_REBUILD_PLAN.md](../NATIVE_REBUILD_PLAN.md)。包本身尚不包含新的性能计数/硬限时逻辑，不将原 `--max-steps` 调用次数误写为 AMP 成功更新数，也不以 `args.time` 改变原 LR 日程。
