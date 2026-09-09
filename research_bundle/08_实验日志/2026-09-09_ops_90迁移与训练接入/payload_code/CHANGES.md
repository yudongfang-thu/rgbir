# 90 训练代码端口（2026-09-09）

本包是待部署、待90实测的代码端口。未运行GPU、模型加载、训练、远端命令或新hash。C1/N/C0配置均为blocked模板，不能凭文件存在启动训练。blocked只表示技术资产/验证尚缺，用户已经授权90推进，不是额外用户确认门。

## 部署位置与环境

将本目录的内容部署到 `/mnt/dataX/ydf/projects/RGBT_campaign_90/`，保持 `release_gpu5/`、`tools/` 等相对结构。不要覆盖90已有的其他文件。Python目标为 `/mnt/dataX/ydf/projects/RGBT_campaign_90/environments/rgbir90/bin/python`。dispatcher还可通过 `RGBIR90_PYTHON` 指定解释器，通过 `RGBIR90_PROJECT_ROOT` 指定同一个项目根。

默认唯一lease为 `<项目根>/runs/.project_resource_leases.json`。全部dispatcher必须使用这一份资源池；没有固定GPU、预约峰值或94租约状态。保留原guard更严格的任务/显存限制，未自行放宽。

## 取材与修改

- `provenance/release_gpu5.original.tar.gz`：原冻结归档，字节复制保留，不是可直接用于90的配置；没有解压执行其中程序。
- `release_gpu5/`：由冻结归档解出。将原SpaceNet工程根和RGBT根替换为90项目根；`train_independent.validate_execution()`及legacy builder增加blocked模板检查，旧示例YAML也明确blocked。legacy项目根支持环境变量，嵌套dispatcher和compatibility输出前缀同步改90。其余trainer、criterion、paired loader、selection、observer、优化器和损失计算保留。
- `fast_candidate/batched_selection_v1.py`、`native_fast/native_fast.py`：原快算子直接复制。C1有历史94完整步骤测试，N有历史94步骤计时；C0快算子仅有历史CPU比较，不能宣称已完成90验证。
- `benchmark_review/cadence106.py`：保留6批热身+100批同步计时和实际optimizer计数；输出根改90；源码副本记录大小及路径，不生成hash；回执标识server=90。仍仅支持Drone C1/N、seed42、B32、17990训练图，不能直接当LLVIP工具或完整epoch。
- `production_entry/train_c1_fast.py`：保留原C1训练注入、完整E200和回执流程；有效计划必须为 `PORT90_READY`、server=90，且实际90 cadence/readiness均存在。原hash校验改为明确的accepted source/config副本字节比较；来源记录使用路径/大小/mtime，绝不把这些字段称作hash。未降低原 `admission.check_readiness()` 的科学配置检查。
- `dispatch_single_formal.py`、`tools/`：独立副本，路径指向90；共用project guard。`job.template.json` 的显存/RSS为null，且command为空，不能当作有效任务提交。
- `configs/research/jstars_dataset_exposure_v1.json`：回执工具所需的原数据曝光记录副本；属于来源信息，不代表新的90准入。

`PACKAGING_RECEIPT.json`记录复制源和路径修改。没有复制94的有效production plan、admission通过文件、峰值回执或GPU选择。

## 模板阻塞与后续接入

`configs/C1_s42.template.yaml`、`N_s42.template.yaml`、`C0_s42.template.yaml` 均含 `port90.status: BLOCKED`、`protocol_status: TEMPLATE_BLOCKED`，正式授权为false。训练入口遇到它们立即拒绝。

初始化路径已指向用户主任务在90准备的 `weights/pretrained/yolo11n.pt`；主任务报告该文件为80类generic初始化，但尚未证明与94权重逐字节同源。教师与参考仍为 `inputs/TO_BIND_*/weights/last.pt`，各自父run的args.yaml也必须随权重恢复，并匹配实际data YAML。

双模态YAML、pair mapping和HBB/split协议仍待与90已上传数据绑定；模板路径只给出原协议映射后的目标位置，不声称这些资产已存在。主任务正在生成 `data_attempt1/prepared/{dronevehicle,llvip}/` 下的数据YAML，完成后应据实际receipt更新派生配置。C1系数保留历史值用于代码模板，未声明已在90校准或获准训练。

接入完成后必须记录90实际数据/权重身份并产出该环境的计时及资源测量，才能填写派生配置和计划；不能改几个status字段就继承94通过结论。当前没有创建新的方法、实验矩阵或AP筛选任务。

## CPU smoke方法

本地仅做语法检查，脚本使用compile()，不产生pyc：

```powershell
python .\smoke_cpu.py
```

部署到90后，从项目根执行：

```bash
ROOT=/mnt/dataX/ydf/projects/RGBT_campaign_90
PY="$ROOT/environments/rgbir90/bin/python"
"$PY" -B "$ROOT/smoke_cpu.py" --runtime --project-root "$ROOT"
"$PY" -B "$ROOT/production_entry/train_c1_fast.py" --help
"$PY" -B "$ROOT/benchmark_review/cadence106.py" --help
```

`--runtime`隐藏CUDA设备，只导入依赖、构造criterion类并确认3份模板被阻止；不加载权重、不前向、不查询显卡、不占lease、不运行训练。期望 `CPU_IMPORT_PASSED_TEMPLATES_BLOCKED` 且 `cuda_initialized=false`。Linux guard依赖fcntl，因此Windows只运行默认语法检查。90 CPU导入和GPU计时未在本子任务执行。

版本应为Torch 2.10.0+cu128、Ultralytics 8.4.115、NumPy 2.2.6、SciPy 1.15.3、PyYAML 6.0.3；dispatcher另外需要psutil。

本地已实际执行：75个Python文件内存语法编译通过，C1入口及cadence入口的CLI帮助通过；见 `LOCAL_SMOKE_RECEIPT.json`。运行环境导入尚待主任务在90执行，不能把语法通过写成GPU训练验收通过。
