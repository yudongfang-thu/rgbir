# LLVIP 固定 2048 图子集：CPU 已完成，原完整 dev 保留

**已在 94 创建固定自然分层子集：2048/9619 图（21.29%），5592 person GT，14/14 治理 prefix 全部覆盖；原 dev 完整 2406 图/7879 GT 保持。没有训练、推理、图像读取或新文件 hash，也不解除原 L1 几何阻塞。**

远端输出：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/subset_llvip_v1/`。本地 [15 份逐字节收集回执](collection_receipt.json) 与 [CPU 绑定验证](cpu_validation_receipt.json)；总大小 2,987,177 字节。

## 直接可用输入

- [visible data YAML](remote_subset_llvip_v1/data_visible.yaml)、[infrared data YAML](remote_subset_llvip_v1/data_infrared.yaml)。train 为新 processed-alias txt；val 仍为原 `grouped_v1/{visible,infrared}/images/dev`。
- [原 canonical mapping 的选中子集](remote_subset_llvip_v1/visible_to_infrared_train.json)、[manifest / 分层分配与完整统计](remote_subset_llvip_v1/subset_manifest.json)、[模型路径和 stat](remote_subset_llvip_v1/checkpoint_identity.json)。
- [选中对象密度/尺度 metadata](remote_subset_llvip_v1/selected_metadata.jsonl)、[治理 prefix 适配表](remote_subset_llvip_v1/source_groups.tsv)。无第二套训练子集。

## 采样与 split 身份

复用上一轮 `build_subset.py` 的 `sample` 函数，完整源副本 [natural_sampler_source.py](natural_sampler_source.py) 与原文件 byte exact；只新增 [LLVIP metadata 适配器 v2](build_llvip_subset_v2.py)。seed20260908，source × 正样本图 GT 数四分位 × 每图 median sqrt(normalized w*h) 三分位，空图独立层；最大余数比例分配、随机打破余数并列、层内不放回等概率抽样，输出固定排序。118 层均遵守分配数距比例期望小于 1 图，不按类补底、不看 AP/teacher 输出。

source 来自原 `data/processed/llvip/splits/grouped_v1/fit.tsv` 的 `sequence_prefix`，不是自行用目录猜测。fit 9619 图、26251 GT、2 空图；dev 2406 图来自原 `dev.tsv`，7879 GT；两者 stem 与 prefix 均无交集。fit prefix 为 02/03/05/06/08/09/10/11/13/14/15/16/17/18，dev 为 01/04/07/12/25。未读取 test.tsv 或 test 图像。dev 标签只用来验证既有身份/计数，不进入分层/选图。

子集 2048 图全含 person、5592 GT、无空图，是自然分配及舍入结果，不为保留 2 个极少空图另行补采。14 个来源全部保留，但这不证明每个细粒度场景都被保留，也不能把图/GT 计数当梯度份额或定位信息质量。

LLVIP 原 mapping 使用 raw/LLVIP/{visible,infrared}/train canonical 路径。新训练 txt 必须使用 processed/images/fit 别名才能保持 YOLO 标签推导正确；builder 验证这些别名的 resolve 目标与 mapping 及治理 TSV exact，mapping 本身保留原 canonical 子集。paired loader `_canonical` 原先也将 key/value resolve 后索引，因此二者有明确绑定。IR/RGB train 标签 bytes 相同仅是标注共享，不能证明物理几何或认证 L1。

## 模型与运行入口

学生 warmstart/参考旧 visible42：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/llvip/visible_seed42_native_b32a2/weights/last.pt`。

IR 教师：相同父目录 `infrared_seed42_native_b32a2/weights/last.pt`。两个原 `args.yaml` 与原 data YAML 小副本在 [originals](remote_subset_llvip_v1/originals/)，逐项确认 data 路径、seed42、E200、B32、640、workers8、SGD、lr0=.01/lrf=.01。checkpoint 只 path/bytes/mtime stat，未加载权重；它们是旧 baseline，不能改称新协议的 N 端点。

远端 builder：`.../rgbir_direction_screen_20260908/subset_builder_llvip_v2/build_llvip_subset_v2.py`；用同目录原 sampler，pinned Python 下 `--self-test`，生成时 `--output <尚不存在的新目录>`。固定默认 size2048/seed20260908。本地部署/收集入口 [prepare_on94_v2.py](prepare_on94_v2.py)，已执行 [remote_build_receipt_v2.json](remote_build_receipt_v2.json)。不要重跑覆盖现有输出。

首次 builder v1 错将原 canonical mapping 当作 processed mapping，在严格路径断言处停止，尚未创建子集，无 GPU；保留原 [失败回执](remote_build_receipt.json) 和源，不覆盖远端 v1。v2 仅修显式身份适配，采样公式及 seed 未改。

新 BN/LR 协议和科学边界见 [短评](PROTOCOL_RISK_REVIEW.md)。Drone 继续使用上一轮固定 2048 子集及同协议 weight0 N42 起点；本任务没有重采 Drone。
