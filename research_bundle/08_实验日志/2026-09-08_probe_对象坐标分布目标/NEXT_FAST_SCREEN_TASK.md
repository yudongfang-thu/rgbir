# 下一心跳任务：2048 子集 N/C0 筛选器校验

**不重复全量 E8。** [已完成三臂](../2026-09-08_train_分类快速反馈E8/README.md) 的 N/C0/C1 mAP=43.376542/43.920584/43.408107，C0−N +0.544042 pp，队列约2h40m，已回答“C0是否出现过早期正向信号”。唯一值得新增的问题是：**更便宜的固定子集能否保留这一已知方向？** 这是已知结果驱动的筛选工具校验，不是独立方法增益验证。

**固定任务：** 新条目/新 artifact 使用 `DRONE_SUBSET2048_PRETRAIN_E8_CHECK`，只跑 N/C0，λ=0/.1、seed42；每臂8轮×64批=512批，独立8轮日程、last EMA、结束后各一次完整 dev；不增加 C1、不校准或扫 λ。

**现成数据与初始化：** 复用94 `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_hourly_screen_20260908/subset_v1/` 下 `data_rgb.yaml`、`data_infrared.yaml`、`rgb_to_infrared_train.json`。固定2048图/33,196 GT、五类齐备、63/86来源组，不重采样；完整 dev 1469图/22,462 GT不动（[子集合同](../2026-09-08_ops_小时级筛选重构/subset/README.md)）。学生用原通用预训练 `/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/artifacts/int8_cross_modal_stage2_v1/weights/yolo11n.pt`，T/R原IR42/RGB42路径直接继承[E8实际配置](../2026-09-08_train_分类快速反馈E8/launch_evidence_attempt1/source/configs/drone_C0_s42_E8.yaml)。保留img640/B32/nbs64/workers4/AMP、SGD lr0=.01/lrf=.01 linear、warmup3、momentum .937、weight decay .0005、原增强及正常BN训练、fresh optimizer/EMA；不使用成熟初始化或本次BN冻结FT3配方。

**最小实现：** 复制[已执行 E8 runtime](../2026-09-08_ops_训练吞吐诊断/short_screen_E8_release/)到新目录，保留 pinned `build_trainer(..., historical=False)` 与原 N/C0 算式/门/base分母。改 common/train/evaluate/queue 的 scope/endpoint、train2048、两臂、新配置和预算绑定；原17990/三臂/12h硬编码使其不能仅换YAML直接运行，旧admission也不能复用。冻结T/R仍核原完整data身份，loader才用子集；复用[已执行绑定](../2026-09-08_ops_小时级筛选重构/release/train_hourly.py)的 `auxiliary_data_identity`，不改旧源。

**技术门和时间上限：** 各新解释器先24成功update canary，最多48次尝试；核N/C0初始学生张量、共同流前缀的像素/双标签、有限native/梯度、非零C0 KD及实测资源。五类头沿原生初始化，只要求两臂初始化互相一致，不能要求等于原80类checkpoint全张量。完整训练重置回共同初始化，并与canary流前缀核对。复用全局lease，**项目执行硬上限45分钟，排队另记**；正常cadence实测外推两臂，至少20%训练余量并计入canary/初始化/两次eval，预计超预算就blocked；超时留incomplete，不缩轮数补救。审计时间单列。[当前C1实测](../2026-09-08_ops_C1训练路径提速/README.md)1.748899→.221360秒/批（7.90×）不等于C0吞吐或epoch倍数；本任务不迁移该候选、不接管C1长训，原C0是否够快由canary决定。

**一次性判读：** accepted analyzer核全dev后报告raw AP和C0−N。正向只表示这一已知对照在子集/单seed/E8方向保留；非正向表示本版廉价筛选器未通过该校验，停止本版，不重抽子集或扫日程，更不能推出C0无效。子集曝光远少于全量E8，任何结果均不证明一般方法排序、跨模态归因或E200增益。当前仅准备，未启动GPU/训练、复算旧AP或修改旧源。
