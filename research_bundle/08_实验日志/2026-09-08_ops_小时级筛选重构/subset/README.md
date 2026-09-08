# 已生成：2048 图自然分层训练子集

**94 已生成固定 seed20260908 的 2048 图子集，五类均有覆盖；不做类别补底、交换或第二套训练子集。** 主用途是同一 baseline warmstart 下 N-ft/C0-ft/C1-ft 的三轮快速筛查；B32 时每轮 64 批、每臂 192 批。没有启动训练、推理或 GPU 工作。

## 来源与固定采样

完整 train 配对来自原 prepared mapping，恰为 17,990 对；source-group 来自原 `hbb_v1/rgb_train_source_groups.tsv`。这份 TSV 已有本地副本，但既有 probe 的 per-image 表只覆盖抽样图，不能作为全 train 标签统计。本次只读取 17,990 个 RGB GT 文本（共 13,192,432 bytes），检查对应 IR 标签存在；不打开、扫描或计算图片内容，不读取模型预测。

每图记录 GT 数量、五类计数、GT 相对尺度中位数。相对尺度定义为 `sqrt(normalized_width × normalized_height)`，它是相对图像面积，不是 feature stride 或像素尺度。仅用完整 train 的正样本图定义分箱：密度四分位切点为 6/12/21，尺度三分位切点为 0.0697897232316478/0.08731461658617401；空图另成一档，边界采用 `bisect_right`。

按 source-group × 密度档 × 尺度档形成 662 层。先按各层原图数比例分配 2048 个名额，整数取整后用最大余数法补足；同余数用固定 RNG 打破平局，再在每层等概率无放回抽样。每层选中图数与比例期望相差不到 1 图。最终名单按原 RGB 路径排序；真正训练的顺序与增强由新筛查任务自己的固定训练 seed 控制。

**采样输入不包含 dev AP、IR 预测或现有方法表现；类别计数只用于密度及覆盖报告，不施加类别配额。** 保留原 split、原 RGB→IR 对应和 processed 路径别名，避免把图像路径 resolve 到 raw 目录后破坏标签推导。原完整 dev 的两个路径不变，1469 图身份沿用已有合同；本次没有重读 dev 图或标签。

## 实际覆盖及分布边界

|类别|完整 train 含类图数|子集含类图数|完整 train GT 数|子集 GT 数|含类图比例变化（pp）|
|---|---:|---:|---:|---:|---:|
|car|16566|1894|246699|28654|+0.3960|
|freight car|3836|433|8712|936|−0.1804|
|truck|4503|491|13685|1669|−1.0560|
|bus|3095|366|10421|1155|+0.6671|
|van|2971|330|7275|782|−0.4014|

图像保留率为 2048/17990=11.384%；GT 从 286,792 个变为 33,196 个。空图从 512 张中保留 57 张。原 source-group 为 86 个，子集覆盖 63 个；小组可能因比例取整而未入选，这个局限保留，不补抽。source-group 是原来源文件夹分组，不能当作物理配准认证或采集独立性证明。

这是一份近似保留所选联合分层分布的子集，不保证逐类分布或完整 train 的方法排序不变。当前五类都未缺失，因此没有另行覆盖补救。含类图数与 GT 数均不是 KD 梯度份额。若后续某类有效选择很少，应如实限定该 screen，而不是看 dev AP 后改名单。

baseline N42 已经过完整 train 训练；这里的子集只限制后续微调的曝光和计算，不是“从零只用约 11% 数据训练”。三轮结果用于后续方向筛查，不替代完整训练、多 seed 或归因实验。

## 可用入口

远端产物目录：

`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_hourly_screen_20260908/subset_v1/`

- RGB data YAML：`data_rgb.yaml`；IR data YAML：`data_infrared.yaml`。
- train 名单：`train_rgb.txt`、`train_infrared.txt`，各 2048 条。
- 子集配对：`rgb_to_infrared_train.json`，2048 对，保留原路径与对应关系。
- [subset_manifest.json](remote_subset_v1/subset_manifest.json)：来源 stat、分箱、662 层名额、完整与子集五类统计及边界。
- [train_metadata.jsonl]（服务器/本地保留，未包含于本阶段发布：remote_subset_v1/train_metadata.jsonl）：17,990 条 train 标签元数据及 selected 标记，供只读复算；无图像或模型输出。
- [checkpoint_identity.json](remote_subset_v1/checkpoint_identity.json)：RGB N42 warmstart/reference、IR42 teacher 的原路径、size 与 mtime。仅 stat，不声称加载或验证了权重内容。

RGB warmstart/reference 原路径：

`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/rgb_seed42_native_b32a2/weights/last.pt`

IR teacher 原路径：

`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/infrared_seed42_native_b32a2/weights/last.pt`

原 data YAML 与配对来源保持不变，位于 `artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/`。新 YAML 只把 train 指向子集名单，val 保留原完整 dev；没有修改正式训练 YAML。新筛查执行入口仍需绑定这些新数据身份，不得静默套用“17990 train”的旧验收字段。

## 脚本与实际验证

[build_subset.py](build_subset.py) 是独立 CPU 构建器；94 执行副本为 `artifacts/rgbir_hourly_screen_20260908/subset_builder_v1.py`。实际参数完整记录在 [remote_build_receipt.json](remote_build_receipt.json)，入口为 [run_cpu_builder_on94.py](run_cpu_builder_on94.py)。输出目录已存在，重复执行会拒绝覆盖；需要复跑时只能显式使用新的 attempt 目录并保持已固定规则。

本地与94基础真值检查通过。收回的 8 个小元数据文件共 11,120,934 bytes，均做 byte copy 比较，见 [collection_receipt.json](collection_receipt.json)。随后用本地 CPU 从完整 train 元数据重新计算，同样的 2048 名单、分箱、662 层名额及覆盖统计全部一致，两个 train 名单与 mapping 顺序/成员一致且无重复，原 val 路径保持；见 [cpu_validation_receipt.json](cpu_validation_receipt.json)。

未修改原 mapping、split、标签、原 YAML 或 checkpoint；未读图像、加载权重、使用 GPU、使用模型表现选样或计算新文件 hash。当前状态只说明子集产物可用，不代表训练已经准入或已经取得 AP。
