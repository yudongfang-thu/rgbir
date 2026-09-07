# 历史 native 与当前 weight0 的只读审计（2026-09-07）

> **当前 weight0 没有获得非零 KD 监督，但它不等于历史 native 的完整训练轨迹。已实测 workers 8→4 会从第5个 batch 改变真实 RGB 增强图像和标签；这可以造成端点不同，但尚未因果分解其对 AP 的贡献。当前 weight0 相对 CGKD W1 历史 native 是2/3 seed上升，平均+0.468864 pp，并非三seed全同向。**

## 审计对象与范围

本报告中的“历史 native”专指 `94:/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/cgkd_w1/native_rgb_s{0,42,123}_e200`。项目还存在 `rgbt_p3_causal_v1/formal_native` 等其他 native 目录，不能混用。当前 N/C 的独立端点取 2026-09-07 11:19 快照。只读代码、配置、现存权重与数据；新增检查只在 CPU 执行，没有训练、评估模型 AP、占用 GPU 或修改在跑任务。

原始小产物在 [native_raw_evidence](native_raw_evidence)，机器可读比较在 [native_comparison.json](native_comparison.json)。

## 一、历史差值被拆成了什么

指标 mAP50–95，数值为百分数、差值为百分点。三seed均值使用样本SD。

| seed | 历史 native | 当前 weight0 N | 当前 C | N−历史 | C−N | C−历史 |
|---|---:|---:|---:|---:|---:|---:|
|0|54.357648|54.346185|54.772186|−0.011463|+0.426001|+0.414538|
|42|53.815556|54.513608|54.658162|+0.698052|+0.144554|+0.842606|
|123|53.687433|54.407436|54.636847|+0.720004|+0.229411|+0.949415|
|mean±SD|53.953546±0.355778|54.422410±0.084710|54.689065±0.072770|+0.468864±0.416121|+0.266655±0.144373|+0.735520±0.283062|

平均 `C−历史 native = (N−历史 native) + (C−N)`，即 `+0.735520 = +0.468864 + +0.266655 pp`。这是代数分解，不是“workers贡献0.469 pp”的因果估计。seed42 的旧 +0.843 pp 中，+0.698 pp 已经出现在无KD的当前N里，KD同代码净差只有+0.145 pp。

因此用户质疑有依据：不能将历史参照差当作 KD 增益。另一方面，N−历史不是三个seed全正，而C−当前N确实是三个seed全正。当前三seed支持的是这套冻结工程与数据流下的有限净收益，不能因此证明跨模态内容归因或统计显著性。

## 二、哪些变了，哪些没有变

直接比较 seed42 `args.yaml`，非路径性质差异如下；完整字段差异留在 JSON。

| 项目 | 历史 CGKD W1 native | 当前 OEv1 weight0/C | 证据与含义 |
|---|---|---|---|
|训练入口|`YOLO(pt).train()`，标准 DetectionTrainer|直接构造自定义 ObjectEvidenceTrainer|确有工程路径变化，不应声称原生完整复现|
|训练 DataLoader|8 workers|4 workers|**真实CPU回放证明随机增强轨迹改变**|
|RGB读取|原生 YOLODataset|原生学生 dataset 外包 DualLabelRGBIRDataset|学生增强执行一次；IR图/独立GT做相同RNG重放并恢复学生后继RNG|
|训练期验证与绘图|`val=True, plots=True`|`val=False, plots=False`，覆写validate/final_eval|旧每轮验证，新完整预算后独立评估；尚未隔离其对轨迹的贡献|
|辅助教师|无|C/N都加载同一冻结IR教师与RGB参考，执行相同选择计算|N权重字面为0；无非零教师监督|
|设备args|空字符串|逻辑`0`|物理GPU仍由lease/CUDA_VISIBLE_DEVICES配置，不能据此推断换多卡或训练预算|

两边记录相同：通用 `yolo11n.pt` 初始化文件路径、同RGB数据YAML、E200、imgsz640、batch32、nbs64、SGD、lr0/lrf=0.01、momentum0.937、weight_decay0.0005、warmup3、AMP、deterministic、rect=False、cache=False、patience0、close_mosaic0。增强均为translate0.1/scale0.5/fliplr0.5，mosaic/mixup/cutmix/HSV关闭。不能再以旧目录名 `b32a2` vs `b32_e200` 认定有效batch不同。

同一路径和超参数并不证明历史每个文件的所有字节、运行时源码和每个batch完整相同；历史回执没有当前完整源码/样本轨迹审计强度。

## 三、这次新增的真实 CPU 检查

脚本：[native_cpu_replay.py](native_cpu_replay.py)，原始输出：[native_cpu_replay_stdout.txt](native_cpu_replay_stdout.txt)，结构化结果：[native_cpu_replay_result.json](native_cpu_replay_result.json)。`CUDA_VISIBLE_DEVICES=''`，Torch线程4，检查末尾 `cuda_initialized=false`。未保存任何原图、训练权重或数据副本到新目录。

### 3.1 两种建模入口是否偷偷换了初始化

分别重放旧 `YOLO(pt)`后经`get_model(weights=model,cfg=model.yaml)`路径，与新`setup_model()`从pt经`get_model`路径。相同seed0/42/123、当前pinned代码、相同真实数据class names：

- 每seed **499/499 state tensors 精确相同**；Torch RNG后继精确相同。
- 两路径都报告80类→5类、按类名映射3/5分类行、加载451/499张量。
- 因此，没有发现本次 N 因构造路径获得“更好的5类检测头初始化”。此前 OS-SSL clean五类模板的初始化混杂是另一件事，不能移用为这里的结论。

限制：这是当前pinned源码下对构造路径的重建，不是恢复历史run开始时从未保存的完整初始模型与回调/RNG状态。它排除了一个具体入口差异假设，不能证明历史整程等价。

### 3.2 workers真的是只有性能区别吗

用真实 Drone 17990张训练集、冻结增强配置、相同seed42、相同原生dataset与原生build_dataloader，分别4和8 workers；比较前10个batch。

- 10/10批原始图像文件顺序完全相同。
- 第1–4批抽查首张增强图像、全部该batch标签，精确相同。
- **从第5批起，第5–10批的首张RGB增强图像和该batch标签全部不同。** 每张首图变动约103.8万–120.1万个uint8通道值。
- worker数改变了分给每个worker的样本及其局部RNG推进；虽然几何增强的概率分布相同，具体翻转/缩放/裁剪实现不同，所以训练轨迹不同。

这也是为什么“首批RGB一致”不能证明workers不同的长训数据流相同。不能推导workers4一定更好，亦不能将平均+0.469 pp全部归因给这一项，需固定其他条件的消融才能量化。

另一个限制：当前pinned `build_dataloader` 的独立generator固定为`6148914691236517205 + RANK`，不含实验seed；已有当前三seed前批数据流检查一致。现有0/42/123重复主要覆盖随机任务头初始化和其后训练差异，未覆盖三条独立增强种子。三seedSD很小不等于完整随机性已经充分覆盖。

## 四、weight0是否仍通过别的路径用了IR

原 `EvidenceCriterion.__call__`：先调用原生检测criterion，再对teacher/reference做`no_grad()`且`eval()`前向，执行相同paired选择与KD计算，**最后**按arm选择`weight=0.0`或0.1。weight0没有绕过T/R计算的提前返回。总损失为：

`native_total.sum() + actual_batch_size * weight * kd`

对于weight0，有限KD前提下加权KD严格为0。现有真实canary核验同batch的native loss与score gradient精确相同；teacher/reference没有梯度、不在optimizer/EMA/部署state。IR独立GT仍被读取并用于辅助计算/日志，所以应称“有额外IR计算、零IR监督剂量的工程对照”，不应把它称为运行时完全不读IR的RGB-only实现。但未发现IR内容通过非零梯度偷偷改变N的证据。

DualLabel loader的学生原生增强只执行一次；IR增强使用保存的Python/NumPy/TorchCPU RNG，finally恢复学生after状态。已有合成与真实loader等价检查支持这个局部合同。它不修改RGB GT监督、不按IR删去学生样本。原始mapping完备性异常会失败，不会悄悄过滤训练图像。

近期“新旧 C/N 精确等价”中的“新旧”是 **Task-Conditional实现 vs 已冻结OEv1实现**，不是本报告中的历史CGKD标准native vs 当前N。其适用范围不能扩大。

## 五、实际更新预算和端点口径

当前N/C六run均完成112600个batch、56722次optimizer尝试与56722次EMA更新。成功优化步与AMP跳步如下：

| seed | N成功更新 / AMP skips | C成功更新 / AMP skips |
|---|---:|---:|
|0|56697 / 25|56694 / 28|
|42|56695 / 27|56695 / 27|
|123|56696 / 26|56695 / 27|

EMA沿用native：GradScaler跳过optimizer时仍进行EMA更新。当前协议锁定E200和batch/attempt schedule，没有承诺各臂成功更新数必定相同；差1–3次已经真实记录。旧native完成回执没有同强度的实际成功步/skip轨迹，不能声称与当前N严格相同。

历史指标取`metrics_record.json`，其checkpoint明确为`weights/last.pt`、同RGB train/val-only YAML和val角色。旧适配器和新评估脚本核心均调用`YOLO(last.pt).val(...)`并取`metrics.box.map`，不是拿历史best和当前last比较，也不是拿CSV占位0比较。新评估明确固定full1469图roster/last-EMA，workers4；历史通常workers8、独立receipt较薄。训练期val/plots差异和历史源码冻结缺口仍存在，尚未将旧last用当前评估合同重新复算，故暂列历史参照，不能充当正式N。

## 结论与建议

1. 用户担心是合理的：**历史C看起来更大的增益中，多数已在当前无KD N上出现；正式KD净差必须继续使用C−同代码N。** 当前N不是“有暗中非零蒸馏”的证据，而是工程/增强轨迹不同的基线。
2. 已有的CPU证据把“为什么可以不同”具体落到了实际输入上，而不是泛称随机误差。尚不知道workers、val流程或其他未记录差异分别贡献多少AP，不应用推测替代消融。
3. 后续轻量验收最有价值的是原生trainer适配到**同workers4、同禁val/plots、同初始化、同样本增强**，与当前weight0作初始化/多batch/optimizer/EMA轨迹检查；通过则当前N可以更清晰解释为统一配方的native。若要量化旧新+0.469 pp的来源，需要单因素完整E200对照，不能从短测给AP归因。本次未追加长训。
4. 比较CMDistill/CCLKD等历史方法也要重新检查/统一同一工程recipe；不能利用当前N整体提高后的端点差，夸大方法相对历史基线的优势。

## 路径与复核入口

- 本地当前端点：`../2026-09-07_audit_RGBIR实施起点/snapshots/2026-09-07T111931.041292_0800/`。
- 历史产物：`../2026-09-06_audit_RGBIR晚间进度与新结果/osssl/raw/runs/cgkd_w1/`（目录名osssl是当时收集器的保存位置，文件实际身份仍是CGKD W1 native）。
- 当前实现：`../2026-09-06_audit_RGBIR晚间进度与新结果/oev1_snapshot/raw_runs/full_paired_s42_attempt1/run_evidence/source_snapshot/trainer/01_train_object_evidence.py`及同目录`02_paired_rgbir_data.py`。
- 本次CPU只读重放通过SSH stdin，未产生94新训练目录；脚本与结果原件均在本条目。
- 原始小文件与当前pinned源码只读副本：[native_raw_evidence](native_raw_evidence)；重建比较脚本：[assemble_native_evidence.py](assemble_native_evidence.py)。
