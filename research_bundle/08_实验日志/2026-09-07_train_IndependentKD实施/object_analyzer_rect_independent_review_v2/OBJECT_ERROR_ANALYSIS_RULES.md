# 固定操作点对象诊断规则 rect v2

**状态：DRAFT / NOT_ACCEPTED；等待独立复核，不用于真实分析。** 2026-09-07，仅修正原v1把名义imgsz640误等同实际canvas最长边640的问题。原v1已接受三文件、回执及真实失败attempt保留。匹配阈值、面积阈值及统计分母未变，未分析新C1 AP，未改训练release或原对象产物。

## 匹配与对象身份

- 输入是新版 evaluator 的 `evaluation_val.json`、其 `objects.jsonl.gz`、实际 `evaluation_contract.json` 与完整 eval receipt。先核原字节副本、checkpoint/seed/arm/source/config、完整 dev（Drone1469 / LLVIP2406），再处理对象。两个端点同 seed/dataset/endpoint/evaluator kwargs，N 为 paired。训练 recipe 是否可作正式对比，继续由已接受 `analyze_independent.py` 核验。
- N 与候选的图像集合、逐图 canvas/original shape、GT boxes 和 GT classes 的**完整数组及原顺序**必须相同。样本主键为 JSON `[image, gt_index]`。不按 IoU 重排 GT，不用 GT 坐标字符串合并重复标注；重复图像行直接报错。
- 仅用原生 NMS 后 evaluator 保存的预测；不重新 NMS。confidence≥.25。预测按 confidence 降序、原预测序升序排列；依次匹配剩余同类 GT 中 IoU≥.50 的最大者，同 IoU 取原 GT 序较小者。一对一。两个模型独立执行同规则，不用另一模型决定它的匹配。此固定操作点诊断不等于原生 AP 匹配或 best-F1 precision/recall。
- 无同类匹配视为对象不正确。另记录类别错误（他类 IoU≥.50）、同类定位不足（.10≤IoU<.50）、粗候选类别错误、分配竞争或当前 confidence 下无粗候选。无粗候选只描述操作点，不推断 raw anchor 中无观测。

## 修复、损伤与误检

- repaired：N 不正确、候选正确；分母为 **N 不正确的 GT 对象数**。
- damaged：N 正确、候选不正确；分母为 **N 正确的 GT 对象数**。
- stable_correct / still_incorrect 分别记录；任何分母为0，rate为 null，不写0或用 epsilon。所有 rate 是0–1比例，同时保留分子、分母。
- 每个预测 background FP 当且仅当 confidence≥.25 且 `max IoU(任意类别GT)<.10`；无GT图所有保留预测均为 background。IoU 恰 .10 不算背景。错类但框贴近对象、重复检测不算背景。
- 保留预测分为 correct / background / duplicate_or_assignment_competition / wrong_class / localization_or_mixed。优先级依该顺序，互斥计数。背景 FP/image 的总分母为完整 dev 图数，包含无 GT 和无 FP 图。

## 分组

尺度使用实际输入网络的xyxy框面积，名义imgsz640同时由bound config及receipt-bound contract.effective_kwargs核验；实际canvas和original_shape则来自另行receipt-bound objects，必须为有限正整数。contract没有记录shape，不把其中observed_images计数冒称shape证明。按contract.actual_loader_roster及实际batch设置重建观察顺序，核每个实际batch的objects具有共同canvas；不由原图纵横比猜测或强制重算canvas。N/候选仍逐图严格匹配canvas、original_shape、GT原数组及顺序。

实际pinned Ultralytics8.4.115矩形评价会额外padding：build.py评价pad=.5，base.py以ceil(shape×imgsz/stride+pad)×stride给实际batch画布。Drone真实512×640图可形成544×672输入；原框仅平移，面积不因padding改变。本版不缩放框，不改写原shape或contract，也不只硬编码接受544×672。混合原图纵横比允许共用一个实际batch画布。

尺度标签迁移为 `input640_small_lt32sq`（area<32²）、`input640_medium_32sq_to_lt96sq`（32²≤area<96²）、`input640_large_ge96sq`（area≥96²）；input640指名义输入imgsz，不指实际最长边。与v1的canvas640_*一一更名，32²/96²阈值原封不动，不使用原图面积或称COCO原图尺度。GT分组用GT框，背景FP尺度用预测框；两种总体分别报告。名称迁移与实测失败依据见本目录README及change_receipt.json。

类别以 GT class_id 分组；类别名称可从冻结 metadata 的 `class_names` 读取，没有名称标 UNKNOWN，ID 不丢失。来源和亮度仅从冻结现有映射读取，不从当前预测、文件夹名或新图像统计猜测；缺失为 UNKNOWN。亮度必须附既定代理定义，不能当真实昼夜标签。

对象分组分别按类别/尺度/来源/亮度统计自己的 N正确/N错误分母。背景 FP 按预测类别/预测尺度分组时使用**全部图像**作分母；按来源/亮度分组时使用该图像组的全部图像数。即便该组无GT也保留背景统计，避免漏掉无目标图。

## 输入 metadata 与运行接口

```text
python object_error_analysis.py --baseline-evaluation <N/evaluation_val.json> --candidate-evaluation <arm/evaluation_val.json> --output <new-analysis-dir> [--metadata frozen_metadata.json] [--source-groups-tsv rgb_val_source_groups.tsv] [--review-receipt accepted_review.json]
```

冻结 metadata 示例（既有映射的导出格式，不要求重做特征图）：

```json
{"frozen":true,"key_type":"image","brightness_definition":{"source":"既有冻结亮度分桶","is_day_night_label":false},"class_names":{"0":"car"},"images":{"/absolute/image.jpg":{"source_group":"known_group","brightness_bin":"low_proxy"}}}
```

`key_type` 可为 image 或 stem；stem 在 dev 中不唯一时拒绝。已有来源 TSV 支持前两列 stem/source_group（无表头亦可），两份已有映射冲突时报错；未找到条目保持 UNKNOWN。

## 产物与接入边界

产出 N / 候选各一份 error_analysis、pair_summary、逐对象/逐图 gzip JSONL、输入小证据及分析源码原字节副本、analysis_receipt。只读取预测与标签数字，不读取原图，不推理，不覆盖已有输出。

已接受 `analyze_independent.py` 期望的 `contract` 内容为 confidence=.25、match_iou=.50、coarse_iou=.10、background_definition=all_gt_iou_below_0.10、background_unit=false_positives_per_image；同时需要 checkpoint、seed、完整 roster 和 background_fp_per_image。该已接受 loader 目前不检查本工具的接受状态，因此**默认只发 draft_contract，不发 contract**。独立审阅者用 `rgbir-object-error-review-v1` / status=ACCEPTED、source_files 的 relative/accepted_copy 绑定本脚本、fixture、规则三份源码后，运行时 `--review-receipt` 才发布可接入 contract。自测通过本身不接受。

输出的 `localization_diagnostics_consistent` 固定 null。本工具只提供完整检测操作点的修复/损伤，不能代替 L1 的同 anchor 定位机会/几何/梯度诊断，不能据 repair>damage 自动释放 L 扩展。任何 harm 只用于复核与暂停扩展，不停止既定 E200 训练。
