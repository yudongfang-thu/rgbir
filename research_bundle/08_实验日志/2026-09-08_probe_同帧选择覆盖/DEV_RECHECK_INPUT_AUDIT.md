# 完整 dev 检测机会重核：现有输入可用性

**LLVIP 已有可直接用于 CPU 重核的完整旧 native RGB/IR post-NMS 缓存及逐图 GT 配对；Drone 已有完整 N/C0 三 seed 缓存，但本轮本地清点未找到同一完整 dev 的 IR42 post-NMS 输出。建议下一项只先做 LLVIP 完整 dev 的固定阈值一对一机会表，不需要 GPU。** 本轮仅读合同/既有接受报告及少量 schema 行；没有计算新的全量机会数字或运行新实验。

## LLVIP：可用，完整配对已接受

根目录：`E:/SHARE/光sar/08_实验日志/2026-09-07_probe_双数据集证据优先推进/llvip_full_eval/`。

| 输入 | 准确相对路径 | 本地存在/schema检查 |
|---|---|---|
| RGB N42 全 dev | `remote_completed_attempt2/N42_full_attempt1/capture/objects.jsonl.gz` | 1,334,078 bytes，第一行已读 |
| IR T42 全 dev | `remote_completed_attempt2/T42_full_attempt1/capture/objects.jsonl.gz` | 714,925 bytes，第一行已读 |
| 跨模态逐图身份 | `verified_pair_manifest.jsonl` | 显式 pair_key/rgb_image/ir_image/GT数/canvas/original_shape |
| 完成/跨模态验收 | `completed_verification_receipt.json` | 已有根验收；独立复核见下 |

每个模型任务根另有 `capture_contract.json`、`capture_metrics.json`、`population.json`、`model_identity.json`、`summary.json`、`sources/{3_val.py,4_validator.py,6_nms.py,7_metrics.py}`。上层 `ap_error/llvip_manifest.json` 已逐项绑定这些输入；`ap_error/llvip_independent_review/EXPERIMENT_AUDIT.md` 与 `verification_receipt.json` 已确认：2406键唯一且相同，逐图 GT类/框顺序、canvas、original_shape exact；alias→canonical 为显式双射，不是 basename 猜配。两模型分别完整 2406 dev图/7879 person GT，0无GT图，N有1空预测图、T无空预测图，空预测图仍保存整行GT。

JSONL 每行是一张图，字段为：

```text
image: 实际 processed alias 路径
canvas_shape: [H,W]
original_shape: [H,W]
gt_boxes: [[x1,y1,x2,y2], ...]
gt_classes: [0.0, ...]
pred_boxes: [[x1,y1,x2,y2], ...]
pred_classes: [0.0, ...]
pred_confidence: [probability, ...]
```

GT/pred 均是 native rect 输入画布绝对 xyxy（本次已接受544×672），不是原图坐标，也不是旧对象诊断的640×640静态 letterbox。预测是原生后处理后的完整有序框数组，conf=.001、NMS IoU=.7、multi-label、max_det300、FP32、无augment。全部输出框 N=28406/T=12929；最大每图62/27，没有达到max_det300。只保留该后处理的输出，不含全部 raw anchors 或被 NMS 抑制框。

两模型是旧 E200 native seed42：

- N：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/llvip/visible_seed42_native_b32a2/weights/last.pt`
- T：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/llvip/infrared_seed42_native_b32a2/weights/last.pt`

模型身份回执记录 stripped checkpoint_epoch=-1、ema_present=false、loaded_source=model；不能伪称另检查了EMA字段。这两个模型对应当前 LLVIP FT 初始 S/R 与 T 的历史模型路径，但完整dev推理是eval/FP32/native rect，本轮首批是train/冻结BN/AMP/训练增强，**不是同批、同 preprocessing 或同端点 FT3 输出**。N42旧标签也不是新协议N的三seed结果。无效的外层 campaign attempt1 全GT丢失不得接入；这里 `remote_completed_attempt2` 内部任务叫 `full_attempt1` 是有效第二轮任务编号。

## Drone：N/C0 完整缓存可用，完整 T 尚缺可核验输入

本地根：`E:/SHARE/光sar/08_实验日志/2026-09-07_train_IndependentKD实施/legacy_diagnostics_snapshot_20260907_161459/`。

六个准确路径模板：`{N,C0}_s{0,42,123}_attempt1/predictions/objects.jsonl.gz`。每任务根的 `evaluation_contract.json`、`evaluation_val.json`、`reevaluation_receipt.json` 及 `eval_evidence/` 提供完成/模型/源绑定；统一输入表是 `../2026-09-07_probe_双数据集证据优先推进/ap_error/drone_results_v1/input_manifest.json`。

已接受完整1469 dev图/22462 GT，五类顺序 `car, freight car, truck, bus, van`（0–4），schema同上；conf=.001、NMS IoU=.7、max_det300、FP32/native rect，保留空预测图。N42缓存3,583,101 bytes，第一行已核。部分Drone图达到max_det300，完整性只能指原固定后处理输出，不能声称可重建全部 raw responses。

N42是 `runs/rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1/weights/last.pt`，与当前Drone FT初始学生路径一致；不是旧 `formal_native/dronevehicle/rgb_seed42_native_b32a2` 的R模型。期望teacher是 `runs/rgbt_p3_causal_v1/formal_native/dronevehicle/infrared_seed42_native_b32a2/weights/last.pt`。本轮对本地 JSONL/gz/npz/预测文件清点未找到这个teacher的**完整1469图 native post-NMS+GT+合同**，所以不能以现有六端点完成Drone N/T机会交叉表；也不据此断言服务器从未保存，只是本地尚无可用绑定。

`2026-09-07_probe_Baseline蒸馏机会重诊断/remote_exports/dronevehicle_full_attempt1/` 虽然目录名含full，实际上是1024train+200dev的旧 raw/object子集，`objects.jsonl`、`logits.npz`、`features.npz` 不可冒充完整dev post-NMS。TaskConditional d1/d2、natural_flow selection表也不是完整dev检测缓存。旧200dev的N/T assigned低置信统计仍只是空间分配代理，不能替代此缺口。

## 下一项最小固定协议（本轮未执行）

1. 只取上述 LLVIP N42/T42 两个已接受全dev缓存，先核既有pair manifest、alias合同、逐图GT数组/顺序/画布与7879总GT。GT身份定义为登记pair_key+该图接受GT行号；不得把本轮train stable IDs混入，也不靠最近邻匹配类别框猜身份。
2. 预先固定 confidence=.25，沿 native NMS边界用 **score>.25**，同时报告exact .25边界框数；不扫阈值。对缓存 post-NMS 数组保序筛选，然后分别在 IoU=.50 与 .75 **重新调用 pinned native匹配**。不沿用旧低阈值 TIDE TP标签后筛分，因为匹配方法、输入预测集合和分母不同。
3. 沿用本轮 witness 已测试的实际 `DetectionValidator._process_batch` / `BaseValidator.match_predictions` 身份捕获，每个IoU单独iouv=[阈值]，保存真正native返回的GT↔prediction见证并逐TP核对。CPU可从已接受完整源快照加载，避免本地库版本漂移。当前 post-NMS缓存不需再做 NMS；明确“缓存后处理输出在.25处筛分”的口径，若要声称原生.25重新NMS bitwise等价，另需验证原排序/边界/截断语义，不能从缓存冒造全raw证明。
4. 两个IoU各输出完整GT四桶（T正确/N未匹配、N正确/T未匹配、双方正确、双方未匹配），独立每模型TP/FP/FN、对象/图片数及分母、对应预测框见证。匹配阈值造成的IoU50→75变化可分开观察，但不归因具体学习机制，不算AP或可达KD上限。
5. 若要审旧200dev“90/146低置信”代理，还须单独冻结其200图与这套dev的严格身份和GT行映射；640方形与native矩形坐标不可直接比。此为后续可选输入匹配，不把旧桶比例先当真实miss收益，也不把首批32图变化率外推到全dev。

本次没有新GPU/SSH、模型或checkpoint内容读取、训练、hash计算、阈值搜索或全量新结果统计。建议先完成这个两模型×两个IoU的单一CPU诊断；Drone全量IR缓存缺口不应阻塞LLVIP，也不以此为由启动新训练或扩大矩阵。
