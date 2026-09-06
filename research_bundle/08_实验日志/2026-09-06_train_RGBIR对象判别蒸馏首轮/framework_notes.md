# RGBIR 对象判别蒸馏：现有框架接入核查（2026-09-06）

> 只读核查完成。建议复用 pinned Ultralytics 原生目标分配和检测损失，在新目录实现独立 criterion/trainer；首轮 N/P 同框架重跑，不能直接宣称旧 native 与新 weight0 字节等价。特别注意：native loss 是三元素向量，向它直接加标量 KD 会在 trainer 求和时把 KD 放大三倍。

## 核查范围

读取本地 README、实验索引、最近 RGBIR probe 和最小验证设计；通过 ssh 94 读取代码、配置、checkpoint 参数及 Python inspect。没有 GPU 推理、训练、服务修改或原始文件写入。唯一产物是本文件。

服务器工程根目录 S：
`/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction`

RGBIR 项目根目录 B：
`/mnt/dataset/yudongfang/projects/RGBT_campaign`

运行 Python：
`S/environments/sn6-int8-kd/bin/python`

已实测依赖：Ultralytics 8.4.115；其 loss 源码：
`S/environments/sn6-int8-kd/lib/python3.10/site-packages/ultralytics/utils/loss.py`。
CPU importlib 核查 albumentations **未安装**（find_spec 返回 None）。

## 1. 最直接的复用入口

- `S/tools/train_rgbt_cmdistill.py`：
  - `build_cmdistill_trainer` 子类化 `DetectionTrainer`。
  - `build_dataset` 仅为 train 包装 `RGBTSharedGeometryDataset`，val 不包装。
  - `preprocess_batch` 先完整调用 native，再把 strong_img 转 device/float/255。
  - `set_model_attributes` 绑定 criterion；本次建议改为 `super()._setup_train()` 后绑定，原因见 EMA 项。
  - `validate/final_eval` 关闭训练期评估，独立 evaluator 在 last.pt 上评估。
  - `optimizer_step` 计实际调用次数；canary 设置 self.stop 后 native loop 在本 batch 后退出。
- `S/yolo_osssl/rgbt_hnewa_pairing.py`：
  `RGBTSharedGeometryDataset` 输出 img / weak_img / strong_img / pair_info。
  weak_img 是 native img 的 alias。
- `S/yolo_osssl/paired_detection.py`：
  `transform_paired_record` 先存 Python/NumPy/Torch CPU RNG，执行 native student transform，保存 after；恢复 before 重放 teacher transform；最后恢复 after。因此当前无 Albumentations 的配置下，teacher replay 不额外推进上述 student RNG。
  若新代码调用随机数选对象，使用独立 Generator，不能调用同一全局训练 RNG。

### 配对 loader 的边界（必须改或披露）

当前 `_image_record` 是复制 RGB record 后换 IR 图；它没有载入真实 IR 标签。
`transform_paired_record` 断言两侧 cls/bboxes/batch_idx 完全相等，因此它的 strong targets 是 RGB GT 的拷贝，不能称为“独立 IR GT 下教师正确性”。

DroneVehicle RGB/IR 标签分别标注，若新方法需要可靠对象对应，应在新 adapter 读取真实 IR 标签，并在相同几何参数/同 RNG 下分别变换，保留各侧目标身份和匹配关系。不能用“同源目标数量相同”替代身份匹配；随机缩放裁剪会使两侧不同目标被过滤。
本次 frozen augmentation 中 mosaic/mixup/cutmix/copy-paste/HSV 关闭，几何仅 translate/scale/fliplr，因此工程复杂度可控。

Pinned 8.4.115 的 `RandomPerspective` 已拆成 get_params/apply_image/apply_instances；get_params 返回 M/scale/orig_shape/size，apply_instances 对 cls 和 Instances 使用同一个有效框 mask。`RandomFlip` 同样拆为参数/图像/实例操作。
可选方案是捕获 native 实際几何参数后复用到独立 IR Instances；若用全管线 RNG replay，应保留原生 student transform 的输出，并恢复 after RNG，不二次生成 student 数据。

`PairedDetectionDataset._cache` 是无容量上限的 IR 图片缓存，而且是每个 DataLoader worker 独立一份。全训练期间会逐渐膨胀；建议新 adapter 用有限 LRU（例如 128–256 图）或不缓存。不要改现有 canonical loader。

## 2. Native assigner 与 raw class logits

本机 pinned API 可直接调用：

```python
student = native.parse_output(prediction)
assigned, components, native_items = native.get_assigned_targets_and_loss(student, batch)
fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assigned
native_total = components.sum() * batch_size
total = native_total + kd_weight * batch_size * kd_mean
return total, native_items
```

该入口执行**一次**原生 TaskAlignedAssigner 和所有原生损失，不需要复制源码或为拿 assignment 再算一次 native loss。

- student['scores']：未 sigmoid 类别 logits，形状 [B,nc,A]。
- student['boxes']：raw DFL logits，[B,4*reg_max,A]，当前 reg_max=16。
- student['feats']：各级空间 feature maps；make_anchors 顺序为层优先，然后行优先。
- teacher eval forward 返回 (decoded, raw_dict)，训练 forward 返回 raw_dict。
- `native.parse_output` 原生支持 dict 或 tuple；teacher mapping 不可错取 decoded tensor。
- `target_gt_idx[b,a]` 是该图**padded GT 内部序号**，不是全 batch flatten ID。
- native.preprocess 使用 batch_idx，把输入 GT 依照 batch 内每图顺序堆为 [B,max_objects,5]。新辅助数组必须保持 collate 后相同 per-image GT 顺序。
- `fg_mask` 是 native task-aligned 学生正分配，KD 可以读取同一对象的正位置集合；不能为了 KD 改它而仍声称 native loss 未变。
- raw 类分数没有独立 objectness 分支，不能新增“原生 objectness”说法。

### 标量广播陷阱

`v8DetectionLoss.loss` 返回 `loss[3] * B` 和 dict；
`BaseTrainer._do_train` 随后执行 `self.loss = loss.sum()`。
旧 CMDistill criterion 的 `native_total + float(B)*kd_total` 会向三项全部广播，得到有效 3×KD。
本次应明确返回 scalar native_total + 单次 KD，或只在三项之一加一次。
这一核查是实现审计发现，不单独证明历史方法失败原因。

## 3. 模型/EMA/优化器接入

教师/reference 应是冻结 eval 模型，所有参数 requires_grad=False，teacher forward 包在 no_grad；对象选择/target detach。
把教师放在 ordinary Python criterion 内，而不注册成 student 子模块，可避免 optimizer 包含教师或 checkpoint 改变部署结构。

但 `ModelEMA.__init__` 会 deepcopy 整个模型；如果 criterion 提前挂上，ordinary criterion 内教师也被 deepcopy 到 EMA，造成无用途额外显存。native save_model 也是先 deepcopy(EMA) 再 criterion=None。

推荐在 `super()._setup_train()` 完成 ModelEMA/optimizer 建立后再给 live student 装 criterion。EMA 保持没有 KD 状态，部署/保存结构干净。native init_criterion 此时也能看到 class_weights 等最终属性。
若不能延后绑定，必须明确处理 criterion 的 deepcopy 并实测没有教师/reference 副本进入 EMA。

已有 `load_teacher` torch.load(weights_only=False) 选择 model 或 ema，之后 float；本次应校验 names、nc、训练数据路径、权重 SHA，不能仅凭文件名。
Teacher checkpoint 可继续使用已核实的 s42 native IR 模型，但首轮 student 从 frozen yolo11n.pt 初始化，不是从训练完成 RGB baseline 微调。

当前执行遵循工程内 AGENTS.md 的“禁止写哈希和 SHA256”：本轮只保存源码/配置副本、路径、文件大小和张量直接等价检查；上文身份校验建议中涉及 SHA 的部分不执行。

## 4. Frozen Drone recipe 与可用路径

现行历史 recipe：
`S/configs/research/rgbt_cmdistill_protocol_drone.yaml`

初始化：
`S/artifacts/int8_cross_modal_stage2_v1/weights/yolo11n.pt`

RGB data：
`B/artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/rgb.data.yaml`

IR data：
`B/artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/infrared.data.yaml`

训练配对：
`B/artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/mappings/rgb_to_infrared_train.json`

历史新 native s42：
`B/runs/cgkd_w1/native_rgb_s42_e200/weights/last.pt`

模型 probe 用 IR：
`B/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/infrared_seed42_native_b32a2/weights/last.pt`

recipe：YOLO11n、640、200epochs、batch32、nbs64、workers8、SGD、lr0=.01、lrf=.01、momentum=.937、weight_decay=.0005、warmup_epochs3/warmup_momentum.8/warmup_bias_lr.1、cos_lrFalse、patience0、AMPTrue、deterministicTrue、seeds0/42/123。
增强：translate.1、scale.5、fliplr.5，mosaic/mixup/cutmix/degrees/perspective/flipud/hsv_h/hsv_s/hsv_v/erasing 全0。
历史N进行每轮val且plotsTrue，KD trainer关闭val/plots；不能仅凭 args 同主要超参断言长程 RNG 与新框架字节相同。本轮 N/P 同框架重跑最清楚。

Pinned native nbs/batch 会动态改变 warmup accumulate；记录 actual optimizer step 和 EMA.updates，不只报告 epoch×batch数量。

### 需要封住的自动修复

`BaseTrainer._do_train` 会在首epoch CUDA OOM 后自动 batch减半重建 optimizer/loader，最多3次；这会悄悄改变预算。
建议覆写 `_build_train_pipeline` 对 batch frozen 值做 assert；首次可建，OOM改batch重建时硬失败，并保存 attempt。
`_handle_nan_recovery` 会从 last checkpoint 重载；新方案应对非finite损失/梯度直接技术失败，不能自动跳过而 still completed。
`save_model` 会对 fp16序列化做 nan_to_num，并可能修复EMA异常；保存前可先核验 student/EMA state 全finite，避免只看checkpoint以为训练没NaN。

## 5. 一张 GPU 的执行与 guard

`S/tools/project_resource_guard.py run` 参数：

```text
--job-id unique_id --kind train
--candidate-gpu PHYSICAL_ID --gpu-count 1
--expected-vram-mib CANARY_MEASURED_PLUS_MARGIN
--expected-rss-mib REALISTIC_RESERVATION
--free-safety-mib 2048
-- PYTHON TRAIN_SCRIPT --device 0 ...
```

guard 会设置 CUDA_VISIBLE_DEVICES 为获批物理卡，所以 train脚本 --device 应为局部0。
本用户新限制是最多一个GPU：只能传一个 candidate-gpu；不要依赖 guard 默认上限3卡。
guard additionally 限制项目显存 <70%、预留RSS admission <240GiB、hard300GiB，并要求同卡第二个 formal train 显式已通过 profile。
建议同一个 screen/queue 串行执行 N_s42 与 P_s42；这样两个实验只占一张卡、每时一任务且无需第二并行任务 profile。
guard 不自动等待：资源不足返回2/QUEUED。启动器若等待，应保留日志且不换到另一张卡。
canary 使用 --kind train --non-formal-train（仍计资源），正式用 formal 默认。
训练自身调用 `require_bound_lease_from_environment()`。
CUDA reserved/allocated 峰值不含整个CUDA context；必须结合 nvidia-smi/guard峰值和RSS，包括 workers。

## 6. 统一评估

`S/tools/eval_rgbt_detector.py` 的 `evaluate_record` 接受 last.pt，配置 evaluation_role只可dev/val，data yaml不能含test键，既定 student_data_yaml 的val作为唯一开发集。
调用 `tools.eval_yolo_detector.evaluate` -> YOLO(last).val，输出AP50/AP75/mAP50_95/precision/recall，数值为0–1。
可在新独立脚本直接复用 evaluate_record，但 `_method_identity` 未识别新方法名时会默认 HNEWA-INSPIRED，不能照搬该CLI receipt身份。新评估wrapper应自行记录本次method identity。
旧 evaluate 的 YOLO.val 若不显式传 project/name，可能使用默认 runs/detect；本次须把 cwd/输出project明确放数据盘，或新wrapper明确设置project/name，避免系统盘默认工作目录。
新 N/P 都使用固定budget last/EMA相同端点，独立评估同一个frozenfullval。首轮seed42仅工程/方向预实验，不得升级三seed增益claim。

## 7. 最小等价检查建议

1. 固定seed，对同一原始RGB样本取得 base native transform 和新 adapter student输出：img/cls/bboxes/batch_idx精确相等；记录RNG after相同。
2. 单模型、固定batch、同native criterion：native原调用 sum 和新 weight0标量调用 loss/grad精确或严格容差一致；避免两份随机初始化模型混淆。
3. P/N 同batch学生输入和目标SHA一致；KD>0时只新增类别梯度，无teacher/ref梯度。
4. N/P model参数名/shape一致，optimizer参数集合与native所有requires_grad学生参数一致。
5. 人工构造错配/低质量/无候选/空GT/多同类对象/裁剪后丢目标案例，验证对象mask、固定候选分母、随机对照K，不只看loss finite。
6. 保存前 model/EMA全finite；确认同一GPU、资源峰值、实际updates、teacher/ref与init SHA全部入 receipt。

## 8. 已实现的独立双标签 adapter 与 CPU 验证

实现文件：

- `03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_object_evidence_v1/paired_rgbir_data.py`
- 同目录 `test_paired_rgbir_data.py`

API：`DualLabelRGBIRDataset(base, teacher_base, strong_by_weak=None, same_modal=False, max_teacher_cache=64)`。
base 是原生学生训练 YOLODataset；teacher_base 是另外创建的 IR YOLODataset，cache=False/rect=False，类顺序与 imgsz 相同。只读取其真实图像与真实标签，教师数据集本身的 transforms 不调用；adapter 将该专用 teacher_base.augment=False，禁止原生图片 buffer，然后使用至多64记录的独立 LRU。学生 base 的 buffer、标签、图像和 transform 均不改。

当前 pinned load_image 在 resize 时始终使用 INTER_LINEAR，augment=False 只改变 native buffer，所以这里没有插值差异。

返回 strong_img、strong_cls、strong_bboxes、strong_batch_idx；经 collate 后强目标的 batch_idx 正确指向每张图。两侧独立裁剪/框过滤导致对象数量可以不同；由 loss 根据存活后的真实类别/坐标匹配，不能按列表下标硬配。

94 CPU 测试目录：`B/artifacts/rgbir_object_evidence_v1_20260906/adapter_cpu_check/`。
命令：在该目录执行 `CUDA_VISIBLE_DEVICES= S/environments/sn6-int8-kd/bin/python test_paired_rgbir_data.py > cpu_test_attempt1.log 2>&1`。
本地日志：`adapter_cpu_test_attempt1.log`。

实际结果：6 tests 全通过，0 failures/0 errors，0.181秒，cuda_used=false。

1. 实际 Ultralytics RandomPerspective/RandomFlip/Format，24个固定随机种子下 img/cls/bboxes/batch_idx 与 native 逐张量精确一致；Python/NumPy/Torch CPU 后续随机数精确一致。
2. IR 真实标签类别/数量与 RGB 不同仍正确保留；相同几何框在两侧的实际增强结果精确相同。
3. 相同仿射裁剪下 IR 额外目标被独立过滤，存活目标类别仍正确，没有列表下标错配。
4. collate 强弱各自目标数量和 batch_idx 正确。
5. same-modal 输出学生原图/GT 的精确副本。
6. bounded LRU 和 teacher读取异常后 student RNG恢复通过。

这组检查验证数据/几何代码，不是模型性能结果；没有使用训练或开发集增益调参。

## 9. 本次 run receipt 建议

`emit_bound_run_receipt` 接受 `method_identity='PROTOCOL-ADAPTED'`；具体新方法名/arm 放 inputs.method_id/inputs.arm，不要冒用 HNEWA 身份。trainers 应包含新trainer与pairedloader；losses包含新loss和native loss源文件；configs包含新协议、RGB/IR data yaml；split_rosters包含显式 train配对表；metric_files包含实际训练完成回执。工具会保存实际源码/配置副本和bound lease资源，无需额外摘要码。

原 Drone prepare_receipt 记有 teacher_labels_used_by_kd=false，是旧方法的事实。本次使用独立 IR 训练标签筛选/匹配，必须显式登记 true；学生检测监督仍仅为RGB native GT。

train路径：`B/data/processed/dronevehicle/yolo/hbb_v1/{rgb,infrared}/images/train`，各17990；val各1469。类序：car / freight car / truck / bus / van。
现有 exposure ledger 对 Drone 记 UNVERIFIED_SEALED、confirmatory=false，本次沿用，不能把已反复使用的val称为封存test。
