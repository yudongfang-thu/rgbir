# Drone seed42 E20 三臂短程草案

**DRAFT：三臂配置已准备，未启动、未排队、未放行。单seed独立20轮日程，只给方向反馈；不构成E200或多seed结论。**

`generate_configs.py` 从已实际运行的formal C1 seed42配置生成 `configs/` 中N/C0/C1三个可审阅YAML，并保留源配置逐字节副本和path/size/mtime。共同原始yolo11n初始化、RGB/IR/reference身份、17990训练图/1469完整dev图、B32/nbs64/workers4/640/AMP/SGD及全部增强均继承原配置。

变化仅为独立E20日程、方法臂及原系数N=0/C0=.1/C1=.09227393550836771、SHORT_SCREEN元数据与DRAFT状态。原lr0/lrf=.01、warmup3轮、cos_lr=false、close_mosaic=0保留；LR以20轮为衰减分母，**不等同E200前20轮**。旧校准路径只说明系数来源，不继承旧E200/旧canary的放行。

C1候选暂为UNRESOLVED，后续可显式绑定已独立接受的pool、完整selector或criterion_factory。当前YAML不能运行；需新的24update技术审阅、实测显存/RSS和三臂总耗时预算，以及单独SHORT_SCREEN准入记录。所有GPU仍由root的原globallease管理，本目录无排队器或自动放行逻辑。

独立 `train_short_screen.py` / `evaluate_short_screen.py` / `screen_common.py` 已就位，`test_short_screen_cpu.py` 的12项CPU合同/负例全部通过（`cpu_checks_attempt2.json`），没有导入torch/runtime。首次11项通过的回执与测试源码也保留；第二版添加criterion_factory真值。它们不修改或调用正式E200完成检查，不写可冒充formal的completion_receipt.json；训练终点只写SHORT_SCREEN_TRAINING_COMPLETED，评价只写SHORT_SCREEN_EVALUATION_COMPLETED。原native evaluator与完整dev roster/GT人口用于短程评价，评价口径同一而训练recipe不同。

## 文件与实际入口

|文件|职责|
|---|---|
|generate_configs.py / configs/|生成并保留三臂DRAFT YAML、原配置副本及生成回执|
|screen_common.py|短程配置合同、独立准入依赖、数据盘输出、stat与源码字节副本|
|train_short_screen.py|原通用trainer、独立E20 LR、固定last/EMA、单独短程完成/失败回执|
|evaluate_short_screen.py|完整dev原native指标，固定原kwargs，GT/图数/逐类完整验证，单独短程评价回执|
|test_short_screen_cpu.py / cpu_checks_attempt2.json|12项实际CPU真值/拒绝检查，无运行环境导入|
|admission_DRAFT_example.json|仅展示未来独立审阅字段，status=DRAFT，不能执行|

只在未来另行审阅并确立资源窗口后，入口形状为：

```text
PINNED_PYTHON train_short_screen.py --reference-dir EXISTING_RELEASE_GPU5 --config REVIEWED_E20_CONFIG --admission ACTUAL_SHORT_SCREEN_ADMISSION --output NEW_DATA_DISK_RUN
PINNED_PYTHON evaluate_short_screen.py --reference-dir EXISTING_RELEASE_GPU5 --run COMPLETED_SHORT_RUN --admission ACTUAL_SHORT_SCREEN_ADMISSION --native-profile-binding EXISTING_EVALUATOR_PROFILE_BINDING --output NEW_DATA_DISK_EVAL_ATTEMPT
```

所有三臂固定训练20轮，并以自己的short_training_receipt.json验证完成；没有按AP选best或改轮数。C1可选original（原完整selector）、显式pool/完整selector源码导出函数，或criterion_factory；唯有独立准入中的候选源码与批准副本逐字节相等才载入。N/C0强制原入口；不借用新selector路径改变它们的损失。

**当前selected_only_v1必须选criterion_factory，不能当完整selector直接接原损失。** 该路径按实际API调用 `make_api(selection_adapter, classification_logit)`，再调用 `make_criterion_type(criterion_module, api)`；工厂内部取原IndependentCriterion，并共同绑定薄selector及其匹配loss/statistics。返回类必须为原criterion的子类。若模块同时提供make_api/make_criterion_type却被配置为pool/selector，入口直接拒绝。12项CPU测试覆盖此路由及错误路由拒绝；它只验证接口，不代表真实候选已经24update准入。

训练人口键也已经从真实legacy源码核实：`task_conditional_reference/legacy_oev1/train_object_evidence.py` 第198–204行的runtime_ready.json明确写train_images/val_images，分别来自实际train/test_loader.dataset长度。

训练调用原build_trainer但不调用原run，也不调用emit/capture证据流程。所有源码包括已有release、当前短程入口与候选同目录Python依赖留副本；实际训练框架trainer源码也记录。输入模型不改写，前后path/size/mtime核对。全量权重仅留新服务器run，不拷到本地。

## 评价合同

复用已接受的原profile binding，其真实根为：

`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/evaluator_profile_attempt2/evaluation_profile_binding/binding.json`

该binding的evaluator源码确为release_gpu5，configuration identity不含训练epochs，所以可验证相同检测评价口径；它不证明短程训练已完成。短程入口先验自己的E20回执，再调原纯roster/metric/capture helper；原E200 run/load_run_configuration/publish/baseline_identity一律不调用、不改动。

实际kwargs要求640/B32/workers4、conf=.001、IoU=.7、max_det300、class-aware NMS、rect=true、augment=false、quantize=null/half=false。评估前loader labels须为1469图/22462 GT，完成后对象缓存逐图人口与GT数闭合，AP类别必须覆盖0–4。primary mAP/AP50/AP75/precision/recall及逐类AP来自原pinned Ultralytics metric_record，单位fraction0–1；不是TIDE定义。三臂比较须只用这组三份完整短程终点；旧E200不能拼作本组对照。

上述GT检查是人口/数量闭合，**不是逐图GT框与类别对历史版本的字节一致证明**；原profile binding固定YAML、roster及评价源码，并未绑定全部label内容。本草案不把这些数量检查称作GT exact。逐类三种AP还要求finite且在[0,1]，YOLO初始化失败也落独立短程failure。loc_stress已只读核对上述native接口与已验收dev roster/kwargs，没有发现DRAFT接口阻断；未做真实GPU执行。

## 独立放行依赖仍缺失

本包**没有生成任何有效admission**。入口在导入torch/runtime前先拒绝缺失/DRAFT准入。将来若决定执行，root需独立审阅并创建：

1. SHORT_SCREEN_ADMITTED决策，绑定该臂config、train/eval入口批准副本、明确candidate和源码批准副本；仅单seedE20，不授予E200。
2. PASS_FOR_SHORT_SCREEN技术复核：对应臂、至少24新成功更新、实际回执路径及候选实现审阅；已完成诊断不是自动PASS。
3. PASS_FOR_SHORT_SCREEN_RESOURCE_AND_BUDGET复核：实测训练NVML/RSS、所有三臂含余量总预算≤43200秒，并保持globallease。字段finite且正值才接受；未知排队时间仍不能承诺12小时。

这些文件由独立审阅形成，不能只把DRAFT字符串改成PASS。对已有C0专项暂停、新候选是否真正等价、单卡新任务可否满足显存/内存与负载条件，仍需root明确处理。当前草案没有解除任何已有约束。

不计算新增hash，不改正式release或旧产物。完整配置/源码供审阅；本说明不代表执行许可。
