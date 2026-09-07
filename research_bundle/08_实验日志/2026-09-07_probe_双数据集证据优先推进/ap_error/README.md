# 完整检测 AP 错误诊断

> 两数据集完整dev已完成官方TIDE AP50/AP75。在六类main error中，Drone AP50以类别、AP75以定位为主；LLVIP旧visible/IR两模型均以定位为主。special FP/FN另列，不能混称所有oracle中最大；这些诊断不证明可达KD收益。

## 已完成结果

- [Drone完整结果与限制](README_DRONE.md)：六端点、逐seed差、mean±SD、原pinned指标参照、分数区间和背景误检复核。
- [LLVIP完整结果与限制](README_LLVIP.md)：有效attempt2两份旧baseline各2406图/7879 GT；N AP50/AP75=71.427586/24.537259，T=92.229627/44.135239。N的AP75 Loc dAP=56.373148pp，不是可达KD收益上界。
- [首轮原始结果](drone_results_v1/summary.json)：每份完整1469图、22462 GT，含空预测图。独立NumPy AP101逐类交叉检查通过。
- [逐类Cls oracle补充](class_oracle_v1/README.md)：N的AP50宏Cls贡献约94.3%来自freight car/truck/van，不能用car占主导的对象计数替代宏AP瓶颈排序。
- [全部五类的200dev对象桥接](baseline_class_bridge_v2/README.md)：van的55个N错误中28为class_only、26为low_confidence；T相对N0在共同eligible子集仍有独有正确候选，但潜在损伤与单seed/recipe差异均保留。完整3084对象入口可复核，不外推全量AP。
- [已知真值与验证](drone_results_v1/synthetic_validation.json)：类别/定位/同时错/重复/背景/漏检、空图、score ties、无GT类别、单调分数变换。
- [根任务独立COCOeval复核](root_coco_review_v3/summary.json)：六端点×AP50/AP75在相同TIDE recall网格下与官方TIDE最大差2.84e-14pp；默认COCO浮点recall网格有微小积分差，已单独记录，不强称默认实现逐位相等。
- [首次冻结协议](PROTOCOL.md) 与 [来源记录补充](PROTOCOL_ADDENDUM.md)。后者禁止后续新增hash计算，首轮原始attempt保留。

## 复跑

仅CPU，不导入训练工程、不访问SSH：

```powershell
& 'D:/Anaconda/envs/KGJ_proj/python.exe' './run_tide_audit.py' --output './new_attempt'
```

指定额外端点：`--manifest endpoints.json`；manifest为列表，每项至少含name/path/class_names。逐图JSONL或gzip记录字段为image、canvas_shape、original_shape、gt_boxes、gt_classes、pred_boxes、pred_classes、pred_confidence，框为同一实际画布的xyxy。可绑定metric_path/contract_path/receipt_path；未提供正式contract的端点必须另有根任务采集回执完整性核验。

隔离的官方源码位于`official_tide/`，运行依赖仅在`deps/`和现有本地CPU解释器。`prepare_official_tide.py`是首轮下载脚本历史副本，包含当时已执行的来源hash逻辑，**后续不要再执行**；无须重下载。

## 下一步与边界

逐类官方Cls oracle补充输出到`class_oracle_v1/`，LLVIP同口径输出到`llvip_results_v1/`。AP瓶颈与IR可提供知识分开判断；真实可修复/可学性仍须在明确跨模态身份与几何合同下继续检验。本实验不修改或准入任何训练分支。全部新结论仍须根任务独立审阅接受，不升级为论文或四臂跨模态收益结论。
