# LLVIP旧baseline全dev统一重评：独立代码审阅

**状态：ACCEPTED_FOR_CANARY，0个已知阻塞代码问题。** 接受本地当前 `PROTOCOL.md`、`export_full_dev.py`、`run_campaign.py`、`prepare_remote.py` 与 `spec.json` 的只读评价实现。此结论不是实际canary、完整2406图评价或L1训练准入回执。

## 核查依据

只读审阅以上文件及已返回的 `remote_metadata.json`；对照本地实际release_gpu5来源模块 `evaluate_independent.py`、`evaluator_profile.py`、旧统一 `resource_dispatch.py`，并阅读已执行8.4.115 `engine_validator.py` / 评价审阅记录。没有SSH、GPU、权重写入或新哈希计算。

## 通过的实际接口与边界

1. **数据与模型身份。** `dev_roster`只解析train/dev YAML的val，拒绝test键/路径、空集合和重复；必须完整2406。当前spec分别绑定旧LLVIP visible42/infrared42 last和对应visible/infrared数据YAML；运行再次验证旧args.data的resolve值、seed42/E200及模型完整class names/order等于数据定义。只有`person`类。旧workers8身份完整保存，不称新workers4 N。
2. **推理精度。** 已修复初版`half=False`：实际8.4.115接口使用`quantize=None`得到FP32；pinned BaseValidator向AutoBackend传`fp16=(quantize==16)`并将实际状态回写quantize。新代码用已验证`capture_contract`读取实际quantize/half，并拒绝非FP32。其余固定640/B32/workers4/rect/conf=.001/NMS IoU=.7/max_det300等与本次冻结协议一致。
3. **原生指标与对象捕获。** capture继承真实DetectionValidator，先`super().update_metrics`再序列化GT、post-NMS预测，不改native metric tensors。canary的native/capture分别重建模型和固定RNG；五汇总指标和逐类AP exact、实际kwargs和loader顺序相同才通过。`verify_population`检查含空预测图的对象记录数量与唯一名单一致。full使用相同capture路径保留原生AP与2406行预测，不能把阈值命中率当AP。
4. **同shape短测。** 先读取全dev原始尺寸并要求本模态仅一种尺寸，再以固定首64图/B32双批执行；这检验同推理网格的实际路径。NMS/场景内容仍可影响峰值，因此形状相同不代替运行时资源监测。full全量2406的最后小batch不增加最大图像batch。
5. **资源与排程。** 新runner复用既有全局lease而非另建池；每个模型先独立canary，只有native/capture exact且NVML/进程树RSS为正才以实测峰值＋余量预约full。旧dispatcher持续查整卡≥2GiB余量、项目每卡最多2任务、常规3卡/空卡条件下4卡及总RSS≤300GB。较AGENTS每卡上限3更保守，不造成越界授权。canary的4096MiB/12288MiB是预约而非伪造实测。
6. **输出和旧权重保护。** exporter仅执行`YOLO(...).val`，不调用train/save/strip_optimizer，不把评价输出指回旧run；新目录必须不存在且位于`/mnt/dataset/yudongfang/`。旧权重stat前后检查、args/data/source复制到新artifact。标准loader可能读取/刷新其派生标签cache，这不等于改GT或checkpoint；本审阅不把stat检查冒充字节哈希。source训练E200由旧args和既有baseline证据支持，`checkpoint_epoch=-1`仅表示可能已strip，单独不证明完成200轮。
7. **部署。** `prepare_remote --launch`在读取实际独立审阅文件后，exclusive mkdir新远端artifact目录、scp指定源码/协议/spec/审阅、screen后台启动；新screen只运行diagnostic campaign。无旧路径递归删除、移动或权重改写。重新执行同attempt会因目录已存在失败，符合保留失败attempt原则。

## 已修问题与残余说明

- 初版half旧参数已换为quantize，并增加实际FP32断言；当前已通过API来源复核。
- 初版只看类别数/seed/E200，现已补data路径与全部class names/order断言；当前spec与返回metadata的真实路径一致。
- 特意复用`dev_roster/metric_record/assert_equal_metrics`等通用helper，**没有调用Drone1469专用的profile配置绑定**；因此不会把LLVIP伪装成Drone评价资源合同。
- 完整native/capture真实canary、NVML/RSS余量和旧checkpoint stat不变仍须由root执行并验收。full只在各自canary通过后允许沿已审路径继续；新L1几何/校准/N基线训练准入保持未满足。

接受者：独立子代理 `baseline_feature_analysis`。本报告只做代码与API来源审阅。
