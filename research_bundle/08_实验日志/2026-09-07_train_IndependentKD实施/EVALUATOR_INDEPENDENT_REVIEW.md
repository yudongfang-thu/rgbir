# 独立评估入口源码审阅

**初审三项问题及恢复/证据问题已按根agent授权修复，10项CPU测试通过。随后由未编写该文件的C1实现agent独立复读及复跑10项测试，签发 CODE_SCOPE_ACCEPTED，详见 EVALUATOR_CODE_SCOPE_ACCEPTANCE.md。无GPU、无新AP，真实端点数值等价仍待执行。**

审阅者未编写`evaluate_independent.py`。审阅源码为2026-09-07本地版本，引用行号以初审版本为准。依据用户固定E200 last/EMA dev协议，以及本地已经保留的pinned 8.4.115源码，未做在线版本替换。

## BLOCKING

1. **实际精度未被正确记录**（evaluate_independent.py:60–65）。pinned `pinned_engine_validator.py:191/194`使用`args.quantize == 16`传入AutoBackend，并将实际`model.fp16`回写quantize；当前只读不存在的`args.half`，合同会出现None，不能表示实际FP32/FP16。应记录实际quantize，合同兼容字段half显式由实际quantize推导；不向model.val新增half/quantize参数以改变旧评估行为。
2. **预期名单冒充实际名单**（43–46、61–65）。`diagnose_opportunities.split_images`会去重，合同只绑定预先枚举结果，没有核`validator.dataloader.dataset.im_files`及最终`validator.seen`。应在on_val_start核实际dataloader名单无重复、集合及数量与预期完整名单一致，并记录实际顺序；rect排序可合法不同于字典序。完成时核seen和对象记录一图一条，缺图/重复图不写成功端点。标准化比较合同可使用已核一致的规范排序名单，但不能省略实际执行核对。
3. **评价输入未与实际训练运行配置先行核对**（33–35、43、81–90）。当前接受任意args.config，只查completion状态和last_epoch；同dataset名称下的数据YAML或方法配置可偏离真正训练配置后继续执行。应读取run内冻结的有效protocol_config.yaml并与实际train receipt绑定配置一致，再核completion的dataset/method_id/arm/source/seed/checkpoint。args.config若保留为入口，允许差异必须明确，不可用它覆盖实际训练身份。dataset应显式枚举Drone/LLVIP，避免所有未知值落到LLVIP数量。

## 技术恢复和诊断证据

- 84行先写根evaluation_val.json，再85行创建eval receipt；若后者失败，37行拒绝所有retry，`--attempt`不能恢复。建议完整attempt目录先持久化结果/回执后再发布根入口；失败attempt保留，不通过删除/覆盖原端点补救。
- `objects.jsonl.gz`只以路径出现于结果，未纳入receipt原始产物副本。使用它计算harm/修复损傷率前需额外绑定原始对象证据与固定诊断脚本；单有文件路径不等于证据已被接受。
- 当前`implementation_files(DetectionValidator)`只返回检测val.py，不递归绑定继承的engine validator、NMS/metrics实现。至少应把实际BaseValidator和AP/评价辅助实现纳入可复核源码范围；版本字符串不能代替源码副本。

## 原生AP不变性的审阅依据

`EvidenceValidator.update_metrics`首先原样调用`super().update_metrics(preds,batch)`，之后仅读取pred和重新准备的GT以写JSON。已留存的pinned detection val.py中`_prepare_batch`通过布尔索引获得bbox，再换算到当前canvas；本扩展未修改预测/GT或metrics状态。pred和GT在同一canvas，未混用原图坐标。主mAP、AP50、AP75直接取原生metrics对象；model.val主要kwargs与旧evaluate_object_evidence一致。

因此没有发现扩展显式改变AP公式、NMS、匹配阈值或统计器；这只是源码审阅。尚未执行相同checkpoint、相同batch上的原生/扩展输出逐项相等测试，不能把语义判断写成真实数值等价回执。旧N/C0向新合同的兼容桥接仍须独立复核。

## 依据路径

- 新入口：`03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/evaluate_independent.py`
- 旧入口：同目录`task_conditional_reference/legacy_oev1/evaluate_object_evidence.py`
- 实际精度：`08_实验日志/2026-09-07_audit_native与蒸馏载体/native_raw_evidence/pinned_engine_validator.py`
- 检测框/GT及指标流程：`08_实验日志/2026-09-07_audit_RGBIR实施起点/comparator_snapshot/cclkd_eval/s42/eval_evidence/source_snapshot/trainer/02_val.py`
- 源码解析函数：`03_现行工程/SpaceNet6_OTD_official_reproduction/tools/write_jstars_run_receipt.py::implementation_files`

## 已实施修复与CPU验收

根agent授权本审阅者修改evaluator后的版本：

- 从实际quantize记录half，保留model.val原参数，不新增half/conf/IoU覆盖。
- 规范roster仍为固定字典序，与旧N/C0及分析器合同一致；另保存actual_loader_roster。真实dataloader名单、seen和对象记录逐一核完整数量、无重复及集合。
- 必须使用run内有效protocol_config，并核对实际train receipt中的配置/完成指标副本、E200、dataset/method/arm/source/seed/系数和last checkpoint；未知数据集不默认作LLVIP。
- 数据、预测tensor及metrics原生更新之后只读捕获。主指标仍直接取原生metrics。实际BaseValidator、检测validator、YOLO.val、check_det_dataset、AP及NMS源码均进入receipt。
- 原始成功attempt目录内先保存evaluation_val.json和完整eval_evidence；objects.gz作为第二个原始metric artifact逐字节复制。分析器根入口最后发布，不挪动原attempt。
- 发布使用派生临时副本和不覆盖的原子文件链接；中途失败保留原attempt，可同`--attempt`只补发布，不重算AP。失败推理/不完整receipt保留原目录并使用新attempt。既有异内容拒绝覆盖。

新增`test_evaluate_independent.py`，执行`python -m unittest -v test_evaluate_independent`：`Ran 10 tests in 0.353s / OK`。不导入Torch，不连接服务器。测试覆盖真实临时文件配置/receipt相互绑定、错端点/数据集、rect排序、FP32/FP16字段、缺/重名单、seen对象缺损、原生调用先行且无输入/指标改写、发布幂等、模拟中途IO失败无重算恢复、无receipt不发布和对象/既有副本冲突。

其中原生调用测试使用明确synthetic父类和tensor夹具，只验证包装器调用顺序与无改写；它不是Ultralytics AP数字等价证据。现阶段可部署代码后执行真实fixed-checkpoint原生/扩展比较；旧端点合同桥接仍后续独立处理。
