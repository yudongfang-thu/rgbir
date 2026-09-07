# 旧 N/C0 评估桥接候选：独立代码范围审阅

**候选生成工具代码范围接受；六端点评估桥接本身仍为DRAFT，未签发ACCEPTED。** 2026-09-07，独立审阅者 `/root/review_matrix_spec`，未编写或修改 `prepare_evaluation_bridge.py`。本次无SSH、GPU、模型载入或原结果改写。

独立逐段读核候选工具、已接受分析器的实际bridge读取契约、成功N42 probe合同与实际源码副本。复跑 `python -m unittest test_prepare_evaluation_bridge -v`，6/6通过（0.304s）。首次从工作区根误选测试cwd导致ModuleNotFoundError，改在实施LOG执行后通过；这不是工具运行失败或GPU attempt。

## 已核对

- 新probe的N42 native/evidence完整dev1469、五项聚合指标与逐类AP精确相等；历史N42与新native五项指标也精确相等。候选明确只外推评估语义，不声称旧六均重新执行过逐类或对象观察器。
- 旧六端点的训练/评估receipt先经原验证器检查；实际bound config、RGB YAML、roster、环境版本和原生model.val调用参数与新probe推导合同一致。权重只读，random三个端点只盘点，不重标为新方法控制。
- canonical源按正式evaluate_independent调用implementation_files的对象顺序，以及receipt writer序号规则构造：自身、DetectionValidator、BaseValidator、YOLO.val、data utils、metrics、NMS，共七个实际副本。没有把profile专用的辅助源码混入正式身份。
- 未来源集合检查逐文件byte相同，不使用hash，不修改旧receipt或配置。六候选均保持DRAFT且当前analyzer拒绝升级，候选失败清单为空不自动签接受。

## 桥接仍需保留的边界

旧六未保存实际库源码、observed seen或loader迭代顺序，原sidecar args的single_cls也不是当时receipt绑定证据。现有材料支持“同版本、同入口/参数、同总体名单，且N42重复结果一致”的聚合指标语义桥接候选；不能将其表述成六端点均已取得运行级逐字节精确等价。

未来正式evaluation还需用工具核验实际七源副本，并由accepted analyzer核实际effective kwargs和完整dev合同。旧六缺逐类AP和objects的事实不因桥接消失，因此不能用该桥接单独宣布harm检查通过或形成对象损伤/修复结论。需要这些分析时，另保留独立只读评估产物及新观察器回执，不回填旧原始证据。

本审阅不阻塞C1正式训练准入；未接受六端点作为正式分析器bridge、没有修改已接受分析器源码，也不由候选工具自身决定论文claim等级。
