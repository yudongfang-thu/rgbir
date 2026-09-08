# 收集时发现的映射输入修正

第一次收集附带请求 `prepared/dronevehicle/mappings/rgb_to_infrared_val.json`，返回 FileNotFoundError。该路径来自旧准备代码的生成命名规则，未实际观察到文件；因此不能说已读取或绑定该映射。错误只影响本地小产物收集，没有中断canary或完整推理，两个GPU任务正常完成。

修复后的收集器保留预测/合同，并在 collection_receipt.unavailable_inputs 明确记录这项缺失，未补造映射。后续CPU配对改为读取原 DualLabelRGBIRDataset 构造器已经支持的 strong_by_weak=None 分支，在两份已接受的完整dev canonical清单上使用其唯一stem规则，并生成明确的派生映射及回执；必须检查各1469、唯一性、完整覆盖和无复用。它是按既有数据配对规则新生成的绑定，不是找到了先前的JSON文件。

本地 `evidence_1401/` 为首次成功收集的终态快照；源GPU结果与旧模型均未修改。本修正不涉及阈值、GT配对算法或方法训练。
