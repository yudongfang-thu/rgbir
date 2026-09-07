# LLVIP 现有baseline完整dev诊断 v1

执行前冻结：现有visible42、infrared42固定last/EMA分别在其模态全部dev2406评价。仅train/dev登记的val路径，禁止test。旧模型workers8身份保留，本次统一推理imgsz640、batch32、workers4、FP32、rect=true、conf=.001、NMS IoU=.7、max_det300，不使用增强、不挑best，不生成新协议N训练回执。

复用冻结release_gpu5的原生DetectionValidator和已验证的只读post-metric对象捕获。先分别对两模型的固定首64个dev样本执行native与capture，要求五汇总及逐类指标完全一致、真实loader名单一致、对象完整、显存及进程树RSS实测。先验读取所有dev图尺寸，要求每模态单一原始尺寸，确保短测覆盖全量最大形状；否则不自动放行。canary资源预约4096MiB/12288MiB，不等于实测结果；完整任务依据两模型各自测得峰值+余量预约。所有任务使用已有全局lease、动态GPU、screen，遵守最多双开等工程约束。

full依次visible42→infrared42，每份保存原生指标、完整2406行GT和低阈值post-NMS预测（含空预测图）、actual kwargs/坐标/名单、权重与原args身份、数据及源码副本、资源/完成/失败回执。记录完成checkpoint的epoch/EMA身份；仅能称旧baseline统一重评，不与新N混用。随后同CPU TIDE入口做AP50/AP75错误诊断；与原生mAP口径分别报告。

两模态dev按stem一一对照并核对各自标签内容；共享/相同标签不提供物理几何配准证明。读取GT只用于评价，不能用这些dev指标调门、扫lambda或挑正式训练端点。正式L1几何/自然流/校准准入均未被本实验满足。

所有输出写新的数据盘artifacts目录，旧证据和训练不改。短测失败保留attempt，技术修复使用新attempt。
