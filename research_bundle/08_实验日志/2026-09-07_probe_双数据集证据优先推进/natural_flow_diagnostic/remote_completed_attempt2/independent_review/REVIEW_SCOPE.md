# 自然流选择诊断独立审阅范围

审阅者：协作agent `/root/ap_error`，执行作者为另一协作agent `/root/baseline_feature_analysis`。本审阅仅CPU，不访问GPU/SSH，不新增hash；用户禁hash约束优先于experiment-audit技能默认的hash归档要求，改存路径/size/mtime、源码副本、直接字节相等核验。

检查固定20260907自然流：无重抽样、替换、短batch；每批原覆盖记录中的source、增强信息、RGB与IR标签、顺序完全相等。只读取训练流，不读取test/AP结果决定阈值。

模型必须为固定T/R且eval、FP32、无梯度，各批每模型一次前向。分类student槽仅复用R以调用现行选择接口，不存在新增学生参数/梯度；全batch调用选择保留全局rho/ceil/质量排序。

定位只调用原teacher同anchor选择，记录所有gate、base/eligible/selected及图/组计数。None/false geometry仅未验证几何上界。若为归组逐图重调原选择，必须逐真实batch逐字段核整batch完全相等，并恢复全局图/GT索引；normalizer须用max(1,全batch base)，不能累加逐图normalizer或平均剂量。

独立CPU真值至少覆盖：多图C全局配额（逐图做法会多选的反例）、L不同GT数/含空图/索引非连续和gate拒绝、逐图与整batch重建完全相等、旧自然流内容被微改即拒绝、2batch canary/64full入口约束。数值统计仅说明原选择集合和目标可用性；不证明梯度、校准、可学性、收益或几何准入。

审阅将区分“源码/CPU真值通过”与“94真实canary/64batch运行完成”。真实流尚未执行或不可访问时不得以CPU通过替代。
