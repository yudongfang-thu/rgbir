# 实测证据绑定工具独立复核

当前判断：基础绑定方向正确，19 个 CPU 磁盘/模块列表夹具测试独立复跑通过；定位实测输入文件的内容绑定仍需修复，暂不签署该新工具的最终接受。本文件不改变此前独立接受的分析器源码回执，也不表示任何 GPU canary 或正式准入已通过。

复核范围：`rgbir_independent_kd_v2/evidence_bindings.py`、`test_admission.py`，并只读检查 `admission.py` 中该 API 的集成。复核者：`/root/review_l1_spec`；日期：2026-09-07。

已确认的行为：

- 捕获实际 `sys.modules` 中模块目录内的 Python 源文件，并复制源文件和数据文件的实际字节；不使用哈希。
- 路径/源码/配置变化会拒绝复用；原始绑定目录以 `exist_ok=False` 创建，不覆盖旧实测证据。
- canary 的 arm/source 严格匹配；校准允许 C1→C1_y 的预定载荷差异、L1→L_GT 的预定目标差异，仍保持其他已记录配置一致。
- compatibility 仅排除新分支的配置，保留 native/evidence 和增强设置的比较。
- 系数在该辅助工具内只记录，明确留给 admission 验证；当前 admission 已检查 canary 中的系数及绑定配置系数等于请求配置，校准的 lambda 等于请求系数。
- CPU 测试只是合成证据绑定夹具；没有被记为真实训练、GPU 运行或技术准入接受。

待修问题：

1. **定位实测文件绑定缺失。** 当前捕获文件只有 `student_data_yaml`、`privileged_data_yaml`、`paired_train_mapping`。定位的 `geometry_contract`、`d2_receipt` 只以路径出现在配置中。若同一路径文件在校准后、readiness 副本生成前改变，旧校准绑定仍可能通过，而 readiness 副本只证明就绪时的内容一致，不能证明实测校准使用同一门控。calibration/canary 应同时捕获并验证实际相关小 JSON 的字节，视真实配置补充 source geometry 输入。
2. **需核对 YAML 间接引用的 split 名单。** YAML 字节不变不等于它指向的 train/val 名单内容不变。应检查实际路径解析和 paired mapping 的关系，必要时将使用的 split 名单列入小体积实测输入绑定。此项尚未宣称已通过或已复现。

已将问题发给工具作者 `/root/review_matrix_spec` 和根执行者；作者修复后需要独立复跑与内容检查，不能仅凭本轮 19 项测试签接受。

## 第二轮复核：接受绑定辅助工具

作者已为 calibration/canary 捕获并逐字节核对非空 geometry_contract、d2_receipt 和实际路径型 source_geometry_scope；缺少预期辅助记录也会拒绝。两份数据 YAML 所引用的 train/val `.txt` 名单同样存真实字节副本，不访问 test。独立阅读修复后源码并复跑新增测试：**22/22 CPU 通过**。

上述两项待修点已解决，当前接受范围仅为实测配置/源码/小输入绑定辅助工具。目录型 split 的图像和标签内容治理不由本工具伪装完成；本接受不表示 L 几何、校准、GPU canary 或 formal readiness 已通过。

接受源码副本与辅助工具回执：`evidence_binding_review_v2/`。此前 19 测的发现保留在上文，便于追溯。没有修改已接受分析器的三个绑定源码。
