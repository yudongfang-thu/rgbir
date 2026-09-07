# 新统一导出脚本的静态复核

> 这是 root 首版 `export_baseline_information.py` 的只读审阅，问题已即时回传；不是修订版、实际 canary 或 GPU 验收的终态结论。

## 确认正确的合同

- `legacy.load_frozen` 验证 checkpoint 的 args.data 与 class names/order，取 EMA/model、float eval、禁用梯度；无优化器。
- 公共 anchor只由N42 P3/P4候选决定；各模型的同anchor原始logits是同一个数组位置。无参考候选时改用RGB-GT中心P3格点且记录标志，后续必须分层，此分支含GT特权。
- `_decode_boxes` 使用 `[B,4,R,A]` 的 l/t/r/b bin顺序，单anchor `reshape(4,-1)`一致；统一用RGB GT评教师同anchor/assigned框，较“各对各GT”能直接回答坐标目标质量。
- 背景环排除两模态全部GT；背景窗口有明确 `is_background` 与annotation proxy标识。
- 输出对象序与features/logits append序一致；JSONL保留object_id、split/source、GT、pair_info、anchor与每模型分配，具备共样本分析基础。

## 首版需处理的问题

1. **填充区未排除。** `roi_vector`背景环与`background_boxes`仅检查GT，未用letterbox valid mask；Drone512×640缩放时上下填充64px，y80中心的64px背景窗口跨入上边填充。应由pair_info两侧原shape/matrix计算共同有效内容区，背景窗完整落在其中，ROI/环排除填充。已作为确定问题回传。
2. **raw feats身份尚未验证。** `legacy.raw_prediction`只确认字典keys，不证明`feats`为Detect输入。建议canary Detect forward_pre_hook抓真实输入与raw feats逐层shape/value精确核对，并保存通道数；这是证据缺口，尚无拿错特征的实测。
3. **多模型layout需逐一检查。** 首版只调用N42 `_layout`，其余模型只核stride。补全每模型class/anchor数、regmax16、三层网格和中心顺序一致断言，避免同下标不同语义；现有三个YOLO11n身份使风险有限，但输出合同仍应实证。
4. **GT ROI只能条件读出。** 2×2 GT窗口汇聚可分析已知对象区域语义，不足以声称独立定位可读出。定位特征需共同anchor固定窗口；若不额外导出，就仅用现有raw DFL评定位并明确限制。
5. **invalid零值不能当真实证据。** 无FG或无BG时relative logits写0、bgmean置0且valid=false。后续读出必须用有效掩码，保存全基础分母，避免将缺失编码的0当真实零差异。
6. **身份记录强于目录名、弱于端点完成验证。** 已记录args/path/size/mtime；尚未断言seed/结构及checkpoint完成轮数，应结合既有端点评价回执核对。无需新增权重哈希。

曾提示回执可能隐式hash，随后阅读本地 `tools/write_jstars_run_receipt.py` 已核销：当前 `build_receipt/_copy_group`只复制快照，无hashlib/digest调用。该猜测不构成问题。

全部意见已发给主代理，本子任务未改root脚本或启动GPU。

## 修订版回读

随后回读同路径修订版，已确认加入共同有效内容域、全部模型 `_layout`/DFL64/中心一致断言、seed与args E200断言、Detect pre-hook捕获及首图raw feats精确相等检查、共同anchor固定邻域特征。首版主要问题已处理，静态检查未见阻止canary的确定错误；实际特征身份和运行验收以root的canary输出为准。

后续分析保留三项界限：定位特征主表限定 `anchor_has_reference_candidate=true`，GT中心fallback另列；固定64×64背景窗与可变GT ROI存在尺度/采样差异，前景读出加入窗口元数据控制或用固定anchor patch；invalid ROI写0的行不能当有效零证据。新增summary已区分 `anchor_patch_region_gt_privileged=false` 与 `anchor_association_gt_privileged=true`，前者不能抵消后者。
