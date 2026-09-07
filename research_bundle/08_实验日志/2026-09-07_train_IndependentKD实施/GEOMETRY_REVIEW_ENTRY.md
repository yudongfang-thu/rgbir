# 最多24对的真实几何审核入口

> **候选选择工具已实现，6项CPU测试通过；正式24对名单待真实自然64批产物生成。当前没有新增实际点标注、独立复核或verified区域。**

## 可复用材料与显示入口

已有几何审计目录为 `08_实验日志/2026-09-07_probe_TaskConditional几何审计/`。

- `geometry_audit_tools.py::make_panel(row, output_dir, width=640)`：读取 row.rgb_path/ir_path，生成无GT/预测框的并排图；左RGB/右IR，上方36px标题。每幅模态单独缩放至640宽，返回真实原图尺寸。适合初筛，不直接视为原像素测量。
- `make_overview_pages`：六对一页，供快速排除不可辨认/严重遮挡；不在overview上量对应误差。
- 现存 `panels/llvip_050001_RGB_raw.jpg` / `_IR_raw.jpg` 为1280×1024原分辨率入口；030237、030474、060258也有对应原图文件。它们的旧拒绝/接受范围不能被新名单自动覆盖。
- 本次实际打开了 `panels/llvip_050001_unannotated.jpg`，确认只含模态标题与原图内容，没有GT/预测框。没有重新测点或复核其六点。
- 旧初筛45对、另精查3对的解释见 `TRIAGE_45_PAIRS.md`、`LLVIP05_TEMPORAL_SCOPE.md` 和各批独立review。050001只接受六个局部静态结构点，不能授权整个prefix。

原图测量时应分别查看两张原分辨率文件。若借助并排640图定位，必须按每模态实际缩放比例及36px标题偏移换算，随后回原图确认；不得把“屏幕显示看起来相同”当作1px误差证据。

## 新候选选择工具

位置：`03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/prepare_geometry_review.py`。

输入示例结构（路径由根任务绑定真实产物）：

```json
{
  "datasets": [{
    "dataset": "dronevehicle",
    "train_roster": "/actual/drone_train/frozen_roster.json",
    "d2_jsonl": "/actual/drone_train/d2_anchors.jsonl",
    "natural_batches": "/actual/coverage_attempt/natural_batches.jsonl",
    "coverage_receipt": "/actual/coverage_attempt/coverage_receipt.json"
  }]
}
```

每份train_roster必须显式为train/fit。读取原D2的selected=true且quality_gate=true对象；其源图不在声明train名单中则报错。使用真实canonical路径或名单中提供别名，禁止用相同文件名把val图混入。

优先级固定为：自然64批出现过的源图优先；该分区内均衡数据集、真实来源组和代表性尺度；稳定路径破平局。不根据AP、教师领先幅度、logit误差或未来人工成功率排序。每图只入选一次。总数≤24，独立复核数固定ceil(20%×实际入选数)，若满24则为5对，均匀分散在最终冻结顺序。

若任何数据集缺少完整自然64批结果，输出 `DRAFT_AWAIT_NATURAL_STREAM`。有结果时要求receipt完成、actual_batches=64、generator_seed=20260907、数据集一致，且文件包含恰好batch0..63；然后才给 `FROZEN_CANDIDATE_ROSTER`。这个状态是名单冻结，**不是几何通过**。

命令：

```bash
python prepare_geometry_review.py --inputs /actual/geometry_inputs.json --output /actual/new_review_attempt
```

输出目录必须新建，包含：

1. `candidate_selection_receipt.json`：选择证据、自然命中批号、静态尺度组成、来源与全部候选身份。
2. `visual_review_roster.json`：只保留原图路径和审核身份，去除对象ID、尺度、模型选样数量等信息，交给点标注/独立复核者。
3. `primary_observations_template.json`：所有points为空，original_shape/uncertainty为null，状态PENDING，formal_geometry_verified=false；等待实际测量填入。

工具不会修改任何校准batch，不会过采样候选，不会因抽到24张就判断达到16个非零梯度batch。

## 真实点记录格式与接受边界

兼容既有 `primary_visual_observations_batch2.json`：每个observation包含audit_id、实际original_shape=[H,W]、原像素uncertainty_raw_px和points。每点至少记录id、物理结构文字描述、rgb=[x,y]、ir=[x,y]；建议新增可辨认性、遮挡/深度范围及拒绝原因文字。

标点者只读原图：优先同一静态井盖中心、路桩顶、能辨认的墙体/路缘交点；不能把GT框边、预测框、灯光/热扩散边缘或只有单模态可见结构当作对应真值。局部点不足或误差无法可靠量化时记录UNKNOWN并保留观察，不凑六点。

本轮代码仍要求至少六点、至少三个整图象限和严格局部范围；若只在一个象限的目标邻域找到六点，不能通过已有合同。两侧完整GT及按实测误差膨胀的边界邻域需受同一经验支持域覆盖，anchor也需在覆盖域；这由新 `geometry_support.py` 执行，不交给标注者凭框预判。

独立复核者必须查看原RGB/IR与点说明，保存明确的点身份、位置不确定度和范围判断。旧 `build_exact_contract` 只是落盘工具，不是独立审查器；本子任务没有调用它，也没有生成accepted_ids或虚构reviewer。接受后的source-folder/prefix仍仅是抽样层，不能扩张为相机标定组。

## source_group 的真实入口

正式校准不能用 `Path(image).parent` 代替Drone来源，因为processed train是扁平目录。已有冻结 `diagnose_opportunities.py::source_groups(data, images, dataset, split)` 可复用：

- data=`dataset_config(student_data_yaml)`，images=`split_images(data,'train')`。
- Drone从 `rgb_train_source_groups.tsv` 查stem→来源；缺失时抛错。
- LLVIP从 `splits/grouped_v1/fit.tsv` 查sequence_prefix；现有函数的fallback是stem前两位。
- 真实路径与symlink alias需在同一服务器canonical化。候选工具直接使用既有train D2 roster中的source_group，不重新猜测。

## 验证与未完成工作

本地6项合成测试覆盖：≤24/5独立复核、natural优先与多来源/尺度、唯一图、缺natural仅draft、重复运行一致、val名单拒绝、同basename异源拒绝、不完整64批拒绝、视觉feed无模型目标/点模板为空、输出不覆盖。

尚未运行真实候选命令，没有新的24对名单；没有SSH/GPU或原图下载。本任务只准备审核入口，由根任务取得真实自然流结果与原图后分配实际视觉审核。

## 真实64批结果到达后的名单执行

根任务已完成并下载 `remote_cpu1/coverage_drone_attempt1` 与 `coverage_llvip_attempt1` 的真实自然64批。按冻结工具执行后，产出 `geometry_review_24_v1/`：Drone286个静态候选图中38图命中自然流；LLVIP184个中35图命中。最终24唯一对为12 Drone＋12 LLVIP，全部命中自然流且有原静态selected对象，5对标记独立复核；未标任何新点、未设verified。

本批经根任务明确保留为双数据集**几何可辨认性和审核成本的有限试样**。每边仅12唯一图且本窗口各出现一次，不能单靠这批承诺16个有效定位batch；若Drone审核可行，再据预算增加合格exact-image覆盖。这个预算事实不说明整体不存在定位机会，也不要求无条件看完全部24。

本次在Windows读取服务器POSIX路径时，候选工具改为保留POSIX原路径做词法匹配，避免生成伪造的`E:\\mnt...`路径；natural记录同时含processed image和raw source，可提供明确别名。没有通过同basename猜测路径相同。6项候选工具测试此前已通过；实际24名单JSON仍保留原服务器图像路径，根任务据此取图。
