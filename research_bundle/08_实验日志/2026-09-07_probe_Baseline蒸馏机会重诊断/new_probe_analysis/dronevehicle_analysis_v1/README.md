# Baseline 额外信息诊断

本报告是固定 train 拟合、dev 评价的探索性信息诊断，不是 AP、蒸馏训练收益或几何准入证明。

GT 共同窗口/anchor 与两模态无标注目标背景选择带有诊断特权；推理时无法直接取得这些关联。主分类表包含已配对和未配对GT位置；只有object_pairing=paired_gt子表可讨论标签关联对象，未配对位置的读出不能称同对象IR信息。

## 对象机会（全部 RGB GT 为分母）

|split|GT|N正确|T可修复|替换损伤|净修复|
|---|---:|---:|---:|---:|---:|
|train|15782|13574|1602|905|697|
|val|3084|2386|461|193|268|

修复需 paired GT IoU≥0.5，教师预测在 RGB 标签坐标下类别正确、conf≥0.25、IoU≥0.5。损伤包括替换为未检出教师；不是实际蒸馏造成的伤害。互斥错误表将低置信排在分类/定位前，JSON 另保留可重叠错误旗标。

## 分类线性 probe（dev）

固定 α=1 的均值平方误差 ridge，未用 dev 选参数；各层固定随机投影128维后，仅 train 标准化。背景类别为 nc，macro recall 仅对当前子集有 GT 支持的类别求均值，与 balanced accuracy 同义。

固定 anchor patch 是前景/背景的主读出；GT-ROI仅为辅助且要求所有模型P3/P4区域共同有效。metadata_only只含log宽、高、stride，用来暴露抽样/ROI尺寸混杂，不是视觉分类优势。各臂注明cohort；不同cohort之间不直接作差。

|输入|cohort|dev N|All accuracy %|All balanced %|前景 accuracy %|前景 macro recall %|背景 recall %|
|---|---|---:|---:|---:|---:|---:|---:|
|N_anchor_feature_P3|main_fixed_anchor|3794|87.559|36.308|85.603|24.359|96.056|
|N_anchor_feature_P3_P4|main_fixed_anchor|3794|90.485|49.542|88.781|39.873|97.887|
|N_anchor_feature_P4|main_fixed_anchor|3794|88.745|41.850|87.484|31.375|94.225|
|N_logits|main_fixed_anchor|3794|88.139|32.797|86.089|19.947|97.042|
|N_logits+N0_anchor_feature_P3|main_fixed_anchor|3794|90.459|46.924|88.521|36.534|98.873|
|N_logits+N0_anchor_feature_P3_P4|main_fixed_anchor|3794|92.198|59.191|90.629|51.226|99.014|
|N_logits+N0_anchor_feature_P4|main_fixed_anchor|3794|91.697|55.168|90.110|46.483|98.592|
|N_logits+N0_logits|main_fixed_anchor|3794|87.876|32.803|85.636|19.842|97.606|
|N_logits+N_anchor_feature+N0_anchor_feature_P3|main_fixed_anchor|3794|91.460|54.224|89.754|45.294|98.873|
|N_logits+N_anchor_feature+N0_anchor_feature_P3_P4|main_fixed_anchor|3794|93.332|64.935|91.991|58.091|99.155|
|N_logits+N_anchor_feature+N0_anchor_feature_P4|main_fixed_anchor|3794|92.620|60.978|91.342|53.540|98.169|
|N_logits+N_anchor_feature+N0_logits_P3|main_fixed_anchor|3794|90.749|49.748|88.911|39.951|98.732|
|N_logits+N_anchor_feature+N0_logits_P3_P4|main_fixed_anchor|3794|92.567|60.804|91.180|53.247|98.592|
|N_logits+N_anchor_feature+N0_logits_P4|main_fixed_anchor|3794|91.987|57.122|90.629|48.968|97.887|
|N_logits+N_anchor_feature+T_anchor_feature_P3|main_fixed_anchor|3794|92.119|53.508|90.370|44.266|99.718|
|N_logits+N_anchor_feature+T_anchor_feature_P3_P4|main_fixed_anchor|3794|93.648|64.161|92.250|57.050|99.718|
|N_logits+N_anchor_feature+T_anchor_feature_P4|main_fixed_anchor|3794|93.015|60.571|91.505|52.770|99.577|
|N_logits+N_anchor_feature+T_logits_P3|main_fixed_anchor|3794|90.590|48.082|88.684|37.924|98.873|
|N_logits+N_anchor_feature+T_logits_P3_P4|main_fixed_anchor|3794|92.356|59.413|90.856|51.521|98.873|
|N_logits+N_anchor_feature+T_logits_P4|main_fixed_anchor|3794|92.066|56.815|90.629|48.516|98.310|
|N_logits+N_anchor_feature+shuffled_T_anchor_feature_P3|main_fixed_anchor|3794|90.116|46.018|88.132|35.475|98.732|
|N_logits+N_anchor_feature+shuffled_T_anchor_feature_P3_P4|main_fixed_anchor|3794|92.172|58.547|90.629|50.482|98.873|
|N_logits+N_anchor_feature+shuffled_T_anchor_feature_P4|main_fixed_anchor|3794|91.487|54.357|90.013|45.651|97.887|
|N_logits+N_anchor_feature_P3|main_fixed_anchor|3794|90.142|46.253|88.197|35.786|98.592|
|N_logits+N_anchor_feature_P3_P4|main_fixed_anchor|3794|92.198|58.554|90.661|50.490|98.873|
|N_logits+N_anchor_feature_P4|main_fixed_anchor|3794|91.539|54.441|90.110|45.780|97.746|
|N_logits+T_anchor_feature_P3|main_fixed_anchor|3794|91.197|47.547|89.235|37.112|99.718|
|N_logits+T_anchor_feature_P3_P4|main_fixed_anchor|3794|92.752|57.548|91.083|49.058|100.000|
|N_logits+T_anchor_feature_P4|main_fixed_anchor|3794|91.697|52.097|89.851|42.573|99.718|
|N_logits+T_logits|main_fixed_anchor|3794|88.324|32.926|86.154|19.962|97.746|
|N_logits+metadata|main_fixed_anchor|3794|89.642|39.024|87.646|27.167|98.310|
|N_logits+shuffled_T_anchor_feature_P3|main_fixed_anchor|3794|88.166|32.820|86.089|19.947|97.183|
|N_logits+shuffled_T_anchor_feature_P3_P4|main_fixed_anchor|3794|88.166|32.820|86.089|19.947|97.183|
|N_logits+shuffled_T_anchor_feature_P4|main_fixed_anchor|3794|88.166|32.803|86.122|19.955|97.042|
|metadata_only|main_fixed_anchor|3794|70.163|16.667|86.316|20.000|0.000|
|N_feature_P3|auxiliary_gt_roi_common_valid|3725|85.074|35.154|82.454|22.945|96.197|
|N_feature_P3_P4|auxiliary_gt_roi_common_valid|3725|87.597|49.160|85.240|39.470|97.606|
|N_feature_P4|auxiliary_gt_roi_common_valid|3725|85.852|39.087|83.814|28.003|94.507|
|N_logits+N0_feature_P3|auxiliary_gt_roi_common_valid|3725|89.987|50.121|87.828|40.314|99.155|
|N_logits+N0_feature_P3_P4|auxiliary_gt_roi_common_valid|3725|91.087|58.648|89.055|50.434|99.718|
|N_logits+N0_feature_P4|auxiliary_gt_roi_common_valid|3725|90.067|50.446|87.927|40.705|99.155|
|N_logits+N_feature+N0_feature_P3|auxiliary_gt_roi_common_valid|3725|90.362|54.206|88.259|45.188|99.296|
|N_logits+N_feature+N0_feature_P3_P4|auxiliary_gt_roi_common_valid|3725|91.758|62.413|89.917|54.980|99.577|
|N_logits+N_feature+N0_feature_P4|auxiliary_gt_roi_common_valid|3725|91.007|57.202|89.088|48.811|99.155|
|N_logits+N_feature+T_feature_P3|auxiliary_gt_roi_common_valid|3725|91.302|52.207|89.254|42.648|100.000|
|N_logits+N_feature+T_feature_P3_P4|auxiliary_gt_roi_common_valid|3725|92.913|62.268|91.244|54.722|100.000|
|N_logits+N_feature+T_feature_P4|auxiliary_gt_roi_common_valid|3725|91.919|56.810|90.116|48.257|99.577|
|N_logits+N_feature+shuffled_T_feature_P3|auxiliary_gt_roi_common_valid|3725|89.020|44.910|86.667|34.089|99.014|
|N_logits+N_feature+shuffled_T_feature_P3_P4|auxiliary_gt_roi_common_valid|3725|90.443|54.452|88.458|45.568|98.873|
|N_logits+N_feature+shuffled_T_feature_P4|auxiliary_gt_roi_common_valid|3725|89.477|46.738|87.264|36.311|98.873|
|N_logits+N_feature_P3|auxiliary_gt_roi_common_valid|3725|89.020|44.979|86.667|34.173|99.014|
|N_logits+N_feature_P3_P4|auxiliary_gt_roi_common_valid|3725|90.389|54.422|88.425|45.560|98.732|
|N_logits+N_feature_P4|auxiliary_gt_roi_common_valid|3725|89.530|46.768|87.297|36.319|99.014|
|N_logits+T_feature_P3|auxiliary_gt_roi_common_valid|3725|90.550|48.489|88.325|38.187|100.000|
|N_logits+T_feature_P3_P4|auxiliary_gt_roi_common_valid|3725|91.624|54.486|89.652|45.383|100.000|
|N_logits+T_feature_P4|auxiliary_gt_roi_common_valid|3725|91.141|50.082|89.121|40.155|99.718|
|N_logits+shuffled_T_feature_P3|auxiliary_gt_roi_common_valid|3725|88.054|32.855|85.871|19.961|97.324|
|N_logits+shuffled_T_feature_P3_P4|auxiliary_gt_roi_common_valid|3725|88.000|32.808|85.871|19.961|97.042|
|N_logits+shuffled_T_feature_P4|auxiliary_gt_roi_common_valid|3725|87.973|32.785|85.871|19.961|96.901|
|N_logits_ROI_valid|auxiliary_gt_roi_common_valid|3725|88.054|32.838|85.904|19.969|97.183|
|N_region_logits_P3|auxiliary_gt_roi_common_valid|3725|85.342|31.558|83.781|19.476|91.972|
|N_region_logits_P3+N0_region_logits|auxiliary_gt_roi_common_valid|3725|85.530|32.354|82.554|19.190|98.169|
|N_region_logits_P3+N_feature|auxiliary_gt_roi_common_valid|3725|85.906|39.146|83.051|27.370|98.028|
|N_region_logits_P3+T_feature|auxiliary_gt_roi_common_valid|3725|88.564|41.636|85.871|29.964|100.000|
|N_region_logits_P3+T_region_logits|auxiliary_gt_roi_common_valid|3725|86.738|32.796|83.748|19.468|99.437|
|N_region_logits_P3_P4|auxiliary_gt_roi_common_valid|3725|85.691|34.931|84.245|23.551|91.831|
|N_region_logits_P3_P4+N0_region_logits|auxiliary_gt_roi_common_valid|3725|87.087|43.821|84.511|32.980|98.028|
|N_region_logits_P3_P4+N_feature|auxiliary_gt_roi_common_valid|3725|89.020|55.296|86.733|46.608|98.732|
|N_region_logits_P3_P4+T_feature|auxiliary_gt_roi_common_valid|3725|90.846|54.659|88.690|45.591|100.000|
|N_region_logits_P3_P4+T_region_logits|auxiliary_gt_roi_common_valid|3725|88.564|43.663|86.003|32.508|99.437|
|N_region_logits_P4|auxiliary_gt_roi_common_valid|3725|69.638|16.667|86.036|20.000|0.000|
|N_region_logits_P4+N0_region_logits|auxiliary_gt_roi_common_valid|3725|70.497|25.082|87.098|30.098|0.000|
|N_region_logits_P4+N_feature|auxiliary_gt_roi_common_valid|3725|87.490|47.658|85.771|38.232|94.789|
|N_region_logits_P4+T_feature|auxiliary_gt_roi_common_valid|3725|89.584|49.013|87.828|39.407|97.042|
|N_region_logits_P4+T_region_logits|auxiliary_gt_roi_common_valid|3725|70.443|24.173|87.032|29.007|0.000|
|metadata_only_ROI_valid|auxiliary_gt_roi_common_valid|3725|69.638|16.667|86.036|20.000|0.000|

以上 T 特征输入是在 probe 评价阶段使用教师的双模态诊断；它衡量该线性函数族是否可解码额外信息，不能叫作 RGB-only 推理提升或 KD 增益。shuffled 在各 split 内按固定 seed 全局置换，不条件化于标签；同模态 N0 仅在已导出时报告。N_region_logits 是共同区域内容代理，不是冻结 C1 算子的复刻。教师logits与feature载体对照固定相同样本；输入维数、参数数目及有效正则仍不同，不能把差异解释为KD带宽/载体的因果效果。

## 同 anchor 定位分布（dev）

正式列示的头部代理子集：2821/3084 RGB GT，需配对、四边GT距离在支持域且存在N42粗候选。

|量|值|
|---|---:|
|N−T GT-DFL CE（4边均值）的均值|-0.12560703499852452|
|教师GT CE更低的对象数|1102|
|KD(T=2) 与 GT 的 native DFL-logit 梯度 cosine 均值|0.2084077871141649|
|cosine正 / 负 / 无定义|1816 / 1005 / 0|

四边目标不截断，支持域外单列；GT CE 用 T=1，KL 用 T=2 且乘T²，梯度均对native raw DFL logits。正cosine只说明该头部局部代理方向一致，不是共享特征参数梯度或完整训练可学性。

## 产物与局限

完整逐类、来源组、尺度及 train-only 亮度分桶统计见 summary.json；probe_predictions.npz 保存逐行预测，probe_train_transforms.npz 保存仅训练拟合的标准化参数，dfl_per_object.csv 保存GT逐对象头部量。

这是一个固定抽样/固定模型组合的探索性诊断，没有多seed蒸馏训练、四臂归因或完整AP；所有层与臂同时列示，不按dev结果选择最佳配置。标签对象关联、同画布坐标与背景排除均不构成独立像素配准证据。
