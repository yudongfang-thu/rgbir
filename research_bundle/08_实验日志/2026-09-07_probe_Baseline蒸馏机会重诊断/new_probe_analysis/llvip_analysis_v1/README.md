# Baseline 额外信息诊断

本报告是固定 train 拟合、dev 评价的探索性信息诊断，不是 AP、蒸馏训练收益或几何准入证明。

GT 共同窗口/anchor 与两模态无标注目标背景选择带有诊断特权；推理时无法直接取得这些关联。主分类表包含已配对和未配对GT位置；只有object_pairing=paired_gt子表可讨论标签关联对象，未配对位置的读出不能称同对象IR信息。

## 对象机会（全部 RGB GT 为分母）

|split|GT|N正确|T可修复|替换损伤|净修复|
|---|---:|---:|---:|---:|---:|
|train|2753|2367|303|110|193|
|val|643|404|146|42|104|

修复需 paired GT IoU≥0.5，教师预测在 RGB 标签坐标下类别正确、conf≥0.25、IoU≥0.5。损伤包括替换为未检出教师；不是实际蒸馏造成的伤害。互斥错误表将低置信排在分类/定位前，JSON 另保留可重叠错误旗标。

## 分类线性 probe（dev）

固定 α=1 的均值平方误差 ridge，未用 dev 选参数；各层固定随机投影128维后，仅 train 标准化。背景类别为 nc，macro recall 仅对当前子集有 GT 支持的类别求均值，与 balanced accuracy 同义。

固定 anchor patch 是前景/背景的主读出；GT-ROI仅为辅助且要求所有模型P3/P4区域共同有效。metadata_only只含log宽、高、stride，用来暴露抽样/ROI尺寸混杂，不是视觉分类优势。各臂注明cohort；不同cohort之间不直接作差。

|输入|cohort|dev N|All accuracy %|All balanced %|前景 accuracy %|前景 macro recall %|背景 recall %|
|---|---|---:|---:|---:|---:|---:|---:|
|N_anchor_feature_P3|main_fixed_anchor|1442|88.835|87.617|76.361|76.361|98.874|
|N_anchor_feature_P3_P4|main_fixed_anchor|1442|93.412|92.856|87.714|87.714|97.997|
|N_anchor_feature_P4|main_fixed_anchor|1442|91.956|91.344|85.692|85.692|96.996|
|N_logits|main_fixed_anchor|1442|96.186|95.723|91.446|91.446|100.000|
|N_logits+N_anchor_feature+T_anchor_feature_P3|main_fixed_anchor|1442|96.255|95.801|91.602|91.602|100.000|
|N_logits+N_anchor_feature+T_anchor_feature_P3_P4|main_fixed_anchor|1442|97.226|96.890|93.779|93.779|100.000|
|N_logits+N_anchor_feature+T_anchor_feature_P4|main_fixed_anchor|1442|96.602|96.190|92.379|92.379|100.000|
|N_logits+N_anchor_feature+T_logits_P3|main_fixed_anchor|1442|94.175|93.468|86.936|86.936|100.000|
|N_logits+N_anchor_feature+T_logits_P3_P4|main_fixed_anchor|1442|95.076|94.525|89.425|89.425|99.625|
|N_logits+N_anchor_feature+T_logits_P4|main_fixed_anchor|1442|94.175|93.605|88.336|88.336|98.874|
|N_logits+N_anchor_feature+shuffled_T_anchor_feature_P3|main_fixed_anchor|1442|94.105|93.421|87.092|87.092|99.750|
|N_logits+N_anchor_feature+shuffled_T_anchor_feature_P3_P4|main_fixed_anchor|1442|94.938|94.475|90.202|90.202|98.748|
|N_logits+N_anchor_feature+shuffled_T_anchor_feature_P4|main_fixed_anchor|1442|94.175|93.741|89.736|89.736|97.747|
|N_logits+N_anchor_feature_P3|main_fixed_anchor|1442|93.828|93.094|86.314|86.314|99.875|
|N_logits+N_anchor_feature_P3_P4|main_fixed_anchor|1442|94.868|94.413|90.202|90.202|98.623|
|N_logits+N_anchor_feature_P4|main_fixed_anchor|1442|94.105|93.664|89.580|89.580|97.747|
|N_logits+T_anchor_feature_P3|main_fixed_anchor|1442|96.394|95.972|92.068|92.068|99.875|
|N_logits+T_anchor_feature_P3_P4|main_fixed_anchor|1442|97.018|96.656|93.313|93.313|100.000|
|N_logits+T_anchor_feature_P4|main_fixed_anchor|1442|96.533|96.112|92.224|92.224|100.000|
|N_logits+T_logits|main_fixed_anchor|1442|92.233|91.291|82.582|82.582|100.000|
|N_logits+metadata|main_fixed_anchor|1442|95.700|95.179|90.358|90.358|100.000|
|N_logits+shuffled_T_anchor_feature_P3|main_fixed_anchor|1442|96.047|95.568|91.135|91.135|100.000|
|N_logits+shuffled_T_anchor_feature_P3_P4|main_fixed_anchor|1442|96.047|95.568|91.135|91.135|100.000|
|N_logits+shuffled_T_anchor_feature_P4|main_fixed_anchor|1442|96.255|95.801|91.602|91.602|100.000|
|metadata_only|main_fixed_anchor|1442|95.354|94.790|89.580|89.580|100.000|
|N_feature_P3|auxiliary_gt_roi_common_valid|1440|90.347|89.219|78.939|78.939|99.499|
|N_feature_P3_P4|auxiliary_gt_roi_common_valid|1440|92.778|91.888|83.775|83.775|100.000|
|N_feature_P4|auxiliary_gt_roi_common_valid|1440|92.361|91.435|82.995|82.995|99.875|
|N_logits+N_feature+T_feature_P3|auxiliary_gt_roi_common_valid|1440|96.597|96.178|92.356|92.356|100.000|
|N_logits+N_feature+T_feature_P3_P4|auxiliary_gt_roi_common_valid|1440|97.014|96.646|93.292|93.292|100.000|
|N_logits+N_feature+T_feature_P4|auxiliary_gt_roi_common_valid|1440|96.042|95.554|91.108|91.108|100.000|
|N_logits+N_feature+shuffled_T_feature_P3|auxiliary_gt_roi_common_valid|1440|92.778|91.888|83.775|83.775|100.000|
|N_logits+N_feature+shuffled_T_feature_P3_P4|auxiliary_gt_roi_common_valid|1440|93.958|93.214|86.427|86.427|100.000|
|N_logits+N_feature+shuffled_T_feature_P4|auxiliary_gt_roi_common_valid|1440|93.681|92.917|85.959|85.959|99.875|
|N_logits+N_feature_P3|auxiliary_gt_roi_common_valid|1440|92.778|91.888|83.775|83.775|100.000|
|N_logits+N_feature_P3_P4|auxiliary_gt_roi_common_valid|1440|93.958|93.214|86.427|86.427|100.000|
|N_logits+N_feature_P4|auxiliary_gt_roi_common_valid|1440|93.681|92.917|85.959|85.959|99.875|
|N_logits+T_feature_P3|auxiliary_gt_roi_common_valid|1440|97.639|97.348|94.696|94.696|100.000|
|N_logits+T_feature_P3_P4|auxiliary_gt_roi_common_valid|1440|98.194|98.003|96.256|96.256|99.750|
|N_logits+T_feature_P4|auxiliary_gt_roi_common_valid|1440|97.361|97.190|95.632|95.632|98.748|
|N_logits+shuffled_T_feature_P3|auxiliary_gt_roi_common_valid|1440|96.181|95.710|91.420|91.420|100.000|
|N_logits+shuffled_T_feature_P3_P4|auxiliary_gt_roi_common_valid|1440|96.250|95.788|91.576|91.576|100.000|
|N_logits+shuffled_T_feature_P4|auxiliary_gt_roi_common_valid|1440|96.250|95.788|91.576|91.576|100.000|
|N_logits_ROI_valid|auxiliary_gt_roi_common_valid|1440|96.250|95.788|91.576|91.576|100.000|
|N_region_logits_P3|auxiliary_gt_roi_common_valid|1440|74.583|71.482|43.214|43.214|99.750|
|N_region_logits_P3+N_feature|auxiliary_gt_roi_common_valid|1440|91.319|90.311|81.123|81.123|99.499|
|N_region_logits_P3+T_feature|auxiliary_gt_roi_common_valid|1440|97.500|97.223|94.696|94.696|99.750|
|N_region_logits_P3+T_region_logits|auxiliary_gt_roi_common_valid|1440|80.556|78.175|56.474|56.474|99.875|
|N_region_logits_P3_P4|auxiliary_gt_roi_common_valid|1440|89.306|87.988|75.975|75.975|100.000|
|N_region_logits_P3_P4+N_feature|auxiliary_gt_roi_common_valid|1440|93.472|92.668|85.335|85.335|100.000|
|N_region_logits_P3_P4+T_feature|auxiliary_gt_roi_common_valid|1440|98.056|97.847|95.944|95.944|99.750|
|N_region_logits_P3_P4+T_region_logits|auxiliary_gt_roi_common_valid|1440|94.792|94.150|88.300|88.300|100.000|
|N_region_logits_P4|auxiliary_gt_roi_common_valid|1440|84.444|82.527|65.055|65.055|100.000|
|N_region_logits_P4+N_feature|auxiliary_gt_roi_common_valid|1440|92.986|92.122|84.243|84.243|100.000|
|N_region_logits_P4+T_feature|auxiliary_gt_roi_common_valid|1440|96.806|96.566|94.384|94.384|98.748|
|N_region_logits_P4+T_region_logits|auxiliary_gt_roi_common_valid|1440|90.556|89.392|78.783|78.783|100.000|
|metadata_only_ROI_valid|auxiliary_gt_roi_common_valid|1440|95.347|94.774|89.548|89.548|100.000|

以上 T 特征输入是在 probe 评价阶段使用教师的双模态诊断；它衡量该线性函数族是否可解码额外信息，不能叫作 RGB-only 推理提升或 KD 增益。shuffled 在各 split 内按固定 seed 全局置换，不条件化于标签；同模态 N0 仅在已导出时报告。N_region_logits 是共同区域内容代理，不是冻结 C1 算子的复刻。教师logits与feature载体对照固定相同样本；输入维数、参数数目及有效正则仍不同，不能把差异解释为KD带宽/载体的因果效果。

## 同 anchor 定位分布（dev）

正式列示的头部代理子集：548/643 RGB GT，需配对、四边GT距离在支持域且存在N42粗候选。

|量|值|
|---|---:|
|N−T GT-DFL CE（4边均值）的均值|0.15589796732108194|
|教师GT CE更低的对象数|335|
|KD(T=2) 与 GT 的 native DFL-logit 梯度 cosine 均值|0.40631273010635316|
|cosine正 / 负 / 无定义|434 / 114 / 0|

四边目标不截断，支持域外单列；GT CE 用 T=1，KL 用 T=2 且乘T²，梯度均对native raw DFL logits。正cosine只说明该头部局部代理方向一致，不是共享特征参数梯度或完整训练可学性。

## 产物与局限

完整逐类、来源组、尺度及 train-only 亮度分桶统计见 summary.json；probe_predictions.npz 保存逐行预测，probe_train_transforms.npz 保存仅训练拟合的标准化参数，dfl_per_object.csv 保存GT逐对象头部量。

这是一个固定抽样/固定模型组合的探索性诊断，没有多seed蒸馏训练、四臂归因或完整AP；所有层与臂同时列示，不按dev结果选择最佳配置。标签对象关联、同画布坐标与背景排除均不构成独立像素配准证据。
