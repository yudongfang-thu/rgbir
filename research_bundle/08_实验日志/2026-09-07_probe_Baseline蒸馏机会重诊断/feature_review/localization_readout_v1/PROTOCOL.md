# 固定 anchor 特征的定位信息读出 v1

> 在首次读出结果产生之前固定本协议。仅 CPU 回归探针；GT 参与对象关联和 anchor 选择，结果不能当作可部署模型、检测 AP 或 KD 收益。

## 来源与冻结对象

分别等待 `remote_exports/{dronevehicle,llvip}_full_attempt1/collection_receipt.json` 完成，再读取同目录 `objects.jsonl`、`features.npz`、`logits.npz` 和模型身份。当前两数据集使用既定 train/val 样本；val 即本探针的开发集，不新划分或调参。

执行前路径核对：实际采集回执位于probe根目录 `{dataset}_collection_receipt.json`，代码同时接受该实际位置；这是文件入口修正，不改变对象或分析协议。

同一数据集所有臂仅使用以下共同对象：排除背景；`paired_gt_iou>=0.5`；`anchor_has_reference_candidate=true`；RGB GT 四边距公共 anchor 的 L/T/R/B 距离除 stride 后全部在 `[0,14.99]`。保留各过滤步骤分母。非有限输入作为错误中止，不按模型表现删对象。GT中心fallback不进入本探针。

目标 `y=[anchor_x−x1,anchor_y−y1,x2−anchor_x,y2−anchor_y]/stride`，四边同权，目标仅在 train 中心化、不按边缩放。定位预测距离不截断；任一负边距使框无效、IoU记0并报告invalid数。有效预测可超过14.99，按原值解码，不事后clip。

## 表示与固定回归

- DFL为原始4×16 logits，展平64维；直接基线为每边softmax(logits)后的bin期望，无拟合。
- 特征只取 `*_anchor_P3/P4`，来自固定anchor邻域，原导出为最多3×3原生格点汇聚成2×2。不得读取GT自适应ROI `*_P3/P4`。
- 每层、每模型使用固定高斯投影到128维：`default_rng(20260907).normal((input_dim,128))/sqrt(128)`，对相同输入维度复用相同矩阵。P3/P4投影结果拼接256维，记作该模型的feature块。投影与标签/结果无关。
- 每个输入块投影后使用train均值/总体SD标准化；近零SD置1。同一数据集各臂复用相同块变换。无validation拟合、PCA或超参搜索。
- Ridge固定 `alpha=1`，解 `(XᵀX/n + I)W = Xᵀ(Y−mean_trainY)/n`，intercept不惩罚。即对象平均平方误差之和加固定L2项；四输出同权。报告指标的MSE再对四边平均。
- geometry meta仅 `[anchor_x/640,anchor_y/640,log(stride)]`，绝不输入GT宽高/类别/匹配IoU。

## 固定比较臂

1. N42 DFL原生期望距离（无训练）。
2. Ridge N42 DFL64。
3. Ridge N42 DFL64 + T42 DFL64。
4. Ridge N42 DFL64 + N0 DFL64（模型存在时）。
5. Ridge N42 DFL64 + N42 anchor feature。
6. Ridge N42 DFL64 + N42 anchor feature + T42 anchor feature。
7. Ridge N42 DFL64 + N42 anchor feature + N0 anchor feature（模型存在时）。
8. Ridge N42 DFL64 + N42 anchor feature + shuffled T42 anchor feature。
9. Ridge geometry meta-only。

shuffle分别在train/val共同cohort内，以object_id排序、seed20260907置乱顺序后循环移位生成双射；不得自配，保存donor对象ID。该null不声称同类/同尺度/跨图匹配，报告同图donor比例。它只检验本冻结设定下的对象配对关联，不能替代正式shuffled训练控制。LLVIP无N0时相应两臂标记不可用，不伪造对照。

## 输出和解释

保存各臂train/dev四边MSE、四边总体MSE、按预测距离解码后对RGB GT的mean IoU、invalid数；同一cohort保存全部距离预测/逐对象IoU及donor映射。train结果仅检查拟合状态，结论以dev为准。执行少量已知真值测试：距离/框逆变换、native softmax bin期望、负距离IoU0、mean-objective ridge闭式解、标准化只用train、shuffle双射无自配。

多模态输入臂的提升只是教师辅助的条件信息诊断；GT关联仍有特权、所有模型并非严格同recipe，四臂单模态部署KD效果需要另外训练验证。固定alpha下线性probe负结果也不证明不存在非线性可读出的定位信息。
