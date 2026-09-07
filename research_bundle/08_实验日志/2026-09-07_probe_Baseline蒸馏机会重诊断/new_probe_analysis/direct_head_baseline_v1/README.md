# 原生head直接读出 sanity（固定 conf=.25）

**结论：该对照只诊断固定ridge是否低估现有head可读信息；所有原始probe、α及结果保持原样。**

|数据集|模型|cohort|dev N|accuracy %|balanced %|前景macro recall %|背景recall %|
|---|---|---|---:|---:|---:|---:|---:|
|dronevehicle|N42|main_fixed_anchor|3794|81.813|66.762|60.143|99.859|
|dronevehicle|N42|auxiliary_gt_roi_common_valid|3725|82.336|66.959|60.380|99.859|
|dronevehicle|T42|main_fixed_anchor|3794|75.119|61.958|54.349|100.000|
|dronevehicle|T42|auxiliary_gt_roi_common_valid|3725|75.248|62.044|54.452|100.000|
|dronevehicle|N0|main_fixed_anchor|3794|76.621|63.994|56.821|99.859|
|dronevehicle|N0|auxiliary_gt_roi_common_valid|3725|77.101|64.166|57.028|99.859|
|llvip|N42|main_fixed_anchor|1442|86.893|85.303|70.607|100.000|
|llvip|N42|auxiliary_gt_roi_common_valid|1440|87.014|85.413|70.827|100.000|
|llvip|T42|main_fixed_anchor|1442|84.327|82.426|64.852|100.000|
|llvip|T42|auxiliary_gt_roi_common_valid|1440|84.444|82.527|65.055|100.000|

输入是此前冻结的共同anchor raw类别logits，直接按原有conf=.25规则拒识背景，其余argmax类别，不拟合参数、不调阈值。各原生head分别使用本模型logits，GT-ROI有效cohort复用冻结probe保存的相同行掩码。

这仍是GT关联位置的读出，不能称AP或常规检测召回率。共同有效ROI表只是相同行集合，直接head并未使用ROI特征。对未配对GT位置的IR读出不能声称同对象。

若原生head的少数类召回显著高于N_logits ridge，说明该固定线性读出器/正则与类别不平衡限制了probe；不能由高维feature相对弱ridge的差值宣称logits缺信息、feature是更优KD载体或预期训练增益。
