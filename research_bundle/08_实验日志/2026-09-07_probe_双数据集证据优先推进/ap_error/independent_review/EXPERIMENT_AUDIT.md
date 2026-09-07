# Drone TIDE 诊断独立审阅（2026-09-07）

**数值与执行链通过限定范围核验。总体暂记 WARN，仅因报告首句应限定为“六类 main error 中的最大贡献”；同表 special FP oracle 的数值更大。该措辞问题不影响保存的 AP/dAP、排序统计或三 seed 差值。**

审阅者 `/root/loc_stress`，独立于 AP 诊断执行者。按 experiment-audit 技能执行；用户明确禁止新增 hash，故仅记录 stat、执行中源码字节相等及新旧 runner 核心函数 AST 相等，不重算 SHA。既有 hash attempt 原样保留。未连接服务器、未使用 GPU、未改 AP 输入/原结果；审阅新增文件仅在本目录。

## 实际核验范围

1. 读取冻结 PROTOCOL/ADDENDUM、当前和实际执行 runner、官方本地 TIDE 源码、六端点 JSON/summary、旧补评合同/receipt/原始 gzip 逐图数据和摘要生成代码。
2. 六端点全部 raw 输入独立复核：每份 **1469 唯一 dev 图、22462 RGB GT**，contract roster 与完整逐图记录相等；六份 sorted image/canvas/original_shape/GT框/类逐字段完全一致，空预测图保留。GT类分母为 **18965/710/1336/751/700**，与原冻结 rgb.data.yaml 的 car/freight car/truck/bus/van 顺序一致。
3. 只对 **N_s0 的两个 IoU** 重新构造官方 Data/TIDE 对象，未调用执行者的 `make_data/evaluate/summarize_run`。AP、六main与两个special dAP逐项复现；AP50 **76.77070078999334pp**，AP75 **63.81923085055706pp**。其余五端点核库存、保存执行结果、既有NumPy交叉核验记录和报告算术，没有声称此次全部重跑官方引擎。
4. N_s0 每个预测的 `used` 与官方 `ap_data.data_points` 真值一致；oracle前后 `used` 未改变。独立从AP数据点重算pooled及五类的全部score bins、TP/FP分位数、错误类型计数，与原JSON逐项一致。
5. 全部六端点 GT/TP/FN/分数桶加总、三seed均值、样本SD(ddof=1)及C0−N差值独立重算通过。当前与实际执行runner的九个核心函数AST一致；后续移除hash仅改变来源元数据，不改变评估算法。
6. 原已知真值测试重跑，输出JSON相等；另补同类同分数竞争顺序真值：TP在前 AP=100pp、FP在前 AP=50pp，官方与NumPy实现均符合预定值。原测试覆盖七种错误/正确状态、空图、无GT类别和单调score变换。

完整机器回执见 [recompute_receipt.json](recompute_receipt.json)，独立附加真值见 [tie_truth_receipt.json](tie_truth_receipt.json)。回执所列输入在本次复算前后stat一致，源码字节相等。

## 关键语义核对

- **GT与坐标：PASS。** 原保存器在native `update_metrics`之后读取相同 `_prepare_batch` 的GT和相同post-NMS预测。矩形输入实际为544×672；adapter仅xyxy→xywh，不重缩放、不裁剪。输入是保存的真实RGB标签，不是模型生成reference。未重新打开服务器原始标签/图像，因此本审阅核验到补评快照证据链，不认证标注本身完全无漏标。
- **完整性：PASS（固定后处理范围）。** 六端点保留conf≥.001、max_det300下全部预测，逐图没有额外TIDE默认100截断。完整性指该冻结post-NMS协议，不能推广到无限低阈值或全部pre-NMS候选。原FP32/rect/non-aug/多标签NMS/last-EMA身份有合同与源代码支持。
- **AP101：PASS（TIDE定义）。** 分数优先逐检测匹配，类别宏平均，101个recall阈值按源码`x/100`生成，再取插值precision的算术平均，单位为pp；不是pinned evaluator的候选IoU优先去重及np.interp/梯形积分。所有当前真实类别都有GT；无GT但有预测的类别在TIDE中可参与宏平均，此特性不应未经说明移植为一般COCO口径。本审阅未重复根任务的COCOeval交叉核验，也不主张其默认recThrs逐位等同TIDE。
- **oracle：PASS（官方语义）。** `fix_main_errors(progressive=False)`各自相对于原AP独立计算，并将负dAP截为0；六项不可相加。Cls/Loc可能修正可用对象或抑制错误预测，并非都产生一条正确检测；Both/Dupe/Bkg删除相关预测。Miss/FN会改变GT分母；special FP将正确/错误预测score分别置1/0，FN改变漏检分母。这些不是KD可达到的收益或可训练的新检测集。
- **排序used：PASS（本缓存）。** 官方 `_clear`只清理GT临时字段，不删除预测used；raw每图≤300且无ignore，所以本次全部预测都获得True/False，未出现used缺失或None。未来若接入ignore或每图>max_det的缓存，需要另核排除语义，不能仅凭当前通过笼统复用。

## 可引用判断与措辞修正

三seed TIDE C0−N AP50为 **−.01713/−.08983/+.34448pp**，均值 **+.0792±.2326pp**；AP75为 **+.23665/−.33213/+.52705pp**，均值 **+.1439±.4370pp**。这些不是原pinned mAP50–95的新估计；其既有差值仍为+.266655±.144373pp。

score≥.25的Bkg计数C0−N **+3/+5/+46** 与AP50 Bkg dAP两降一升同时成立；固定阈值数量增加不能独自判定背景AP损伤。上述算术与语义均支持当前描述性解释。

**需修正的具体措辞：** `README_DRONE.md`与`summarize_drone.py`开头将“最大独立错误贡献”写成类别/AP75定位，但若包含同表special FP，该比较不成立。例如N的AP50 Cls≈10.36pp而FP≈17.04pp。建议统一改成“**在六类 main error 中，AP50下类别oracle贡献最大，AP75下定位oracle贡献最大**”。这只是限定比较集合，不改任何数值。修正后可将本次限定描述性分析整体标为PASS，不需重跑评估。

本次不验证LLVIP新输入、class_oracle补充attempt、三seed四臂跨模态因果归因、教师可修复性或L1几何准入，不据oracle大小选择新方法或预测训练收益。

## 源码入口

- `run_tide_audit.py:29/43/69/112/133/155`：Data输入、NumPy AP、库存、used统计、oracle调用、真值。
- `official_tide/tidecv/quantify.py:187/198/333/364`：清GT临时字段、逐图处理、独立main、special。
- `official_tide/tidecv/ap.py:66/148`：101点AP、宏平均。
- `official_tide/tidecv/errors/main_errors.py:6/24/78/92/106`：Cls/Loc/Miss/FP/FN算子。
- 旧N_s0补评snapshot `eval_evidence/source_snapshot/trainer/01_evaluate_independent.py:98`：read-only对象导出；`02_val.py:121`：multi_label=True。
