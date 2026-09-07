# 固定 anchor 特征的定位信息读出（2026-09-07）

> 配对 IR anchor 特征在两数据集的固定线性读出中均提供了超出 RGB DFL＋RGB 特征的条件定位线索，优于本次随机 donor；Drone还优于独立RGB特征对照。但特征读出的IoU均未超过GT辅助选anchor后的原生N DFL解码，不能据此宣称特征KD更优。

## 目的和固定设置

以相同对象比较定位输出与特征的可读出信息。执行前已落盘 [PROTOCOL.md](PROTOCOL.md)，未据结果调整参数、对象门槛、projection或损失。使用新导出固定 train/val；仅CPU，未启动模型推理或修改正在训练的分类/定位代码，无新哈希。

输入来自检测头前的固定anchor邻域P3/P4，原生最多3×3格点汇聚到2×2，**没有使用GT自适应ROI特征**。每层Gaussian128、seed20260907；所有标准化仅在train拟合，Ridge按对象均值目标固定alpha1。四边目标为统一RGB GT对同anchor的LTRB/stride。GT宽高没有进入任何输入。

尽管patch范围固定，**GT参与了N预测框与对象的几何分配和anchor选择**。共同anchor偏向N有较好预测的位置；这是GT条件下的信息诊断，不能当成检测器会自行选择的可部署输出，也不能把下表原生N的IoU解释成全验证集检测性能。无参考候选时的GT中心fallback已全部排除。

## 同cohort分母

|数据集/划分|真实GT对象|两侧GT匹配IoU≥.5|有N参考候选|四边距全部[0,14.99]：最终共同对象|
|---|---:|---:|---:|---:|
|Drone train|15,782|15,253|15,058|15,019|
|Drone dev|3,084|2,946|2,837|2,821|
|LLVIP train|2,753|2,753|2,742|2,728|
|LLVIP dev|643|643|584|548|

所有臂使用各数据集同一cohort。背景未进入拟合和评价；LLVIP共享标签IoU=1不构成独立几何配准证明。两数据集分别拟合，train结果仅作拟合状态记录。

## Dev结果

`N`为RGB基线，`T`为IR教师，`N0`为Drone独立seed0 RGB基线。`feature`是拼接P3/P4投影后的固定anchor特征。MSE为四边stride单位平方误差的平均，越低越好；IoU是逐对象均值，越高越好。各边独立MSE见原始CSV。

|臂|Drone MSE|Drone IoU|LLVIP MSE|LLVIP IoU|
|---|---:|---:|---:|---:|
|原生N DFL期望（无拟合）|.134982|.848451|.458020|.726483|
|Ridge N DFL64|.295195|.742626|.444346|.696563|
|Ridge N DFL＋T DFL|.222613|.771577|.324368|.732449|
|Ridge N DFL＋N0 DFL|.243820|.764177|无N0|无N0|
|Ridge N DFL＋N feature|.252077|.757384|.463705|.694536|
|再加配对T feature|.232339|.763415|.388241|.712817|
|再加N0 feature|.245157|.760211|无N0|无N0|
|再加shuffled T feature|.252759|.756972|.465964|.693303|
|仅anchor坐标/stride元数据|2.294986|.434518|1.818174|.463384|

Drone `N DFL＋N feature＋T feature`有1个dev对象预测负边距，`＋N0 feature`有2个，其余dev臂为0；严格按预先规则将这些对象IoU记0，MSE保留原始预测，没有事后clamp或删除。所有预测含在输出中。

## 可支持的判断

1. **固定anchor表示存在条件定位线索。** 增加配对T feature使Drone MSE从.252077降到.232339、LLVIP从.463705降到.388241；相同维数的shuffled输入没有该改善。Drone的独立RGB N0也有较小改善，必须保留这一多模型多样性对照，不能把所有增加特征带来的改善都归于跨模态。
2. **当前输出载体仍有直接信息。** 同一冻结线性读出下，N＋T DFL的dev MSE及IoU均优于N DFL＋N/T feature组合。LLVIP的N＋T DFL读出IoU为.732449，而原生N为.726483；这是有教师参与的GT条件诊断，不是单模态KD收益。
3. **不要把线性probe当作完整检测头。** Drone的原生N DFL解码明显优于全部Ridge臂；固定alpha线性函数无法等同原生softmax期望和现成检测头，且N anchor选择使用GT。因此本结果既不能证明特征无定位信息，也不能证明从特征蒸馏优于输出蒸馏。

没有置信区间、训练多seed或有效性干预，本轮不声称统计显著或paper-ready因果增益。跨模型recipe不完全相同；随机donor只限制同split无自配，并不保证跨图/同类/同尺度，具体同图donor数见summary。teacher参与推理的附加输入结果不等于学生可学性。

## 脚本、产物与验证

- [localization_readout.py](localization_readout.py)：CPU执行入口。
- [完整指标表](outputs/all_metrics.csv)：所有臂train/dev四边MSE、总体MSE、IoU及invalid数。
- [Drone summary](outputs/dronevehicle/summary.json)、[LLVIP summary](outputs/llvip/summary.json)：过滤链、模型路径、对照与资源/身份边界。
- 各数据集目录的 `predictions.npz`：同序object_id、GT距离、所有预测距离/IoU/invalid；`cohort_and_donors.csv`：源行索引与全部donor对应；`fitted_models.npz`：标准化参数、投影矩阵和回归系数。
- [已知真值测试](known_truth_test_receipt.json)：6项通过，包括LTRB逆变换、native DFL期望、负距离无效、mean-objective ridge闭式解、train-only标准化和同split无自配双射。
- [完成回执](outputs/completion_receipt.json)：两数据集完成，仅CPU。
- [独立审阅](INDEPENDENT_REVIEW.md)：共同cohort、source row/object_id/GT框独立重建exact；全部保存指标重算最大差0；native DFL期望距离exact，解码框对source原N42框最大差<5e-5像素（FP32舍入）。6项真值测试复跑通过，接受范围为本协议限定的描述性读出。

运行 `python localization_readout.py --test-only` 可仅检验基础算子；完整复算用 `python localization_readout.py --out <全新输出目录>`，脚本拒绝覆盖已有attempt。完整运行约6秒，两dataset的数据读取/预处理/拟合都包含在本次实际执行中。
