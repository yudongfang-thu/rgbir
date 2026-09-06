# 真实训练Loader CPU等价验收

> **真实CPU验收通过：Drone和LLVIP各三图，全部原有RGB/IR张量、标签、collate以及Python/NumPy/Torch RNG后继完全一致；80个增强输出框与独立原GT经记录矩阵投影一致，最大误差0.000070572像素。没有GPU、模型或optimizer。**

## 目的与设置
独立脚本对比release_v4固定vendor DualLabelRGBIRDataset与root实现的TrackedDualLabelRGBIRDataset。验收前直接比较本地和release_v4 tracked源码文本一致，没有哈希。Drone和LLVIP各取首/中/末三图，用固定augmentation seed0/42/123；比较所有原有样本字段、RGB/IR张量与标签、collate、Python/NumPy/Torch CPU RNG后继状态。独立读取native dataset实际label_files对应的原始YOLO文本，用记录矩阵投影并检查全部输出GT坐标，允许native裁剪过滤。检测图像和标签来自train。

## 实际结果

执行时间2026-09-07 02:56:23–02:56:39 +08:00，`CUDA_VISIBLE_DEVICES=''`，torch CPU线程4，无模型前向或GPU初始化需求。

| 数据集 | 实际train数 | 索引 | seed | RGB/IR输出GT数 | RGB最大坐标误差px | IR最大坐标误差px |
|---|---:|---:|---:|---:|---:|---:|
| Drone | 17990 | 0 | 0 | 9/9 | 0.000045504 | 0.000047684 |
| Drone | 17990 | 8995 | 42 | 4/4 | 0.000015497 | 0.000055075 |
| Drone | 17990 | 17989 | 123 | 23/23 | 0.000070572 | 0.000070572 |
| LLVIP | 9619 | 0 | 0 | 2/2 | 0.000030790 | 0.000030790 |
| LLVIP | 9619 | 4809 | 42 | 1/1 | 0.000028275 | 0.000028275 |
| LLVIP | 9619 | 9618 | 123 | 1/1 | 0.000036001 | 0.000036001 |

六样本全部原有字段相同；两个三样本collate全字段相同；每样本三种RNG后继状态相同；RGB/IR增强矩阵精确相同。原GT→增强GT采用四角齐次变换、HBB包络、画布裁剪，再按同类一对一坐标匹配；容差在运行前固定为0.001像素，不依据结果修改。

## attempt与验收结论

第一次执行已通过Drone三图，但审计脚本从LLVIP canonical raw image推导YOLO txt，raw目录没有该标签而失败。这是验收脚本路径假设错误，**不是tracked loader失败**。第一次`result.json/ssh_stdout.json`和94 attempt1原件保留。修复为读取native实际label_files后另建attempt2；没有改root loader、原训练或release文件。release_v2预检发现source已过时，在任何数据运行前拒绝并切到实际一致的release_v4。

结论：接受这次**真实CPU loader等价和坐标追踪检查**；验收脚本独立于root loader实现，原GT投影没有调用被测parameter_matrix。没有新发现需要修改root loader的缺陷。

分析器v2此前20项CPU fixture和真实4个端点的bound recipe检查也通过，详细见上级`ANALYZER_V2_REVIEW_RESPONSE.md`。由于本agent是分析器作者，该项属于作者自检，不能称分析器独立接受；仍由root另行判断。

## 产物与限制
94原件：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907/audits/loader_equivalence_attempt2/{verify_loader_equivalence.py,result.json,stdout.log,stderr.log}`。

本地`attempt2/result.json`和`attempt2/verify_loader_equivalence.py`保留成功原件；`run_remote.py --attempt <新的序号>`可重跑且拒绝覆盖旧attempt。新模块`verify_loader_equivalence.py`为验收入口。

每数据集三图的CPU一致不能取代24次optimizer update训练canary或全部数据遍历，也不证明传感器物理配准、C/N训练轨迹等价或L有效性。没有把本验收冒充formal所需的canary acceptance。
