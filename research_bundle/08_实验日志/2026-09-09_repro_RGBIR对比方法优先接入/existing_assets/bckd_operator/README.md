# BCKD-BCDL 作者分类算子：真实 CPU 复验

**6/6 小检查通过，作者原函数与独立表达式的损失/学生梯度在固定容差内一致。范围仅是逐 anchor 的 BCDL 分类分量；没有完整 BCKD、MMCV reduction/head 接线或 Drone/LLVIP 性能复现。**

作者仓库固定到既有提交 `121c9aa272c9195c0f05cd12727bfe57636a5005`。已下载三个原始文件并逐字节写入/回读，见 [SOURCE_RECEIPT.json](SOURCE_RECEIPT.json)：

- [官方 kd_loss.py 本地原件](official/mmdet/models/losses/kd_loss.py) / [固定作者源](https://github.com/TinyTigerPan/BCKD/blob/121c9aa272c9195c0f05cd12727bfe57636a5005/mmdet/models/losses/kd_loss.py)
- [官方 GFL-R50 配置原件](official/configs/bckd/bckd_r50_gflv1_r101_fpn_coco_1x.py) / [固定作者配置](https://github.com/TinyTigerPan/BCKD/blob/121c9aa272c9195c0f05cd12727bfe57636a5005/configs/bckd/bckd_r50_gflv1_r101_fpn_coco_1x.py)
- [官方 ld_head.py 原件](official/mmdet/models/dense_heads/ld_head.py) / [固定作者 head](https://github.com/TinyTigerPan/BCKD/blob/121c9aa272c9195c0f05cd12727bfe57636a5005/mmdet/models/dense_heads/ld_head.py)

GitHub API 首次因限流403，随后用只读 `git ls-remote` 取得既有 main 提交 ID，再固定到该 ID 下载。urllib 原文件连接被重置后改用 curl；这些经历保留在来源回执，未把失败当下载成功。没有计算新文件 hash，Git 提交 ID 是读取作者既有标识；无 SSH/GPU/权重下载/全局环境安装。

## 复验内容与真实数值

[作者提取函数](extracted_novel_kd_loss.py) 逐行保留 `novel_kd_loss` 第13–38行，只去掉两层 decorator；执行时提供 torch / F 命名空间，不 import 或安装 MMCV/MMDetection。AST 比较确认函数体未改变。[独立实现](bcdl_independent.py) 用两个 softplus 写交叉熵，没有直接复用作者的 BCE 调用。

逐 anchor 算子为：`sum_c abs(q-p)^beta × BCEWithLogits(student_logit,q)`，其中 `q=sigmoid(teacher_logit).detach()`、`p=sigmoid(student_logit)`，默认 beta=1。

实测环境为本机 `D:/Anaconda/envs/KGJ_proj/python.exe` / Torch1.8.0+cu111，全部 tensor 在 CPU，CUDA 始终未初始化。权威结果为 [CPU_RECEIPT_attempt2.json](CPU_RECEIPT_attempt2.json)。

| 检查 | 实际结果 |
|---|---|
| 1、5、80类及空集合，各 FP64/FP32，共8个小形状 | 全部 loss/学生梯度通过；FP64 最大 loss差 `7.1054e-15`、梯度差 `2.2204e-16`；FP32 最大 loss差 `2.9802e-8`、梯度差 `1.1921e-7` |
| beta 与实际分类温度 | 默认与 beta1 exact；core无温度缩放，`NovelKDLoss.forward` 不读保存的 T/threshold。人为除10明显改变结果，不能因类构造函数默认 T=10 就这样实现 |
| 教师 detach | 默认 teacher gradient 为 None；固定非平凡例学生梯度 L2 `0.7136222332`；显式 detach_target=False 的对照有教师梯度 |
| 学生差值权重不能 detach | detach 权重后 forward 数值 exact相同，但学生梯度最大差 `0.1976733518`，梯度 L2从 `.7136222332` 变为 `.3545552545` |
| 加权 BCE 不能静默替为加权 Bernoulli KL | 学生梯度最大差 `0.1398220486`；差值精确符合 `abs(q-p) × H(q)` 的梯度（FP64容差内） |
| 同 logits / 错 shape | 同 logits时 loss及学生梯度 exact零；不匹配 shape被原函数拒绝 |

固定数值容差为 FP64 `atol=rtol=1e-12`，FP32 `atol=1e-6, rtol=1e-5`，未看结果后放宽。此处是数值一致，未把全部浮点结果声称为逐位一致。

为什么 BCE/KL 区别会进入梯度：普通 BCE 与 Bernoulli KL 相差教师熵，但本算子把该差值乘了依赖学生的 `abs(q-p)`。教师熵相对学生虽为常数，其乘积却不是，所以替 KL 会改变学习公式。

首次检查因本地 Python3.8 没有 `ast.unparse`，在生成提取 metadata 时失败，0项算子检查执行；已保留 [attempt1](CPU_RECEIPT_attempt1.json) 及 [当时检查脚本](attempt1_source_snapshot/check_operator_cpu.py)。唯一修正是用 `ast.get_source_segment` 记录原 decorator 文本；作者函数、独立公式、数据和容差未变。

## 下一步接入边界

本包已经给出可调用的 `bcdl_per_anchor(student_logits, teacher_logits, beta=1)`，返回 `[N]`，单类仍非零，适用于准备后续 dense 分类分量。但完整作者 head 的有效位置/label_weights/正样本分母尚未迁移：原 head 在该层有正样本时对 dense logits调用 KD，以 label_weights加权、num_total_samples归一；无正样本层蒸馏归零。这些不是本次去掉 weighted_loss decorator后的逐 anchor检查所能验收的内容。

固定作者配置另有系数4的 GIoU定位项；只接本包必须叫 `BCKD-BCDL classification-only / partial`。不能一边套本项目对象 E/K/质量门、一边称完整作者BCKD，也不能根据这次合成梯度非零推断IR有用、AP提升或跨数据集稳定性。

后续如决定接入，只需在新独立 adapter明确 dense位置、原 native分母/图平均与实际 B倍关系，再验一个真实batch的 frozen teacher/学生梯度和资源。这里不强行移植完整 head、不安装旧MMCV、不启动任何训练。

复跑只需本地Torch：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUDA_VISIBLE_DEVICES=''
& 'D:/Anaconda/envs/KGJ_proj/python.exe' '<本目录>/check_operator_cpu.py' --output '<本目录>/CPU_RECEIPT_new_attempt.json'
```

入口拒绝覆盖既有回执；`fetch_official.py` 固定上述版本并拒绝覆盖已冻结来源。

## 90 CPU 复跑已完成

根代理已在现有 `/mnt/dataX/ydf/projects/RGBT_campaign_90/environments/rgbir90/bin/python`（Torch `2.10.0+cu128`）执行同一小检查，回收 [CPU_RECEIPT_90_attempt1.json](CPU_RECEIPT_90_attempt1.json)：状态 `PASS_BCDL_CLASSIFICATION_OPERATOR_ONLY`，6/6通过，device为CPU。它补了目标环境的作者分类算子可执行证据，仍不涉及 GPU、MMCV完整 head/reduction、模型权重或两数据集 AP。本文作者仅本地读取该回执，没有另连服务器或再次执行。
