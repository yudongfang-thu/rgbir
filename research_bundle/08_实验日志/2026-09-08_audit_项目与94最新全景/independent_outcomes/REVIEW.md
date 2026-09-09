# Drone E200 主线独立结果审查

**结论：N/C0/random 的九个固定 E200 last/EMA 独立开发集端点可追溯，C0 的 mAP 相对 N 与 random 在三个学生 seed 上均为正；C1 与 shuffled/same-modal 尚无完整端点，跨模态内容归因和 C1 效用未闭合。整体审计为 WARN；旧训练 CSV 第200轮的指标列存在确定的字段错位，不能作为 AP 证据。**

## 目的与身份

独立检查94原始小产物，未先采用既有报告的结论。审阅者 `/root/independent_outcomes`，可见模型身份 `unavailable`；不声称跨模型评审。主采样时间 2026-09-08 14:45:39 +08:00，补取执行快照/源码 14:48:14；终态文件与源码远端复核时间见下。全部远端访问只读，没有启停任务、加载或下载权重、模型前向、重新评价或改变原结果。

## 原始端点与独立重算

指标原值为0–1，下表统一乘100；均值±SD使用三个学生seed和样本标准差（ddof=1），不是置信区间。AP基于真实RGB标注（`real_gt`），独立评价指训练后的独立程序入口，并不表示新数据集。

| 臂 | 学生seed | AP50 | AP75 | mAP50–95 |
|---|---:|---:|---:|---:|
| N / weight0 | 0 | 76.992515 | 63.932917 | 54.346185 |
| N / weight0 | 42 | 77.179654 | 64.166566 | 54.513608 |
| N / weight0 | 123 | 76.927417 | 63.689478 | 54.407436 |
| C0 / paired | 0 | 76.986788 | 64.166542 | 54.772186 |
| C0 / paired | 42 | 77.069620 | 63.839323 | 54.658162 |
| C0 / paired | 123 | 77.304557 | 64.214774 | 54.636847 |
| random | 0 | 76.896411 | 63.957860 | 54.630857 |
| random | 42 | 76.478419 | 63.766126 | 54.492157 |
| random | 123 | 76.940931 | 63.742509 | 54.419367 |

| 臂 | AP50 mean±sample SD | AP75 mean±sample SD | mAP mean±sample SD |
|---|---:|---:|---:|
| N | 77.033195 ± 0.130946 | 63.929654 ± 0.238561 | 54.422410 ± 0.084710 |
| C0 | 77.120322 ± 0.164840 | 64.073546 ± 0.204272 | 54.689065 ± 0.072770 |
| random | 76.771920 ± 0.255152 | 63.822165 ± 0.118107 | 54.514127 ± 0.107443 |

配对差按同seed先作差再计算SD，顺序为0/42/123：

| 比较 | mAP逐seed差（pp） | mean±sample SD（pp） | 正方向 |
|---|---|---:|---:|
| C0−N | +0.426001 / +0.144554 / +0.229411 | +0.266655 ± 0.144373 | 3/3 |
| C0−random | +0.141330 / +0.166006 / +0.217480 | +0.174939 ± 0.038853 | 3/3 |
| random−N | +0.284671 / −0.021452 / +0.011931 | +0.091717 ± 0.167935 | 2/3 |

C0−N的AP50只有1/3正（均值+0.087126 pp），AP75为2/3正（均值+0.143892 pp）；不能把mAP的3/3正说成所有指标均提高。C0−random的AP50、AP75也是3/3正。原始precision/recall由原生指标函数在平滑平均F1最大点取值，模型各自选择的工作点可能不同；不能直接当固定score阈值的检出率或对象数。逐项完整数值在 `recomputed_results.json`。

九个run都有200行CSV、training_completed/last_epoch=200的完成回执、独立 `evaluation_val.json`、COMPLETED评价回执，且评价小文件与receipt内metric snapshot逐字节一致。N/C0六个额外posthoc补评的五项宏指标与原端点逐项exact，五类AP均值与宏AP重新闭合。这里只重算已保存数值，没有重新运行检测模型或从全部预测重算AP。

## 实际协议与有效配置

共同设置为 DroneVehicle HBB RGB学生、五类、YOLO11n初始化、640输入、E200、batch32/nbs64、workers4、SGD，lr0=.01、lrf=.01、momentum=.937、weight_decay=.0005、warmup3轮、AMP与deterministic开启。mosaic/mixup/cutmix为0，translate=.1、scale=.5、水平翻转=.5，HSV增强为0。开发集为1469张RGB图；C1冻结配置记录训练集17990张。测试集未被这些评价入口使用；receipt仍标官方test为UNVERIFIED_SEALED，审计不认证整个项目历史测试暴露。

九端点均使用相同历史IR seed42教师与历史RGB seed42参考；三seed变化的是学生训练随机种子，不是教师/参考的三次独立训练。两者路径为 `runs/rgbt_p3_causal_v1/formal_native/dronevehicle/{infrared,rgb}_seed42_native_b32a2/weights/last.pt`。

静态 `protocol_config.yaml` 九份完全相同且seed字段都是42；训练CLI的 `--seed 0/42/123`在执行时覆盖此值。已逐一检查launch_manifest、args.yaml、完成/评价回执的真实seed一致，不能仅按protocol_config认定九个run都跑seed42。N的配置保留kd_weight=.1，但执行源码对weight0施加字面0，并保留相同辅助路径；不能把配置中的.1误当N实际蒸馏系数。

C0为对象级标量相对证据的SmoothL1（λ=.1）；random在同一base集合随机选取相同名义K，K来自原eligible数与rho=.5。其抽样人口是整个base，并非只在原eligible内打乱排名，因此C0−random检验的是教师正确性/质量选择的整体组合，不能单独归因为一个排序步骤，更不等同paired−shuffled内容归因。源码依据见random的 `implementation_snapshot/object_evidence_loss.py:21`、`:209`、`:297`。

C1是独立分类相对logit Bernoulli KL（分类系数0.09227393550836771；同类/异类权重设置见配置），localization_coefficient=0。它的损失与系数改变，不能称为旧C0的同一训练版本；当前仍没有C1完整结果。shuffle/same-modal是旧C0内容控制，并不是C1内容控制。

独立端点为固定预算 `weights/last.pt` / `fixed_budget_last_ema`，不是挑最佳开发集checkpoint。已核执行评价快照调用model.val与completion检查，并核原生trainer保存EMA路径。六个posthoc合同实际参数一致：conf=.001、NMS IoU=.7、max_det=300、rect=true、half=false、augment=false、batch32。原生AP50/75分别取IoU第0/5列，mAP取所有类别×十个IoU阈值均值；这是类宏平均，不是按GT数量加权的对象准确率。

## 未完成臂的正确状态口径

首次快照的CSV已完成轮数（progress可能指正在训练的下一轮）：

| 臂 | seed | CSV已完成/计划 | 独立端点 |
|---|---:|---:|---|
| C1 | 0 | 72/200 | 无 |
| C1 | 42 | 75/200 | 无 |
| C1 | 123 | 73/200 | 无 |
| C0 shuffled | 42 | 144/200 | 无 |
| C0 same-modal | 42 | 119/200 | 无 |

这些run的progress记录仍为运行态，不能用当前loss或CSV占位AP对完整E200排序。进程健康与资源使用由全景审计另一分工复核；本审查不以缺少completion推断失败。采集范围内shuffled/same-modal只存在seed42正式目录，即使这两个端点后来完成，也尚不构成三seed四臂归因。

## 已验证的字段缺陷

旧N/C0/random的CSV表头15列，前199轮15列中的AP是禁用逐轮验证时的0占位；**第200轮只有8列**，native三项loss之后直接写入三项lr=.0001495，CSV解析会把它们错误安到precision/recall/mAP50，mAP50–95等后续字段为缺失。依据是原始CSV第201行，以及旧 `train_object_evidence.py:234` 的 `validate() -> ({},0.0)`、原生 `trainer.py:596`/`:604`/`:919` 的动态指标写入。此为真实的CSV字段缺陷（该文件作AP来源时FAIL），但独立evaluation_val.json及其执行回执不依赖这三个错误槽位，故不据此撤销九个独立AP端点。

C1与内容控制的当前AP列都是0占位，训练args的val=false；这些0不是模型测得AP=0。旧配置seed42、N配置kd_weight=.1同样需要结合执行override解释。本次保留原文件，不修写旧结果。

## Hash、覆盖与局限

九端点评价脚本快照、RGB数据YAML、1469图roster分别全同：

- eval script：`a47b16944161766ee6362eec20781e6e507a08d0f3a4f299b04644cb6333f5b7`
- RGB data YAML：`96f79db23ce400fbba9a1d60fbb3c99227f774a19c80935dd5aa8bac8fb96ec5`
- val roster：`22c75d29a6717864e768afa47851ae62dc4be1c64647088630672834e647d435`

以上是脚本/配置/路径名单的hash，不是图像像素、标签内容或权重hash。没有读取、下载或hash任何权重，也没有遍历hash所有训练/开发图像或标签；权重身份依赖既有绝对路径、launch文件大小与补评stat/执行回执，不能升级为密码学确认权重未变。当前环境原生源码已收集，但不是九次历史安装环境的全量快照。训练侧输入顺序完全一致、全部增强、有效优化步数完全相等及完整outcome-blind历史均不在本次逐步复演覆盖内。

共保留521个去重小文件；`source_manifest.json`与`source_manifest_supplement.json`记录远端绝对路径、本地副本、字节数、mtime及SHA256。2026-09-08T14:50:52.122172+08:00再次远端检查518条（含当前原始评价程序对执行快照），全部匹配；12个运行中progress/CSV/state允许前进并保留首次快照。全部本地副本在生成本报告前重hash通过。

仅支持“固定IR/RGB参考、此Drone开发集协议下C0相对N和随机选择有小幅三seed mAP方向一致优势”。尚不支持独立跨模态内容净效用、C1优于C0、定位蒸馏有效、外部数据集泛化或paper-ready完整四臂结论。对象错误拆解summary已保存供追溯，但本次未从完整预测重建对象匹配，不能把该存档当作新独立对象级验收。

## 复算入口与产物

`recompute.py`重算九端点及配对差，`check_provenance.py`核seed/评价/逐类闭合，`recheck_remote.py`只读远端核小文件hash；首次收集脚本拒绝覆盖原快照。入口使用本机 `D:/Anaconda/envs/KGJ_proj/python.exe`。`REVIEW.json`保存检查级判定，`trace/`保存委派prompt、响应和实际可见审阅元数据。README索引由root统一更新，本分工未修改全局索引。
