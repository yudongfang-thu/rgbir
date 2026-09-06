# OEv1 C 内容归因（2026-09-07）

> 定位分支未满足几何准入，执行用户计划预设的 C 归因分支：冻结 seed42 的 C-shuffled 与 C-same-modal 两个 E200 对照，完成真实 canary 后由统一 lease 启动/排队。

## 目的
检验现有 C 判别蒸馏是否依赖正确配对 IR 内容，并与真实 RGB-only 蒸馏比较。本轮不新增第三、第四个正式长训，不扩展 L 矩阵；原 N/C/random 三 seed 正常收尾。

## 冻结设置
- DroneVehicle，student generic YOLO11n，E200、640、batch32/nbs64、workers4、SGD、AMP及增强完全继承原 C，student seed42；固定 last/EMA、完整1469图 development val独立评价，无中途AP早停，无test。
- C-shuffled：T/R仍原IR/RGB seed42；固定train-only seed20260907无自配IR错排，保留paired的E、对象区域、门、K、分母和0.1系数，仅换内容目标。同几何/RNG replay，错配GT不用于选择。它同时破坏实例和空间，只作压力对照。
- C-same-modal：独立RGB weight0 seed0教师，R仍原RGB seed42；输入、标签、门、目标均RGB-only。RGB identity mapping和RGB YAML单独冻结，不继承IR mask。
- 两个控制均 λL=0。该轮不能回答CL−CGT，也不替代最终三seed四臂。

## 启动前证据
N/C真实canary各24成功update；C-shuffled/C-same-modal同样达到24成功update，均记录6次初始AMP skip。新旧C/N真实B32损失/items/raw梯度精确一致；CPU真实loader/RNG等价通过。四个短测初始化与RGB首批一致；shuffled 30批paired E/K/分母和对象ID逐项相同；RGB-only的strong图与RGB相同。

## 资源与产物
沿用项目global lease；最多4物理卡须剩≥2空卡，每卡最多2个当前工程CUDA任务，项目单卡显存<70%，全卡≥2GiB空余，总RSS≤300GB。按已测峰值申请8300MiB GPU/32768MiB RSS；两个独立screen，每臂训练后承担自己的独立评价，资源不足排队。

94代码与验收：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907/`。
94正式输出：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_task_conditional_c_attribution_20260907/`。
配置、启动与资源回执见同目录阶段快照；不能把已排队写成正在训练。

## 局限与下一步
暂无新正式AP。先收齐当前对照和旧三seed端点，再报告严格配对及是否值得补0/123；不凭单seed声称因果或稳定增益。定位分支目前是证据不足而未准入，不是实验已证明无效。
