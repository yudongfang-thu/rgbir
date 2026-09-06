# 首轮执行验收与去留判断

> 2026-09-07 03:22：工程与真实C路径通过，L未达到几何准入；转入已授权C归因分支。本轮不是定位方法收益的阴性实验。

## 已实现与验证

- 独立 `rgbir_task_conditional_v1` 模块，旧OEv1五文件按94实际release逐字节归档。未修改旧运行身份或其源码文件。
- L固定R anchor、全部RGB GT唯一native归属、未clamp支撑、R/T类别与定位门、同anchor DFL KL；CGT保留同mask、E、分母及温度修正两bin目标。随机定位同E/K，空集可导零；CPU数学/选择测试16项通过。
- GT/teacher内容控制、shuffled额外内容、真实RGB-only C、坐标追踪、经验几何合同、64batch梯度校准工具、独立analyzer、统一lease启动器均有源码与检查。校准工具尚未用于生成正式λL。
- 真实CPU loader：Drone/LLVIP各3图、80框，原RGB/IR张量与标签、collate、RNG精确相同；独立坐标投影最大误差7.06e−5像素。错配loader另做两数据集真实6图/3seed检查，额外内容不推进paired RNG。
- 真实B32 criterion审计：old weight0↔new N，old paired↔new C，loss/native items/raw score与DFL梯度均bitwise相同，状态与RNG一致，教师/参考冻结。原C42端点可按此等价证据复用；这不扩大其单seed证据强度。

## 四条真实训练canary

| 路径 | 成功optimizer update | 初始AMP skip | C选中对象累计 | 实测NVML峰值MiB | 进程树RSS峰值MiB |
|---|---:|---:|---:|---:|---:|
| N（C权重0） | 24 | 6 | 3767（仅计算，不加权） | 6358 | 28744 |
| C | 24 | 6 | 3767 | 6358 | 28714 |
| C-shuffled | 24 | 6 | 3767 | 6656 | 29959 |
| C-same-modal | 24 | 6 | 2670 | 6370 | 28476 |

四个canary初始student tensor完全相同，RGB首批完全相同；N/C/shuffled的paired IR首批也完全相同，same-modal的strong图确实等于RGB。C/shuffled全部30批paired E、eligible/K、对象ID、分母和名义剂量完全相同。记录AMP skip并按24成功update验收，未用attempt数充数；原生EMA更新行为仍沿用原实现。

## 定位门槛与决定

未加几何核验时，D2选中/全部RGB GT：Drone train626/31931（1.96%），dev112/3084（3.63%）；LLVIP train211/5592（3.77%），dev131/643（20.37%）。这些是冻结R/T的letterbox诊断，不是正式学生学习效果。LLVIP dev的高比例不能代表训练可蒸量。

几何冻结300对名单，实际查看48对，只有LLVIP050001局部6静态点得到独立接受。其2个GT均在接受区域之外，实际geometry-qualified D2=0。其余来源未完成独立物理点合同，不把未看区域判为配准失败。**当前本版L不进入长训；λL、CL/CGT训练canary及E200仍未执行。** 没有放宽门、按AP调整或自动切换box loss。

按原计划“两个数据集几何/D2均未通过则继续C归因”的分支，以两条C内容对照seed42继续推进；首轮不额外扩展L矩阵，0/123对照等待当前证据收口。

## 部署、资源和保留失败

最终阶段代码 `94:RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907/release_v8`，所有release独立保留。初期失败包括standalone测试import、路径规范化后的fixture预期、启动器漏传lease路径，以及criterion审计误把native dict items当tensor；均修复后新建attempt，未修补原失败证据，也未伪称连续训练。

原N/C/random五个训练继续。新增C-shuffled42在GPU4与原random123共享卡；C-same-modal42在screen排队，guard指出RSS预约将达240GiB而暂拒新任务。当前仍4物理GPU，空卡1/3/7，援引AGENTS四卡放宽；每卡当前最多2CUDA任务，全卡监控≥2GiB余量，项目VRAM<70%、RSS≤300GB。训练与独立评价均由各自同一stage队列负责。

远端guard常量允许4卡，老lease JSON中policy.max_active_gpus=3是历史元数据；实际源码快照见阶段采集的external_runtime，不据旧字段修改现行guard。目录/系统盘/凭据纪律保持。

## 尚未完成的整轮冲刺目标

本阶段不能宣称整个冲刺已完成：缺L几何覆盖和正式比较，旧C−N三seed未齐，C内容对照尚在训练/排队，最终三seed四臂及封存test评价未完成。已有raw/gzip/图表、单位与配对检查、accepted analyzer、两次状态快照均已落盘；FGD/LD仍在后续外部基线准备线，未占首轮训练资源。
