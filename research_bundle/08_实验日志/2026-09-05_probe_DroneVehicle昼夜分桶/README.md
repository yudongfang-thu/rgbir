> **⚠️ 勘误（2026-09-05 晚，依据全项目复盘审计 §5.3）**：分桶条件差异观察保留（高亮桶 RGB−IR +2.23 / 低亮桶 IR−RGB +11.25 AP50），但：
> ① 亮度中位数是代理、非官方光照标注；② 师生各自用本模态标签评估，未在同一目标集合上量化错误互补；③ native/dist recipe 不一致，"蒸馏有害"不可判读；④ **"固定教师必然在一侧负迁移"表述过强**，降级为"条件与教师相对效用可能相关，待匹配 N/L 分桶差值与门控消融验证"。

# DroneVehicle 昼夜分桶评估（P3 探针，2026-09-05）

> **一句话结论**：教师优势随光照翻转被量化——白天 RGB native 反超 IR 教师 +2.2，夜间 IR 教师反超 RGB native **+11.2** AP50；固定教师蒸馏必然在一侧承受负迁移压力。这是"按模态特点设计防负迁移机制"的诊断基础。
> 服务器产物：`94:RGBT_campaign/artifacts/p3_daynight_20260905/{summary.json, per_image_luminance.csv}`（本目录存 summary.json 副本 + 脚本）。

## 目的
为教师路由/防负迁移机制设计提供分桶证据：IR 教师与 RGB 学生的优势是否随光照条件翻转；现有蒸馏学生在两桶上的表现。

## 设置
- 数据：DroneVehicle val 1,469 对（processed hbb_v1）；按 RGB 图平均亮度**中位数（78.28）**分为 day 735 / night 734（亮度是代理，dusk 样本被劈开；官方无逐图光照标注）
- 模型（均 y11n，formal_native 同协议 b32a2）：T_ir = infrared_seed42_native_b32a2（IR 输入）；S_native_rgb = rgb_seed42_native_b32a2；S_dist_rgb = adapted_v2 dronevehicle_seed42（**注意：dist 用的是 b32 e200 adapted 协议，与 native b32a2 不同 recipe，两者差异≠纯蒸馏效应**）
- 评估：ultralytics val，imgsz 640，COCO mAP；桶目录为硬链接副本（不动原数据）

## 结果（mAP50 / mAP50-95）

| 模型 | day | night | full |
|---|---|---|---|
| T_ir | 78.08 / 57.39 | **82.42 / 60.95** | 80.84 / 59.63 |
| S_native_rgb | **80.31 / 62.80** | 71.18 / 45.10 | 76.01 / 53.82 |
| S_dist_rgb | 79.71 / 62.17 | 69.42 / 44.32 | 75.17 / 53.26 |

- 白天：S_native > T_ir（+2.23 AP50）；夜间：T_ir > S_native（**+11.24**）
- S_dist 与 S_native 差异受协议混杂（b32 e200 adapted vs b32a2 native），**不能读作"蒸馏有害"**；协议匹配的 no-KD 对照（weight0/native 臂）待四臂矩阵补齐
- 勘误：首轮误用 `native_rgb_s42_b64_e200`（实为 VEDAI 模型，得 0.09），已换正确 checkpoint 重评并修正

## 结论
1. 模态优势交叉是真实且量级可观的（夜间 IR +11）；"固定教师"在夜间把学生天花板压在 RGB 教师水平以下——**负迁移的来源被定位到条件子集**
2. 机制设计的靶点：按条件/按样本选择教师知识（门控），或在教师不可信区域抑制蒸馏损失；探针的显著性相关（P2 报告）可作在线门控信号
3. IR 教师夜间 82.4 甚至高于 RGB 学生白天 79.7——IR 通道的信息量在这个数据集上总体更强

## 局限与下一步
- 亮度中位数分桶是代理；如需更干净的条件定义，可按数据集官方 dusk/day/night 比例核对或做三分桶
- 协议匹配的 no-KD 对照待跑（四臂矩阵的 native 臂）
- 下一步：VEDAI y11 配对探针（教师 IR paper80 + 学生 RGB fold01，8 类一致已核对）；随后方法预注册
