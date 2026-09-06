# OEv1 三seed固定端点汇总

> 状态：pending；完整端点 0/6。本报告不读取训练CSV中的占位指标。

UTC采集时间：2026-09-05T23:40:46.746767+00:00。主量为同seed paired−weight0 mAP50–95百分点，固定E200 last/EMA与完整开发val。

| student seed | arm | 状态 | mAP50–95 (%) |
|---|---|---|---:|
| 0 | paired | pending_not_started | — |
| 0 | weight0 | pending_not_started | — |
| 42 | paired | pending_training | — |
| 42 | weight0 | pending_not_started | — |
| 123 | paired | pending_not_started | — |
| 123 | weight0 | pending_not_started | — |

## 配对主量

| seed | paired−weight0 (pp) | 方向 |
|---|---:|---|
| 0 | — | 待完成 |
| 42 | — | 待完成 |
| 123 | — | 待完成 |

三seed汇总未产生；缺失或无效端点不能用CSV、历史baseline或单seed指标补位。

## 解释边界

教师与参考模型固定seed42，本轮只覆盖学生训练随机性。单run的exploratory标记保留；三seed汇总也不自动变成confirmatory证据。paired−weight0仅支持整套干预净效果，尚缺shuffled/same-modal四臂归因，不能宣称跨模态独特收益或避免负迁移。三点不输出p值。

## 原始证据路径

- seed0 paired：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_expand_20260906/full_paired_s0_attempt1`
- seed0 weight0：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_expand_20260906/full_weight0_s0_attempt1`
- seed42 paired：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_v1_20260906/full_paired_s42_attempt1`
- seed42 weight0：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1`
- seed123 paired：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_expand_20260906/full_paired_s123_attempt1`
- seed123 weight0：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_expand_20260906/full_weight0_s123_attempt1`
