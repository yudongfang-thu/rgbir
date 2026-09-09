# LADD Reload / Continued-Training 混杂问题记录

日期：2026-06-23

> 2026-06-26 归档说明：本文件已经归入 reload/resume/transfer 混杂证据体系。它只能用于解释为什么旧 LADD/reload 结果不能作为当前主线正结果；当前主线证据只看 no-mosaic + YOLO-init + same-pipeline det-only control。归档索引见 `docs/experiments/archive_reload_confound_20260626/README_CN.md`。
>
> 2026-06-30 追加说明：新的 YOLO11n baseline 收敛审计进一步强化了本文件的判断。旧 800ep SAR baseline best AP `0.55654` 已被 1600ep detector-only best `0.58973` 明显刷新；从 1600ep best 低学习率 no-warmup reload 还能到 `0.59575`。因此旧 LADD/singleproj 相对 `0.556` 的正向结果需要全部重判。综合报告见 `docs/experiments/current_result_synthesis_20260630/RESULT_SYNTHESIS_20260630_CN.md`。

## 结论先行

当前 OGSOD YOLO11n 证据显示：LADD 主方法相对原始 SAR baseline 的提升，存在严重的 `baseline reload / continued training` 混杂。也就是说，主方法提升可能主要来自“加载已训练 SAR detector 后重新开启优化器和 cosine schedule 继续训练”，而不是来自 RGB teacher、decomposition 或可达性蒸馏本身。

在该问题被排除前，不能再把 `LADD vs original SAR baseline` 直接作为主 claim。更公平的比较应改为：

```text
LADD / Dynamic / Probe-A  vs  reload-or-warm-start detector-only control
```

2026-06-30 后的更强版本是：

```text
LADD / Dynamic / Singleproj  vs  scratch-1600 detector-only  and  reload-low-lr detector-only control
```

## 触发该问题的观察

原 LADD 主线 B 阶段会加载已收敛 SAR baseline checkpoint 作为 student 起点，然后重新进入接近完整的训练 schedule。这个流程同时包含两个因素：

1. 方法因素：RGB teacher、feature decomposition、KD、reachability loss。
2. 训练因素：已收敛 SAR detector reload 后，优化器和 cosine LR schedule 重新开始，并继续训练很多 epoch。

后续补充的 reload control 说明，第二个因素本身就足以解释大部分甚至全部提升。

## 关键证据

主证据包：

- `debug/method_failure_reload_control_20260623/`
- `debug/baseline_reload_n_protocols_20260623/`
- `debug/baseline_reload_vs_original_20260623/`
- `debug/no_reload_warm100_20260623/`

### 1. Reload SAR baseline 已接近或超过 LADD 主线

`debug/method_failure_reload_control_20260623/README.md` 中的当前快照显示：

| protocol | comparison | 观察 |
|---|---|---|
| mosaic100_close100 | reload - original | `+0.02429` mAP50-95 |
| mosaic100_close100 | LADD - original | `+0.02750` mAP50-95 |
| mosaic100_close100 | reload explains LADD gain | `88.3%` |
| nomosaic | reload - original | `+0.02328` mAP50-95 |
| nomosaic | LADD - original | `+0.02008` mAP50-95 |
| nomosaic | reload explains LADD gain | `115.9%` |

这意味着：在 nomosaic 协议下，reload baseline 当前已经超过 LADD 主线；在 mosaic100 协议下，reload baseline 已解释绝大多数 LADD gain。

注意：部分 reload run 在生成记录时仍在运行，因此最终数值可能略有变化；但“reload 本身足以制造接近主方法的提升”这一风险已经成立。

### 2. YOLO-init LADD 与普通 SAR baseline 几乎重合

为移除 detector reload，我们启动了 no-mosaic B-stage 诊断：

| run | detector init | B mode |
|---|---|---|
| `yolo_probeA` | `yolo11n.pt` | Probe-A/mainline |
| `yolo_dynamic` | `yolo11n.pt` | Dynamic |
| `reload_detonly` | warm100 detector | det-only continued training |
| `warm100_probeA` | warm100 detector | Probe-A/mainline |
| `warm100_dynamic` | warm100 detector | Dynamic |

对应记录在：

- `debug/no_reload_warm100_20260623/CURRENT_ACTIONS_20260623.md`
- `debug/no_reload_warm100_20260623/figures/nomosaic_b5_existing_cache_vs_sar_baseline_20260623.png`
- `debug/no_reload_warm100_20260623/figures/yolo_init_loss_vs_sar_baseline_20260623.png`

当前 YOLO-init loss 对比显示：`YOLO-init + Probe-A` 和 `YOLO-init + Dynamic` 的 `train/box_loss`、`train/cls_loss`、`train/dfl_loss`、`val` 检测 loss 以及 mAP 轨迹基本贴着普通 SAR baseline 走。虽然 CSV 中存在 `train/kd_loss`、`train/reach_match_loss`、`train/reach_rank_loss`，但这些额外项没有把检测优化轨迹拉出 baseline。

当前快照：

| run | epoch | train det loss | val det loss | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| SAR baseline | 43 | 4.62696 | 4.20880 | 0.51075 | 0.27180 |
| YOLO-init + Probe-A | 43 | 4.56728 | 4.19480 | 0.51538 | 0.27731 |
| YOLO-init + Dynamic | 41 | 4.66515 | 4.30991 | 0.50004 | 0.25769 |

初步解释：当 B 阶段 detector 不加载已训练 SAR detector 时，当前 LADD loss 更像弱正则，没有形成稳定的跨模态收益。

### 3. Warm100 / reload 仍可能主要体现 detector continuation

当前 warm100 诊断中，`reload_detonly`、`warm100_probeA`、`warm100_dynamic` 都从同一个 warm100 detector 出发。早期结果显示 `reload_detonly` 不弱于甚至略领先 LADD 变体。这说明即使不用完整 800-epoch SAR baseline，只要 detector 已经 warm-start，后续提升仍可能来自 detector 继续训练，而不是 LADD 结构。

重要 caveat：当前 `warm100` 和 A1 cache 复用了已有 mosaic cache，而 B-stage 是 nomosaic；因此这组只作为诊断，不作为正式 paper 结论。正式结论需要同协议 cache 或完整重跑。

### 4. Sixiang 旧主线 reload 对照已补跑

为判断旧 Sixiang 主线是否也存在 reload / continued-training 混杂，已在 90 服务器 `/mnt/dataY/ydf/projects/LADD` 上补充 YOLO11s-OBB 对照。旧主线参考锚点：

| run | mAP50-95 |
|---|---:|
| SAR B0S baseline | `0.55304` |
| RGB B0R teacher | `0.67333` |
| split iterative P1 / P2 / P3 / P4 | `0.58258 / 0.58502 / 0.59157 / 0.59089` |
| split P3 3-seed mean | `0.58793` |
| independent single-proj P3 mean | `0.58644` |
| raw-backbone KD mean | `0.56552` |

当前补跑的两个 detector-only reload control（2026-06-23 22:43 CST 早期快照）：

| control | GPU | status snapshot | path |
|---|---:|---|---|
| LADD trainer shell, det-only, B0S reload | 4 | running, epoch 11/300, current `0.50282`, best-so-far `0.54387`; initial pre-train val `0.55288` | `/mnt/dataY/ydf/projects/LADD/runs/reload_controls_20260623/sixiang_yolo11s_b0s_reload_detonly_laddtrainer_e300_b32_s0_gpu4` |
| native YOLO OBB trainer, B0S reload | 2 | running, epoch 4/300, current `0.49511`, best-so-far `0.54383` | `/mnt/dataY/ydf/projects/LADD/runs/reload_controls_20260623/sixiang_yolo11s_b0s_reload_detonly_yolotrainer_e300_b32_s0_gpu2` |

两条都从 `checkpoints/baselines/sar_yolo11s_obb_clean_best.pt` 启动，使用 Sixiang SAR OBB 数据、`imgsz=512`、`epochs=300`、`batch=32`、`seed=0`、`deterministic=True`、`mosaic=1.0`、`close_mosaic=10`、HSV/erasing 关闭。第一条经过 `validate-before-train` 验证，初始 `mAP50-95=0.55288`，与旧 B0S `0.55304` 基本重合，说明 checkpoint reload 正确。

2026-06-23 23:08 CST 更新：

| control | status snapshot |
|---|---|
| LADD trainer shell, det-only, B0S reload | running, epoch 72/300, current `0.52326`, best-so-far `0.54492` |
| native YOLO OBB trainer, B0S reload | running, epoch 81/300, current `0.54013`, best-so-far `0.54666` |

目前两个 reload control 仍未接近旧 P1 `0.58258`，但 native YOLO reload 已从早期低点恢复到 `0.54` 量级，需要继续观察到 close-mosaic 后段或完整 300 epoch。

同时已启动旧 Sixiang iterative 方案复跑，用于直接对照旧主线是否可复现：

```text
/mnt/dataY/ydf/projects/LADD/runs/sixiang_oldscheme_controls_20260623/iter_oldsplit_fromP1_20260623_P2_a1
```

启动方式：从历史 P1 checkpoint `sar_tskd_tuaux_P1_iterReach1_c_skipb_minimal_role_split_e150_b32_gpu6/weights/best.pt` 继续，用旧脚本 `scripts/experiments/exp_iter_method.sh` 启动 P2。2026-06-23 23:08 CST 快照：A1 running，epoch 7/50，当前 `mAP50-95=0.58258`，与 P1 保持一致；这是预期现象，因为 A1 使用 `det_loss_scale=0.0`。

判定规则：

1. 若任一 detector-only reload 最终接近 P1 `0.58258`，旧 Sixiang P1 gain 就可能被 continued training 大幅解释。
2. 若 detector-only reload 进一步接近 P3 `0.59157` 或 P3 mean `0.58793`，旧 iterative 主线的核心证据也需要重判。
3. 若两条 reload 长期停留在 `0.553` 附近或明显低于 `0.565`，旧 Sixiang LADD 结果相对 OGSOD 当前主线更可能保留独立方法信号，但仍需配套同 seed / 同 schedule 的完整对照。

## 对论文主张的影响

在 reload confound 被排除前：

1. 不应声称当前 LADD 主线在 OGSOD 上相对 SAR baseline 有明确有效提升。
2. 不应把 `original SAR baseline` 作为唯一 fair baseline。
3. 任何主方法曲线必须同时报告 detector-only reload/warm-start control。
4. no-mosaic 与 mosaic100 的曲线必须按同协议、同容量、同 seed 分组画；reload 曲线必须与原 baseline 重叠画，不能接到 baseline 后面制造“累计训练”错觉。
5. YOLO-init / no-reload 结果目前没有显示出独立收益，因此不能作为已解决方案。

## 当前实验判定规则

后续若要重新建立主方法有效性，至少需要满足：

1. `Probe-A` 或 `Dynamic` 在同初始化、同 schedule、同协议下显著超过 `detector-only continued training`。
2. YOLO-init 或其他 no-reload 方案的检测曲线要明显偏离并优于普通 SAR baseline。
3. `train/kd_loss`、`reach` 等 LADD loss 的变化需要能解释 mAP 改善，而不是只在日志中存在但不影响 detector 优化轨迹。
4. 若采用 warm100，需要确保 warm100、A1 cache、B-stage 均为同协议，或者明确标注为诊断。

## 推荐下一步

1. 继续观察当前五条 no-mosaic B-stage 诊断到至少 100-200 个 B epoch。
2. 若资源允许，补同协议 no-mosaic warm100 detector 和 A1 cache，避免 mosaic cache 混入。
3. 设计不依赖 trained SAR detector reload 的主方法，例如：
   - YOLO-init + staged B training；
   - 先 100 epoch detector supervised warmup，再加载 A1 cache 进入 KD；
   - 限制 teacher/decomposition 梯度强度，避免它被 detector loss 淹没；
   - adapter-only 或 frozen-detector 诊断，确认 LADD loss 是否真的提供可迁移信息。
4. 论文层面暂停当前 LADD 主 claim，转为 reload-confound audit，直到新方案通过 detector-only control。
