# YOLO OS-SSL 三种子 v2 方法契约

冻结时间：2026-08-22（Asia/Shanghai）  
方法 ID：`YOLO-OS-SSL-10K-3SEED-v2`  
Campaign：`YOLO_OSSSL_3SEED_V2`

## 冻结科学协议

- 模型固定为 SHA-256 锁定的 `yolo11n.pt`；不包含 AKD 或 YOLO11s。
- SSL arms 为 `sar_only`、`shuffled`、`paired`；每个 cell 恰好 10,000 optimizer updates，physical/global B64，无累积。
- SSL 使用 LARS（LR 0.8、momentum 0.9、weight decay `1e-6`），warmup 1,000 updates 后余弦到零；EMA 从0.99余弦升至1；增强与 v1 的 shared-geometry / independent-photometric 变换完全一致。
- SSL seeds 和 detector seeds 都固定为 `[0,42,123]`。非 Native detector 只能加载同一 `dataset + arm + seed` 的 SSL final；Native 无 SSL 依赖。
- shuffled donor 始终使用冻结的 seed42 derangement；seed0/123只改变模型、采样与增强随机性，绝不重建 donor map。
- detector arms 为 `native`、`sar_only`、`shuffled`、`paired`，physical/global B64，final epoch 是唯一 endpoint；禁止中间 AP/loss/P/R/F1、best checkpoint 与 early stopping。

| Dataset | Detector protocol | Endpoint |
|---|---|---|
| SpaceNet6 | E300, 640 | frozen test200 AP50 |
| SiXiang | E300, 512 | scene-clean val304 mAP50-95 |
| OGSOD | E400, 256 | development test3667 mAP50-95 |

## 历史 seed42 依赖

v1 的九个 seed42 final 可以复用，但仅当独立 v1→v2 compatibility receipt 同时证明：final bytes 拷贝相等、完整 v1 protocol fields 相等、目标 v2 数据 manifest 在移除主机路径后具有相同 pair/group/hash/donor 身份。收据不得读取科学 outcome。

## 调度和防火墙

- 仅 gp94 GPU4--7；GPU4 Native、GPU5 SAR-only、GPU6 shuffled、GPU7 paired。首波启动12个 seed42 detector cell。
- seed0/123只能在同卡一个 seed42进程退出且项目实测显存低于16 GiB后补位；项目目标显存为16--22 GiB，优先级为 SpaceNet6 SSL、SpaceNet6 ready detector、SiXiang SSL、OGSOD SSL、剩余 detector。
- 允许同卡多进程和与他人共享；不信号、不暂停、不修改他人进程。新增任务出现 OOM/Xid 时只停止新增任务，不改变 B64、epoch、seed 或 arm。
- scheduler 只记录 PID、GPU UUID、显存、文件大小、退出状态、数据与源码哈希；成功训练结束先做哈希绑定和有限性审计，四臂 seed42 group 全部通过前不读取该组 endpoint。

## 终态边界

seed42 四臂是每个数据集的初步结论；seed0/123不因结果取消。完整结论使用三组配对 seed 的逐 seed 差值、均值、标准差和方向一致性。该 v2 仍只代表 10k-update short-dose，不外推为论文300/500 epoch协议。
