# 部署与24更新canary入口（待root实际执行）

## 代码与日志身份

`code/` 上传至94新目录 `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_random_20260906/release_v1/`；**不能覆盖主线release_v2**。`source_delta.patch`、本计划和CPU验证也留在对应artifacts根目录；远端采用直接字节比较，不新增哈希记录。本地只读来源清单不作为远端验证脚本。

- CLI新开放 `--arm paired_random`；正常运行日志、completion/evaluation/run_evidence的arm均自动为paired_random。
- `method_id` 保留 `RGBIR-OBJECT-EVIDENCE-v1` 家族身份，不能单看method_id把随机对照误当P。
- 每个新run自动保存真实新训练器到implementation_snapshot；loss/loader/config仍与主线相同。
- 不复用输出目录；每个attempt独立。失败时保留所有产物，不能用exist_ok覆盖。

## 先做seed42 canary

先实时检查GPU空闲显存、任务数、当前项目卡数与内存；原OEv1占3卡时，用第4卡必须满足AGENTS放宽条件并登记。若与CCLKD同卡，两者实际峰值总和仍需留下≥2GB，且不得仅凭10GB预留推测峰值；依据已有canary实测与新canary守卫核验。

```bash
screen -dmS rgbir_oev1random_canary_s42 bash /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_random_20260906/release_v1/start_random_canary.sh 2 42 attempt1
```

脚本调用全项目guard，仅一张物理GPU，24次实际optimizer update；资源申请拒绝时退出，不建立无期限偷偷接续的队列。GPU2仅是当前root拟用分配，实际部署前必须重新核验。不能在已存在同名screen的情况下重复执行。

命令等价于：

```text
python train_object_evidence.py --config config_drone.yaml --output <new_run_root>/canary_paired_random_s42_attempt1 --arm paired_random --seed 42 --max-steps 24
```

通过后，在CPU运行：

```bash
CUDA_VISIBLE_DEVICES='' /mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_random_20260906/release_v1/validate_random_canary.py --paired-run /mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_v1_20260906/canary_paired_s42_attempt1 --random-run /mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_random_20260906/canary_paired_random_s42_attempt1 --gpu 2 --output /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_random_20260906/canary_verified_s42_attempt1.json
```

seed0/123的历史P canary分别在：

```text
/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_expand_20260906/canary_paired_s0_attempt1
/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_expand_20260906/canary_paired_s123_attempt1
```

校验器检查真实初始化/首batch逐张量相等、配置和输入权重路径/大小相同、loss/loader逐字节相同、训练器仅两行预期差异、共同batch文件序列/base/eligible/K/normalizer/nominal dose完全相等，同时确认至少有一个batch所选对象不同。AMP跳步可使完成24更新时batch总数不同，所以比较共同日志区间（至少24batch），不强行要求AMP行为或总batch数相同。

## 完整训练与终态

对应seed canary已通过且资源审计正常后，完整训练CLI：

```text
python train_object_evidence.py --config config_drone.yaml --output <new_run_root>/full_paired_random_s42_attempt1 --arm paired_random --seed 42
```

seed按预注册0/42/123各一次，E200直接来自原config。不设置max-steps，不修改超参，不复制canary模型作为完整训练初始模型。实际root队列应先检查对应`canary_verified_*.json`，失败即停；通过后才训练并调用原样复制的 `evaluate_object_evidence.py --config ... --run ...` 独立last/EMA评估。评估也必须过guard。

本子任务没有部署、GPU canary或完整训练；实际动作应由root单独追加回执，不能把本文件里的入口当成完成记录。
