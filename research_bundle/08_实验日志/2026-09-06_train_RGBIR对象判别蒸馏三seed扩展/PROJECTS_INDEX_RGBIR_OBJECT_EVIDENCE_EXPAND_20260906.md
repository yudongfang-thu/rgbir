# 94 RGBIR三seed扩展新增回执（2026-09-06）

本地记录：E:/SHARE/光sar/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/README.md。

新增94目录为RGBT_campaign/artifacts/rgbir_object_evidence_expand_20260906和RGBT_campaign/runs/rgbir_object_evidence_expand_20260906。

artifacts包括：release_v1（新调度脚本，原方法复用旧release_v2）、launcher_cpu_attempt1/2（7/8项控制流版本保留）、health_audit_snapshot1、endpoint_snapshot1/2（9/10项汇总器版本保留）、launch_verification_snapshot1、workers/（逐seed、stage、attempt隔离）、canary比较回执、cross_seed_realization.json/md及对应CPU采集源码。

runs包括四个canary，新full_weight0_s0_attempt1与full_paired_s123_attempt1已运行，其对应另一臂在各自同GPU队列中。固定GPU5/6；原GPU4队列完全保留，不改原代码、数据、checkpoint、results或receipt。当前全项目最多GPU4/5/6，符合新用户授权与工程AGENTS。

全部训练输出和脚本/日志在数据盘。原始产物不移动、不覆盖、不重命名；不生成摘要码或存放凭据。
