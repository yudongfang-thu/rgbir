# 5. 复现与接管指南

## 5.1 接管第一原则

先确认“现在进行到哪”，再运行任何命令。权威顺序：

```text
AGENTS.md
→ handoff.md 顶部最新 reset 段
→ refine-logs/research_reset_20260722_v2/README.md
→ registry/campaigns.csv + registry/claims.csv
→ 对应 campaign freeze/primary artifact
→ L20 queue/raw evidence
```

`handoff.md` 包含大量追加历史，不能只看末尾或 grep 一个旧结论。2026-07-22 reset-v2 明确 supersede 旧 synthesis；新 primary evidence 仍可修订 reset。

## 5.2 本地环境

本地 macOS 实测 Python 3.12.4、torch 2.5.0 CPU、PIL 可用、无 CUDA、无 ultralytics。用途：源码审计、合成测试、validator/analyzer 测试、文档与 registry；不运行正式训练。

基本检查：

```bash
cd /Users/yudongfang/Desktop/光sar/ogsod400_clean_protocol
python3 -c 'import torch, PIL, yaml, numpy; print(torch.__version__, torch.cuda.is_available())'
python3 -m pytest tests/ -q
python3 -m pytest method/tests/ -q
python3 tools/validate_research_claims.py
```

reset 前记录的本地通过基线：root `tests/` 302 passed，`method/tests/` 7 passed。测试集可能随后增加，接管时以当前实际结果为准。缺 ultralytics 导致的训练入口错误不能通过篡改 runtime 来“修”成本地可跑。

独立方法包例子：

```bash
cd methods/sw_arcs_p0_v5
python3 -m pytest
python3 -m mypy
```

训练 wrapper 只允许 DRY_RUN：

```bash
cd /Users/yudongfang/Desktop/光sar/ogsod400_clean_protocol
DRY_RUN=1 scripts/run_clean_baseline.sh sar 0 0
DRY_RUN=1 scripts/run_clean_comparison.sh fgd 0 0
DRY_RUN=1 scripts/run_clean_lcsr.sh 0 0 main
```

## 5.3 L20 连接与路径

本机已有连接 helper：

```bash
bash /Users/yudongfang/.codex/skills/l20-ssh/scripts/connect_l20.sh 'hostname; date; uptime'
```

远程关键路径：

```text
/private/projects/ogsod400_clean_protocol
/private/projects/ogsod400_clean_protocol/logs/sixiang_anchor_residual_v1_20260722_b2/
/private/results
```

不要在文档、聊天或压缩包中复制 SSH 私钥、token 或未脱敏 trace。

## 5.4 当前 B2 的安全监控

快照时间 2026-07-23 00:48 CST：10 completed、2 running、9 pending；active 为 U1_s42 与 U2_s123。该状态会变化，接管后重新读取。

允许读取：

- queue status/attempt；
- process PID/elapsed；
- GPU memory/utilization/temperature；
- `results.csv`、diagnostics 的**行数**，不读值；
- `args.yaml`/`last.pt`/receipt 是否存在；
- OOM/Traceback/RuntimeError/Xid/nonfinite 关键字；
- 文件 mtime 与 output collision。

禁止读取：

- AP/mAP/P/R/F1/loss 数值；
- arm 排名、曲线形状、best epoch；
- 任何基于 outcome 的提前停止、retry、调参或 queue 重排。

只读状态示例：

```bash
bash /Users/yudongfang/.codex/skills/l20-ssh/scripts/connect_l20.sh \
  "python3 - <<'PY'
import json
from collections import Counter
p='/private/projects/ogsod400_clean_protocol/logs/sixiang_anchor_residual_v1_20260722_b2/queue_state.json'
d=json.load(open(p))
jobs=d['jobs']
print(Counter(j['status'] for j in jobs))
for j in jobs:
    if j['status']=='running':
        print(j.get('arm'), j.get('seed'), j.get('run_dir'))
PY
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits"
```

不要用 `tail results.csv`。如果必须检查训练是否继续，只统计行数：

```bash
wc -l /private/results/sixiang_anchor_residual_v1_20260722_b2/*/results.csv
```

低频监测即可；manager 与训练正常时不要手工干预。

## 5.5 B2 终局开放顺序

只有 queue 到 21/21 terminal 后才执行：

1. 冻结 queue state、每格 args/checkpoint/results/diagnostics/provenance 的 source list；
2. 运行 campaign terminal validator；
3. validator 失败则 `DEFER_ENGINEERING`，不运行 efficacy analyzer；
4. validator PASS 后运行 commit-last analyzer；
5. 验证 commit artifact 与 analysis source lock；
6. 再按预注册顺序读取六个 gate，不做临时新对比；
7. 更新 `registry/campaigns.csv` 与 `registry/claims.csv`，再运行 claim validator；
8. 更新 `handoff.md` 顶部 current synthesis。

本地工具入口：

```text
tools/validate_sixiang_anchor_residual_campaign_b2.py
tools/analyze_sixiang_anchor_residual_b2.py
refine-logs/sixiang_anchor_residual_20260722/B2_ANALYSIS_SOURCE_LOCK.json
refine-logs/sixiang_anchor_residual_20260722/EXPERIMENT_FREEZE.md
```

不要直接照抄命令参数猜测运行；先读工具 `--help`、freeze 与锁文件，确认本地/远程 source root 完全一致。

## 5.6 R2A 接管

物理训练已经完成，问题是分析器接受性，不是缺 GPU run。接管顺序：

1. 阅读 R2A runtime、five-arm freeze、confirmation freeze 和 analyzer invalidation receipt；
2. 在**不读取 raw outcome**的前提下审查 analyzer schema、expected cells、epoch identity、seed policy、late10 和 commit-last；
3. 用 synthetic fixtures 跑 analyzer/validator tests；
4. fresh reviewer 接受后，锁定 analyzer/source manifest；
5. 一次性对 immutable terminal roots 分析；
6. formal 表只用 seeds42/123；seed0 单列 context；
7. 按 freeze 判断，不因结果调整门或换 arm。

关键路径：

```text
comparison/runtime/mm_arcs_v2_r2a_hbb/MM_ARCS_R2A_RUNTIME.md
comparison/ablations/mm_arcs_v2_r2a/
comparison/ablations/mm_arcs_r2a_confirmation_v1/
tools/analyze_mm_arcs_r2a_five_arm_exact400.py
tools/analyze_mm_arcs_confirmation_exact400.py
tools/validate_mm_arcs_r2a_exact400_analysis_lock.py
refine-logs/mm_arcs_r2a_confirmation_20260718/EXACT400_ANALYZER_REVIEW_HISTORY.md
refine-logs/mm_arcs_r2a_confirmation_20260718/R2A_FIVE_ARM_ANALYZER_REVIEW_HISTORY.md
```

如果修复需要改变 estimand、expected cells 或阈值，则不是 analyzer repair，而是新 analysis amendment；必须在读 outcome 前冻结。

## 5.7 Baseline/历史结果复核

不要从远程目录名猜身份。最少核对：

```text
run directory
→ args.yaml 全值
→ results.csv 行数/epoch identity
→ best.pt/last.pt 与 teacher/student hash
→ runtime/config/data YAML hash
→ registry row
→ campaign primary artifact
```

尤其注意：

- old full CMD 和 LD 有 checkpoint CRC 问题；
- OGSOD workers=2 matched-v2 与主 workers=8 协议要分 ID；
- SiXiang baseline/u300 roots 的 lineage 不完全；
- SiXiang `lcsr_v1` 实际 argv 与 ladd_plain 相同；
- smoke/quick/direct50/direct100/exact400 不可池化；
- seed0 不进入当前 OGSOD formal aggregate。

## 5.8 新代码或分析工具的最低要求

- 在独立 method/runtime 目录开发，不覆盖 frozen `shared/`、`baseline/code/`、`comparison/runtime/`；
- frozen protocol 修改必须新 version/amendment；
- analyzer/validator/queue tool 必须有合成回归测试；
- fail-closed：missing/nonfinite/hash mismatch 返回非零或 INVALID/DEFER；
- no-op/zero-weight、resume、checkpoint/load、gradient/EMA identity 必须测试；
- 任何真实 run 先登记 `registry/runs.csv`；
- 不能把工具可运行写成训练结果。

## 5.9 不要做的事

- 不停止、暂停或重排当前 B2，除非用户明确授权且有工程必要；
- 不从旧 `comparison/code/legacy/` 启动正式 experiment；
- 不把 data/weights/trace 打包或提交；
- 不覆盖 remote result root；
- 不清理失败/hung/retry 目录，它们是 provenance；
- 不在 outcome 可见后“补写”预注册阈值；
- 不直接更新 paper claim，而不先更新 campaign/claim ledger 与 validator；
- 不在没有当前授权时调用外部模型传输项目材料。

## 5.10 最小优先级队列

1. **P0**：只读等待 B2 自然结束，执行 terminal validator → commit-last analyzer。
2. **P0**：独立修复/复核 R2A analyzer，保持 outcome blind。
3. **P1**：reconcile SiXiang direct300 SAR/RGB baseline lineage。
4. **P1**：若任一新方法通过，建立新 confirmation，加入 matched full RGB-CMD，并保持 SiXiang test untouched。
5. **P2**：准备 group/near-duplicate audit 和 M4-SAR/其他 external evaluation。
6. **停止项**：不启动第三大矩阵，不重启 LCSR/RIF/H_F 大规模 campaign。

当前 handoff decision：**`DEFER`**。新证据到达前，最有价值的工作是闭环现有证据，而不是生成更多目录。
