# 源码与CPU验证审查

## 原源码来源

2026-09-06 23:18（UTC+8）通过SSH只读读取94真实release_v2。训练器、loss、loader、evaluator、config、既有两组test共7文件，均与首轮本地code直接字节相同。原文件保存在source_baseline；原P/N文件未改。

## 唯一方法接线差异

```diff
- config=self.evidence_cfg, arm='paired', seed=self.cfg['seed']+self.calls)
+ config=self.evidence_cfg, arm=('paired' if self.arm == 'weight0' else self.arm), seed=self.cfg['seed']+self.calls)
- parser.add_argument('--arm',choices=['paired','weight0'],required=True)
+ parser.add_argument('--arm',choices=['paired','weight0','paired_random'],required=True)
```

P仍传paired；N仍在loss内计算paired诊断、在外部乘0。新增R才传paired_random。原loss已实现私有CPU Generator及同K抽样，本次不改其数学定义。新加的test/启动/验证工具不改变训练计算图。

## 已执行检查

1. 本地静态检查：训练器必须恰好等于原源码实施上述两处替换；其余6个来源文件完全相同；所有Python文件可AST解析。
2. 94 CPU流式执行（不保存远端文件、不使用GPU）：既有loss算子13项全部通过。
3. 新增4项CPU检查通过：
   - 从新旧实际训练器提取真实criterion代码，使用受控native/loss替身执行，P/N的loss与student score gradient直接相同，R确实透传paired_random。
   - 稀疏base下K、normalizer、名义剂量一致，抽样无重复且不越出base，确实能抽到未通过可靠性/质量门的对象；私有RNG不修改全局torch RNG，固定seed可复现。
   - 真正object_evidence_loss双目标fixture中，R可选到不可靠教师对象，但base/eligible/K/normalizer/dose与P完全相同，梯度保持有限。
   - base非空但eligible为0时，R也不产生额外蒸馏。

17 tests，0 failures，0 errors；torch2.10.0+cu128，CUDA_VISIBLE_DEVICES为空，cuda_initialized=false。

## 原P canary可比性预检查

本次再次只读核查真实路径：

`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_v1_20260906/canary_paired_s42_attempt1`

其implementation_snapshot中trainer/loss/loader均与当前原release_v2直接字节相同，protocol_config也相同；initial_student.pt、first_batch.pt都存在，kd_batches包含30条日志。因此可由validate_random_canary.py与新R seed42作直接比较。

## 尚待真实执行

尚未完成本随机臂GPU canary、真实模型梯度/显存核查、初始化和首batch逐张量比较、逐batch同K验证或完整训练。CPU fixture通过不替代真实canary；root执行后在对应回执中追加结果。

来源与科学代码复制过程含本地来源清单；远端canary验证采用直接字节/张量比较，不新生成哈希。
