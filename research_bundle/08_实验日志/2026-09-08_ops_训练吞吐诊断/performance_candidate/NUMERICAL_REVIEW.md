# 真实raw-batch等价失败与简单归约检查

**v1的真实batch00总体状态为FAIL_EQUIVALENCE，不能因loss/gradient相同而改写该状态；固定容差不变。后续24update仍可作为诊断检验完整训练影响，不代表已准入长训。**

本地已只读回验 `../remote_real_probe_attempt1/replay_00_attempt1/receipt.json`。同一真实raw输入、435个base对象、两层五类共4350个delta：S/T/R delta的最大绝对差均为1.9073486328125e-6；其中T delta存在超过atol1e-6、rtol1e-5的值，S/R满足该容差但也非bitwise。valid、eligible、selected、质量q、所有对象映射、C0stats、C0 loss均exact；C1 loss和1,344,000元素的S raw-score梯度也exact，S DFL/T/R梯度均None。RNG没有变化。

数学定义未改变：前景和背景各做log-mean-exp再相减。v1将旧每次[C,A]归约改为[chunk,C,A]，CUDA可采用不同并行归约布局；大致相近的前景/背景项相减会暴露微小浮点差。这是目前基于源码的候选解释，不能只凭最大差就断言具体kernel内部行为已经查明。

新增 `pool_block16_v2.py` 与 `benchmark_candidate_v2.py` 仅把masked张量reshape为[chunk*C,A]再logsumexp(1)，恢复二维归约rank。block16、-1e30哨兵、count.clamp_min(1).float().log()、invalid零、所有门、选择及容差均不变。旧实现本来就是torch计数log，不存在math.log/int转换修正。`cpu_checks_rank2_attempt1`的12项CPU检查通过；GPU需用完全相同raw_batch_00复验，不能用CPU通过替代。

算子或raw-score等价、完整轨迹精确、最终训练效果是三层不同证据。当前失配必须保留；24次成功更新可检查实际AMP缩放/优化器/EMA是否放大或消除差异，且必须保留预定容差和每步差异，不能以短检验通过宣称E200 bitwise相同或正式切换已授权。

v1和其失败attempt均未修改；rank2候选单独留文件。没有改在跑release、native loss、GT、C0门或λ，没有新增GPU运行或hash；远端测量由主代理的原lease执行。
