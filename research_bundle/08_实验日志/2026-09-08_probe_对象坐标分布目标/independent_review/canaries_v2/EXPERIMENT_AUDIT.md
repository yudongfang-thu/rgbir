# v2 三臂实际 canary 独立审计

**PASS**：N / L3-DFL / L3-GT 均真实完成24次成功optimizer更新；每臂58个batch、29次更新尝试、5次AMP skip，EMA计数29，分别如实记录。三臂均clean exit，外部monitor无错误。

499个完整初始状态张量（含head）来自同一成熟checkpoint，fresh optimizer/EMA；243个BN running buffers不变，affine可学。执行source manifest绑定reviewed release_v2，effective config逐字节相等，DFL/GT共享λ=.6900524651944485、N为0。教师/R均无梯度，两个KD臂实际共享学生参数梯度非零；N全部可见loss记录等于native total。

三臂前30批32图文件、RGB/IR标签及batch_idx记录完全相同，且三个sample_stream文件逐字节一致；所有可见选择/分母记录匹配，终态各190个选中对象。没有把这称为全58批像素张量逐位核验。

三臂NVML峰均5694 MiB，allocated峰4732.996 MiB，reserved峰5162 MiB。按原实际峰+余量规则独立重算，三臂正式短训预约均为6144 MiB显存、20480 MiB进程树RSS。整卡最小空闲分别14886/14885/14889 MiB，均保留了2 GiB余量。

新快照还补齐了八批校准clean exit且monitor无错；校准四份关键数据与上一审计逐字节一致，解除此前仅缺外层终态的限定。本审计只覆盖校准与真实canary，后续FT3和实际AP仍须终态证据，不作KD收益或物理配准结论。
