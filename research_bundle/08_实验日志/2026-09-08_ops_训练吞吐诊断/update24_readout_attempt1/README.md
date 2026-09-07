# 24次更新诊断 JSON 复核

**轨迹逐位一致=False；固定容差一致=False；中间selection固定容差一致=False。**

这三个判断分开报告；比较完成不等于生产切换准入。

|路径|成功更新|训练跨度秒|审计秒|扣审计后秒|后热身每批中位秒|
|---|---:|---:|---:|---:|---:|
|old|24|110.734|10.823|99.912|2.4517|
|block16|24|70.673|10.506|60.166|1.1634|

后热身观测batch中位数 old/block16 = 2.107。计时带同步且扣除审计复制/写盘，不含loader fetch，不能称整轮训练加速。

本次CPU仅重算已保存JSON逐项到汇总的闭合；未重读完整state .pt，未运行GPU/SSH。完整梯度/参数精度见summary.json分类字段。

原生kd_batches.jsonl存在时另比实际C1 native/KD/加权KD/total标量；缺失时明确NOT_COLLECTED，不以C0兼容scalar冒充C1损失。
