**PASS_CANARY_AND_BUDGET_ADMISSION。** 实际 N/C0 canary 回执与独立预算复算支持按已冻结队列继续；这不是完整训练或 AP 终态验收。

| 臂 | batch / optimizer尝试 | 成功更新 | AMP skip / EMA调用 | 脚本秒数 | 实测均值秒/batch | 训练估计（含20%余量） |
|---|---:|---:|---:|---:|---:|---:|
| N | 30 / 30 | 24 | 6 / 30 | 36.772593 | 0.578092 | 378.495239 |
| C0 | 31 / 31 | 24 | 7 / 31 | 31.384069 | 0.547157 | 353.479754 |

两个阶段均真实 clean exit0、monitor_errors为空。共同五类学生499个state tensor、包括检测头：N服务器保存并完整读回，C0实际全元素比较相等；fresh optimizer/EMA、正常BN，训练中243个BN buffer改变。前30批双模态uint8像素及双标签均绑定同一服务器参考，shape/dtype/device、文件stat和保存的标签/增强元数据逐项一致；实际执行回执记录完整torch.equal。原始张量约2.361521 GB只留服务器，本审阅未下载或重复比较，不能外推到512批。

native梯度有限、weight0精确等式通过，C0实际KD梯度非零（首批0.00445516686886549），T/R均无梯度。C0比N多1个batch及1次AMP skip，不能把二者canary称为相同曝光。两臂NVML峰6692MiB、allocated5543.981445MiB、reserved6160MiB；RSS为18120/18000MiB。独立加余量得到两臂预约7168MiB VRAM /20480MiB RSS，原lease整卡余量与项目RSS均通过。

预算使用全部30/31个完整batch，不使用排除前6批的备选列表。已花执行76.867099秒 + N训练378.495239 + C0训练353.479754 + 两次评价120 = **928.842092秒 <2700秒**。像素I/O5.291256/3.554257秒与初始化及其余开销均计入。两canary观测执行合计76.862592秒、driver交接0.004507秒，排队3.285115秒另记。估计不替代后续实际45分钟硬限时。

仅核现有canary、初始化/输入/资源、源码身份与预算；未读中途AP，未新增测试/GPU或改源码，未计算hash。详细逐项检查和路径见CANARY_BUDGET_REVIEW.json及CANARY_BUDGET_INPUTS.json。
