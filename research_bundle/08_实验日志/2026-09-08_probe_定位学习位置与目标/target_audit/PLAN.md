# 历史L2教师目标与同anchor GT：事前CPU目标审计

固定输入为旧 `results_1203_snapshot/calibration/llvip/calibration_batches.jsonl` 八批及其完成回执，使用已捕获 `early_llvip_evidence/raw_attempt1/release_v1/localization_box_v2.py` 的实际目标与损失公式。本次不重新选择、不修改原门/λ，不读新AP，不前向/GPU/训练，不计算hash。实现和小真值先于真实目标分析执行；当前源记录中的selected总数30是已知输入事实。

## 固定公式和问题

选择由R/T与双GT完成，学习位置为R选定anchor处的学生DFL解码框；教师独立anchor映射到RGB GT的对象坐标。记RGB GT为g，坐标归一化 `z(b;g)=(b-[gx1,gy1,gx1,gy1])/[gw,gh,gw,gh]`。教师目标t=z(mapped_teacher_box;g)，GT目标q=(0,0,1,1)。单位L2损失为 `sum_selected mean_4edges SmoothL1(z(student_box;g)-target, beta=.1)/max(1,base_count)`；训练调用另外乘实际batch B与λ，不能改成selected均值。两种目标使用原同一选择和anchor。

旧记录没有selected student_box、完整raw DFL或逐参数梯度。故本次以保存的R框定义r=z(reference_box;g)，复算的是**R输出处代理量**，不将其默认等同实际学生loss或参数梯度。虽然校准每批恢复共同S/R初始化并冻结BN，这只是解释为何可作初始参考点，不能补造学生输出逐元素exact。

固定读出：

1. 八批图像出现/不同图、selected对象出现/涉及图/重复图，selected与base分母及L2-box/GT共用选择的闭合。无全八批稳定原GT ID时不称“去重后原生对象数”。
2. 四边教师target−GT残差的有符号/绝对分布，及R−GT误差作尺度参照；保留每对象原框、anchor、目标、归一化残差。核双GT相等及映射是否实际上为恒等，但不证明物理配准。
3. 在r处计算两种SmoothL1、输出导数 `clip((r-target)/.1,-1,1)`，分别保留未经分母缩放、真实 `1/(4M)` 缩放、以及xyxy像素链式缩放 `1/[gw,gh,gw,gh]`。它们不是DFL logit/参数梯度。
4. 不再仅报cosine：按每边原β划分“两者同向饱和/两者线性/一个饱和/反向饱和”，报告导数相等、同向不同量、反向、零导数的数量；检查教师target残差是否小于R误差。说明相近由目标接近/同向饱和能解释哪些部分，不能由余弦单独识别成因。
5. 根据已有字段列出何时教师与GT给不同修正；对未selected对象/后门未执行的teacher null明确缺目标，不外推新门的效果。缺学生DFL概率、计算图/参数Jacobian时，不声称可还原实际KD/shared/native梯度或证明额外信息带来收益。

不扫阈值、β、λ，不用dev机会挑新train子集。关键合成真值包括：相同归一化目标在不同GT尺度下的坐标映射；base分母而非selected分母；饱和掩盖目标差；近GT时教师/GT导数可反向；缺学生输出不得标为实际训练loss。
