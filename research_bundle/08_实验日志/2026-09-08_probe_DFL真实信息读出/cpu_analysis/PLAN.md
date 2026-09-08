# 真实DFL分布CPU读出：先验协议

本计划在新增DFL前向结果可用前冻结。输入为producer独立接受的原首32图新一次前向：80个原GT、已核稳定ID和旧cohort（11个双方粗检出仅T精确、1个反向）。本次不改变旧C/L门，不读新AP，不训练，不重映射概率，不跨不同anchor做KL。

## 两层数据语义

`anchor_distributions.jsonl`按(model,image_index,anchor_index)去重，保存真实4×16 logits、原dtype、FP32/native概率、期望/解码、level/stride/center和bins。`objects.jsonl`按原80GT保存各model.roles：S/R的历史R candidate与本模态native IoU.5匹配anchor；T的同历史R索引、自身native匹配anchor及真实存在的历史selected T独立anchor。每object-role单独携own GT及未clamp距离，不能把一个anchor的多GT解释混为同一个target。

所有T读出相对IR own GT，S/R相对RGB own GT。即便anchor编号相同，也不声明物理配准或分布可直接迁移。

## 固定读出与合法性

1. 从有限真实logits用float64稳定log-softmax计算概率，和保存FP32概率/期望做数值闭合；native dtype/AMP概率、卷积期望、native框另列，不将native与FP32算术混写为逐位相同。
2. 每边报告16个概率、期望（bin）、熵（nat）、方差（bin²）。它们是真实分布描述，熵非零不等于额外可学价值，不将方差直接叫不确定性已校准。
3. GT目标固定为该anchor中心和stride下的原未clamp LTRB距离d。每边只在有限且`0≤d<15`时计算相邻bin插值CE：`−(ceil−d)log p[floor]−(d−floor)log p[floor+1]`，其中ceil=floor+1，即整数d右侧权重0。不调用会clamp目标的训练DFLoss来偷偷改变d。此为DFL相邻bin形式的未clamp诊断，不含训练分配/样本权重，不冒称原实际训练loss。
4. 非法距离该边CE/GT误差=null并写reason；anchor四边mean CE只在四边全部合法时给出。熵和方差仍可描述分布，但与合法GT读出的分母分开。缺失role与存在但非法不能混为零。
5. 固定汇总all80、onlyT精确11、反向1，各model×role分别列对象覆盖、唯一anchor、有效对象/边、CE/熵/方差/期望误差。先看真实记录，不用这些值现场选择更优anchor、阈值或系数。

额外于四坐标能确认的是：同一均值可能对应不同概率形状；保存的熵/方差及GT邻bin对数概率确实包含均值没有确定的信息。是否更可靠、能迁移或有KD收益仍需另外的同语义对照，当前小样本诊断不能证明。

## 合成真值与执行门

先验证：相同均值不同分布可有不同熵/方差/CE；相邻binCE对照手算；距离<0或≥15不clamp且null；大幅logit稳定不产生伪NaN；期望/方差单位及空/缺失role分母。producer布局/源/解码合同与新真实结果独立验收通过后才读出实际CPU原值；保留原产物，输出新目录，不计算hash。
