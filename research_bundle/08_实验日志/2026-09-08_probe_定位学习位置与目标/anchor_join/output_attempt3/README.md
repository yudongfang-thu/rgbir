# 首32图native框与历史L2学习anchor

仅CPU连接现有记录，待独立验收。全部80GT、历史base79与selected7保持原分母。

|分组|GT|C selected|L2 selected|R候选对S匹配 same/different/unavailable|实际学习对S匹配 same/different/unavailable|
|---|---:|---:|---:|---|---|
|all80|80|32|7|73/4/3|5/1/74|
|native_T_localization11|11|8|4|11/0/0|4/0/7|
|native_S_reverse1|1|0|0|1/0/0|0/0/1|
|L2_selected7|7|6|7|5/1/1|5/1/1|
|T_localization_intersection_L2_selected|4|4|4|4/0/0|4/0/0|

逐对象保留真实frame/GT/prediction/anchor ID、level/row/col/center、native框与FP32历史/dense字段。same来自ID而不是框相等。C是区域池化，不能把C selected冒称L2学习位置。未通过reference gap而未执行教师门的null全部保留。

缺字段：历史S在学习anchor的DFL和decode未保存；不能用R框或当前native AMP框补成历史S状态。两次forward仅沿用旧桥接已证明的配置/流/GT/初始化和部分FP32重叠，不声称整张raw或像素exact。此项不重跑任何匹配/selector，不提供梯度或增益结论。
