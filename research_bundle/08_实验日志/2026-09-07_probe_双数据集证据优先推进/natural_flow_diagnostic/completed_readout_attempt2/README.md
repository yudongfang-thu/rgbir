# 完整自然64批选择诊断汇总

**仅使用两组完成的64批full；原JSONL保持不变，所有计数由逐对象/逐图记录回算并核对summary。真实结果已获[独立限定接受](../independent_review/REAL_RESULTS_EXPERIMENT_AUDIT.md)，7份CSV共15060行/90780单元另经独立核对通过，见[表格回执](../independent_review/real_results_v1/readout_tables_review.json)。原summary生成时的PENDING标记保留，接受状态以该后续回执为准。**

状态为 UNVERIFIED_GEOMETRY_DIAGNOSTIC。C0/C1共享同一选择集；L为未验证几何的全放行诊断，不是已准入L1或严格数学上界。没有backward、校准、模型更新或AP。

|dataset|C_base|C_eligible|C_selected|C_selected_batches|C_selected_images|C_selected_groups|L_base|L_selected|L_selected_batches|L_selected_images|L_selected_groups|
|---|---|---|---|---|---|---|---|---|---|---|---|
|llvip|5128|4406|2217|64|1317|14|5033|207|62|187|14|
|dronevehicle|28297|15616|7821|64|1317|64|21488|602|63|268|31|

每组固定2048个自然源图；selected_batches指存在至少一个选中对象的批，不是非零真实共享梯度批。以下集中度以该选择集合的对象贡献为分母；未知来源单列，来源组不是标定组。

|dataset|family|unit|objects|unique_units|top1_fraction|top5_fraction|unknown_group_objects|
|---|---|---|---|---|---|---|---|
|llvip|C0_C1|group|2217|14|0.2426702751465945|0.607126747857465|0|
|llvip|C0_C1|image|2217|1317|0.0036084799278304014|0.016238159675236806|0|
|llvip|L_UNVERIFIED|group|207|14|0.1642512077294686|0.6473429951690821|0|
|llvip|L_UNVERIFIED|image|207|187|0.014492753623188406|0.05314009661835749|0|
|dronevehicle|C0_C1|group|7821|64|0.4982738780207135|0.7197289349188083|0|
|dronevehicle|C0_C1|image|7821|1317|0.009589566551591868|0.04104334484081319|0|
|dronevehicle|L_UNVERIFIED|group|602|31|0.6046511627906976|0.8604651162790697|0|
|dronevehicle|L_UNVERIFIED|image|602|268|0.03156146179401993|0.10299003322259136|0|

详细表：gates.csv为定位累计gate的对象损失/前级分母、图组与有对象批；classes.csv为全类别C base/eligible/selected比例及L类别计数；levels.csv为C的有效P3/P4共同层（两层都有效可同时计数，并非互斥物体尺寸档）；strides.csv为L主选择anchor的stride；histograms.csv保留全部图/组贡献便于核算top1/top5。

计数的样本来自固定自然增强窗口，且使用cfg原冻结T/R。Drone reference是旧formal_native RGB42，不能和近期新N42读出混用。自然流完整复现不证明独立几何，也不保证已选对象提供非零/有益梯度。正式几何、校准和对照要求保持不变。
