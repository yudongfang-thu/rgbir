# LLVIP 早期校准：支持稀疏，截断后梯度比例较小

**固定8批有30个合格定位对象，覆盖26图、11个治理来源，7/8批有非零梯度；λ上限1使L2-box活跃批的共享参数梯度范数比中位数仅1.262%，未达到原设10%目标。它与GT控制在初始R输出坐标上的导数高度相似（stack cosine .959），但这不是参数梯度相似性或收益证据。**

快照时间2026-09-08 11:31:42（北京时间），只收已完成的校准、N canary、L2-box canary；当时L2-GT完成回执尚未出现。33文件共1,709,777字节，逐字节收集，未停/改队列、未启动GPU、未读取AP/权重或计算新hash。[收集清单](collection_receipt.json)、[复算脚本](summarize_early.py)、[summary](summary.json)。

## 选择覆盖

校准从共同旧visible42学生起点出发，每批恢复全部参数/缓冲；BN running buffers冻结、affine可训练，无optimizer/EMA更新。8×32=256张唯一图、664个增强后GT，两臂使用相同固定R/T选择。

|连续阶段|对象出现次数|较上一阶段减少|
|---|---:|---:|
|RGB GT / IR GT|664 / 664|—|
|同类配对、pair-IoU门|664|0|
|R原DFL支持|664|0|
|R native unique-owner|663|1|
|coarse R base（固定分母）|649|14|
|R类别正确且置信可靠|629|20|
|R IoU < .7 的定位差距|36|593|
|teacher own框质量|32|4|
|mapped RGB IoU ≥ .6|32|0|
|再要求比R多 .05 IoU|30|2|

base涉及250/256张图；最终30/649=4.6225% base、26/256=10.156%图。主要缩减来自固定参考R已较好：可靠629中593的IoU不低于.7而被排除。它是成熟模型下的固定选择范围，不证明其他GT没有定位改进空间。

输入覆盖全部14个治理prefix；选中支持覆盖11个，其中10组9对象、09组5对象、18组4对象；top1占选中对象30%，top5占73.33%。这是对象出现次数集中度，不能当梯度份额或场景质量。[逐组表](GROUP_TABLE.csv)区分图像曝光、base对象、选中对象和选中唯一图。对象index是增强批内GT行，不是跨批稳定物理对象身份。

## 参数梯度比例与native方向

测量空间是原`shared_parameter_set`的24个参数张量，即model.16/model.19的卷积与BN affine；完整名称见summary。它既不是检测头原始分数梯度，也不是全模型所有参数。比例按实际训练尺度 `||λ·B·∇L2|| / ||∇native.sum()||`，B=32。

|指标|L2-box|L2-GT|
|---|---:|---:|
|raw校准λ中位数|7.926272|8.050264|
|最终共同λ|1|1|
|非零批数|7/8|7/8|
|活跃批实际梯度比例中位数|1.261627%|1.242195%|
|活跃批实际梯度比例范围|0.7358%–3.3627%|0.7262%–3.3627%|
|含零批8批比例中位数|1.138522%|1.198632%|
|native cosine均值|−0.008640|+0.001766|
|native cosine中位数|+0.011160|+0.036682|
|native cosine正/负批数|4 / 3|4 / 3|

L2-box按事前上限截断到1；L2-GT按同mask控制固定共用其λ。不能把λ=1解释成与native等强，也不能声称已经实现10%剂量匹配。L2-box native cosine范围−.187415至+.108765，仅是这些起点、批次和共享参数上的局部一阶关系，不能预测多步训练收益，负cosine也不直接判为有害。[逐批完整精度CSV](BATCH_TABLE.csv)保留原始范数与cosine。复算直接重建7个ratio，与原receipt逐值相等；base/selected计数与两控制臂选择核对exact。

## 教师目标与GT控制是否重复

额外纯CPU计算使用已保存30个selected记录的`reference_box`、`rgb_gt`、`mapped_teacher_box`。令s为R框的GT归一xyxy，t为教师目标，q=(0,0,1,1)，按真实SmoothL1 β=.1计算输出坐标导数`clip((s-target)/.1,-1,1)`。

|输出坐标派生量|原值|
|---|---:|
|全部30×4边未加权stack cosine|0.958663|
|按各批 Bλ/(4base) 缩放后stack cosine|0.959681|
|逐对象cosine均值 / 中位数|0.961448 / 0.979339|
|逐对象cosine范围|0.819382–0.999871|
|teacher−GT归一四边mean-L1：对象均值 / 中位数|0.034885 / 0.034033|
|teacher−GT单边最大归一误差：跨对象最大值|0.199523|
|target逐位相同对象 / 导数逐位相同对象|0 / 0|

因此，**该初始R输出和固定mask下，教师目标与GT目标给出的输出坐标方向高度相似，但并非相同目标或相同导数。** 归一误差按各对象GT宽/高计，不是像素误差；少数单边可到约20%的GT宽/高。LLVIP学生初始checkpoint与R身份一致且BN冻结，但这里仍是从R框作代数推导，未声称真实学生输出逐位相同，更没有得到两损失的Θ参数梯度cosine。卷积Jacobian、其他对象/native分支和后续更新都未被这项坐标计算涵盖。[派生summary](OUTPUT_COORDINATE_SUMMARY.json)、[30对象原值表](OUTPUT_COORDINATE_TABLE.csv)、[含4维导数的JSON](OUTPUT_COORDINATE_OBJECTS.json)、[计算脚本](output_coordinate_similarity.py)。

## 坐标含义与原L1边界

这里使用实例GT定义坐标，是IR GT→RGB GT的正x/y仿射框回归，不是原L1的同anchor/同DFL-bin蒸馏。全部base记录的RGB/IR GT坐标相同，共享标注令这次映射为恒等。正轴仿射本来就保持box-vs-own-GT IoU，因此mapped RGB IoU不是额外独立的物理几何验证。30个选择支持的是这一定义下的教师框候选；原L1几何blocked不变。

## 已完成canary与资源边界

|阶段|秒|NVML峰值MiB|预约MiB|进程树RSS MiB|全卡最低空闲MiB|
|---|---:|---:|---:|---:|---:|
|calibration8|24.883|8506|8192|27508|7515|
|N canary|41.294|5696|8192|27868|10325|
|L2-box canary|42.252|5696|8192|27825|10325|

两个canary均58批、24成功update、29次optimizer attempt、5次AMP skip、29次EMA更新；243个BN running buffers未变。两者均264次selected记录，单凭这个总数不声称轨迹或样本流逐位等价；本快照没有收取重复sample_stream。

**校准NVML实际8506超过8192预约314MiB（3.83%），这项预约没有完全兑现。** 同时全卡最低仍有7515MiB、guard记录COMPLETED且无monitor errors；两事实必须并列，不能据后者抹去前者，也不能把全卡剩余量当作任务峰值。校准allocated/reserved CUDA峰值7377.712/8006MiB；RSS均在32768MiB预约内。以后若复用同校准路径，应依据8506实测另加既定余量；本任务没有修改现有预约。新训练资源按实际canary测量，不把较高校准峰值混称训练峰值。

此快照尚无完整FT3/dev收益证据。下一判断来自已经运行的同协议N/L2-box/L2-GT固定终点；这里不追加实验、不调λ、不按局部统计延长或淘汰。
