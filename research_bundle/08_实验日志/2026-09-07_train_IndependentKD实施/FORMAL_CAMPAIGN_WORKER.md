# C1 正式三 seed 持久协调器

**状态：准备代码，未启动；尚无真实 formal readiness，不制造任何可运行 profile 或接受回执。** 文件位于实施 LOG，不修改已冻结训练 release。需独立审阅后由根决定实际部署。

## 行为

`formal_campaign.py` 有 prepare / coordinate / worker 三个入口。prepare 只读取真实配置和测量、核验并生成新的 campaign，不启动 screen、GPU或训练。coordinate 应放在 `ikdv2_C1_coordinator` screen；它为每seed创建 `ikdv2_C1_42`、`ikdv2_C1_0`、`ikdv2_C1_123` 独立 screen。每个 worker 自己持有训练→独立评估责任，协调器断开不丢弃已启动 worker。

42→0→123 是**实际 LAUNCHED** 顺序：前一seed的训练资源事件中出现 LAUNCHED，才创建下一worker。screen 创建成功或资源 QUEUED 不能代替 LAUNCHED。已经启动的训练可并发，GPU始终由当前 release 的 `resource_dispatch.run_job` 动态选择；不硬编码 GPU、不新建 lease池。

每worker依次直接调用 `run_job(train)` 再 `run_job(evaluation)`，不把依赖的两阶段交给会优先排序 eval 的 `ordered_jobs`。训练完整E200，无max-steps、无中途AP判断；结束后固定last/EMA独立eval。默认仅attempt1，不自动覆盖或接续失败训练。

## 真输入与准备命令

```text
python formal_campaign.py prepare --campaign /mnt/dataset/yudongfang/<new-campaign> --release <immutable-release> --repo <actual-project> --formal-config-dir <real-frozen-C1-config-directory> --canary-profile <real-completed-canary-resource-profile.json> --evaluation-profile <real-completed-evaluation-profile.json> --python <pinned-python>
```

目录必须存在三个真实 C1 paired seed42/0/123、E200、FROZEN、formal_training_authorized=true 的配置；逐个调用该 release 的 protocol / admission 检查。profile 必须是实际完成的 canary / evaluation_profile，保留其原始文件及测量，调用真实 `measured_reservation` 验证源码/config/data绑定；不从假峰值、calibration或布尔ready推导。正式训练VRAM预约固定为ceil(当前C1 canary实际峰值)+2048MiB，评估为ceil(实际评估峰值)+256MiB；RSS均为max(8192MiB, ceil(实际进程树峰值)+4096MiB)。政策及依据写入manifest，不把bootstrap16000MiB上限当实测需要。训练较大余量是为了覆盖原生TAL张量随batch最大GT数量增长、24短步未必遇到密集batch的风险，基数仍为本次C1实测，未借用旧C0峰值。缺少任一项就不能prepare。实际能否共卡仍由真实dispatcher的companion规则和实时资源判断，不满足就排队。

prepare 将配置、两个profile和协调器自身保存原字节副本；worker重读时检查输入未变。生成的manifest是新产物，实际原receipt和失败attempt不改名、不覆盖。所有训练、评估、日志、状态均落 `/mnt/dataset/yudongfang` 下。

准备与独立审阅完成后才可执行的入口（本轮没有执行）：

```text
screen -dmS ikdv2_C1_coordinator <pinned-python> <LOG-or-deployed-ops>/formal_campaign.py coordinate --campaign <prepared-campaign>
```

## 评估优先与竞态

资源不够由真实 `run_job` 每30秒排队。每worker在内存中为 `dispatcher.atomic_acquire` 加一个**只会延迟准入**的 campaign flock门，再原样调用真实 atomic_acquire；磁盘上的release源码没有修改，资源/卡数/显存判断没有复制或放宽。

在同一campaign admission锁下，每次训练申请都重查：是否已暂停、前置seed是否LAUNCHED、是否有已完成E200但尚未LAUNCHED的评估。任何待评估任务都优先于新的训练资源申请，包括先前已在排队的训练；完成文件正在写入时也保守等待。评估真正LAUNCHED后，其余训练可按资源规则并发。优先级的线性化时点是该锁内的资源准入；在这次准入以后才完成的训练不会倒置已经获准的启动。

coordinator/每seed worker各有Linux非阻塞flock单例锁，screen名字冲突不接管；campaign状态更改用同一admission锁和原子替换。只替换派生state.json，所有原事件/日志/结果另存保留。

## 失败与恢复边界

技术失败记录原attempt、worker failure和持久paused状态，后续未获资源的训练退出为DEFERRED_PAUSED；不停止别的已运行E200，也不按AP暂停。仍在运行的worker训练完成后继续履行评估责任。协调器发现worker screen消失会暂停后续启动并保留待eval身份。

若worker意外中断、其训练已形成有效E200完成文件及对应dispatcher成功资源回执，可人工明确重启该seed worker，它只执行尚欠的eval；不会重训。只有训练完成文件而无资源验收不能跳过技术核对。已发布eval但缺dispatcher完成资源回执、已有失败eval attempt或不完整训练，要求先做技术核对，不自动重跑/覆盖。没有自动解除技术暂停或按剩余预算伪装完成。

暂停时其它已完成而待eval的worker仍允许申请评估资源；完整三seed评估完成后coordinator退出。正式结果和claim仍需既有独立evaluator/analyzer准入，本协调器不自行签发训练或分析接受。

## CPU验证范围

`test_formal_campaign.py` 使用明确合成配置/profile/阶段回调，验证seed实际启动顺序、资源门每次重查评估优先、暂停仅阻止后续训练、两个run_job调用的依赖顺序、已完成训练只eval、失败保留、无AP早停。它不验证实际screen/SSH/GPU/显存，不制造真实profile或formal readiness。

最新训练余量修订后作者侧15/15测试通过（0.213s）：canary合成峰值7500MiB预约9548MiB；评估1370MiB显存/6260MiB RSS仍预约1626/10356MiB，而非bootstrap16000/49152。独立代码审阅另行进行，不以本测试声明接受。
