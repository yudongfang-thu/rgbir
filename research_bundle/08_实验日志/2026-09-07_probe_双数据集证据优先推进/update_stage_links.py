"""Update explanatory navigation only; never change raw results."""
from pathlib import Path
WORK=Path('E:/SHARE/光sar');ROOT=Path(__file__).resolve().parent
def prepend_after_title(p,note):
    text=p.read_text(encoding='utf-8-sig');first,rest=text.split('\n',1)
    if note not in text:p.write_text(first+'\n\n'+note+'\n'+rest,encoding='utf-8')
old=WORK/'08_实验日志/2026-09-07_probe_Baseline蒸馏机会重诊断'
correction='> **解释范围补充（2026-09-07）**：下文“461中351低置信”仅为200dev对象计数结论。新完整1469dev、三seed AP诊断显示AP50的少数类混淆更突出，freight car/truck/van贡献macro分类oracle的94.26%。不能将旧对象计数解释推广为整个AP瓶颈。原数值保留，见[双数据集新阶段判断](../2026-09-07_probe_双数据集证据优先推进/STAGE_REPORT.md)。'
for name in ['README.md','STAGE_REPORT.md']:prepend_after_title(old/name,correction)
prepend_after_title(WORK/'README.md','> **双数据集证据推进（2026-09-07）**：[最新阶段判断](08_实验日志/2026-09-07_probe_双数据集证据优先推进/STAGE_REPORT.md)。LLVIP完整dev旧RGB/IR baseline mAP32.8784/48.8529，优先定位；Drone完整AP指向少数类混淆，修正对象数量代表AP瓶颈的解释。C1继续三seed，新的定位训练尚未准入。')
p=ROOT/'localization_stress/README.md';t=p.read_text(encoding='utf-8');t=t.replace('已执行两数据集的固定 25 条件 CPU 诊断，等待独立审阅。','已执行两数据集的固定 25 条件 CPU 诊断，attempt2已获独立审阅限定描述性PASS（见independent_review/EXPERIMENT_AUDIT.md）。');p.write_text(t,encoding='utf-8')
prepend_after_title(ROOT/'llvip_full_eval/README.md','> **独立审阅通过**：[实际全量回执复核](../ap_error/llvip_independent_review/EXPERIMENT_AUDIT.md)限定接受旧baseline dev结果；四个TIDE AP另经独立COCO重算，最大差1.42e−14pp。')
prepend_after_title(ROOT/'ap_error/README_LLVIP.md','> **独立审阅PASS**：[审阅报告](llvip_independent_review/EXPERIMENT_AUDIT.md)验证两模型全量GT、明确相对路径配对、TIDE oracle与COCO AP。完整full只运行capture；64图canary才是双路径对照。')
prepend_after_title(WORK/'refine-logs/EXPERIMENT_PLAN.md','> **证据排序更新（2026-09-07晚）**：LLVIP完整dev与固定定位压力支持优先核真实自然64批定位覆盖；Drone完整六端点AP提示少数类混淆优先于把对象数量上的低置信解释推广到macro AP。C1与旧控制保持冻结训练；新增自然流选择诊断不执行backward、不变更L1，不承诺新长训。见[阶段报告](../08_实验日志/2026-09-07_probe_双数据集证据优先推进/STAGE_REPORT.md)。')
prepend_after_title(WORK/'refine-logs/EXPERIMENT_TRACKER.md','> **最新证据（2026-09-07 21:29）**：LLVIP完整2406dev旧RGB/IR mAP32.8784/48.8529与独立AP审阅完成；Drone六端点AP/逐类/对象桥接和两数据集定位压力均已独立接受。C1三seed第20轮（CSV19），旧控制66/58，项目RSS141.50GiB、3卡5训练。自然64批选择诊断正在执行准备；无新E200或定位准入。')
