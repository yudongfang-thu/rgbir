"""Update only derived GitHub navigation for the completed supplemental stage."""
from pathlib import Path

REPO=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
PREFIX='research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/'
readme=REPO/'README.md';value=readme.read_text(encoding='utf-8')
marker='**16:35 补充阶段：'
if marker not in value:
    first,rest=value.split('\n',1)
    value=first+'\n\n'+marker+'C1三seed进入第4轮；旧N/C0六次补评均与历史五指标逐项一致，实际观察桥接已接受。旧C0净正确对象+4/−3/+101、背景误检+3/+5/+46，后者触发专项复核。C0后续自动扩展暂停，已有训练继续；L仍缺几何证据。** [最新阶段报告]('+PREFIX+'STAGE_REPORT_1635.md) · [真实对象诊断]('+PREFIX+'old_c0_object_diagnostics_v1/README.md)。\n'+rest
value=value.replace('('+PREFIX+'OBJECT_ERROR_ANALYSIS_RULES.md)','('+PREFIX+'object_error_analyzer_rect_v2/OBJECT_ERROR_ANALYSIS_RULES.md)')
readme.write_bytes(value.encode('utf-8'))
(REPO/'LATEST_RESULTS.md').write_bytes(('# 最新结果与执行状态\n\n'
    '[完整16:35阶段报告]('+PREFIX+'STAGE_REPORT_1635.md)。C1三个seed正在E200第4轮，λ=.09227393550836771，尚无新C1完整AP。\n\n'
    '旧N/C0/random九端点齐：C0−N +.266655±.144373pp，C0−random +.174939±.038853pp；四臂归因未齐。旧N/C0六次完整dev补评已全部成功，五汇总指标逐项exact；[观察桥接独立接受]('+PREFIX+'legacy_observed_bridge_root_review_v1/REVIEW.md)。\n\n'
    '新对象诊断：修复473/472/517，损伤469/475/416，净+4/−3/+101；背景误检+3/+5/+46。后者触发C0专项损伤复核，暂停其后续自动扩展，已有训练继续。[逐类、尺度和部分亮度结果]('+PREFIX+'old_c0_object_diagnostics_v1/README.md)。\n\n'
    'L仍因几何证据不足阻塞。正式分析器对posthoc逐类证据的显式接入仍待完成；不向旧JSON注入新字段。新三seed结果、内容消融和完整四臂尚未完成。\n').encode('utf-8'))
prompt=REPO/'INDEPENDENT_KD_REVIEW_PROMPT.md';value=prompt.read_text(encoding='utf-8')
value=value.replace('先读 [实施README]','先读 [16:35阶段报告]('+PREFIX+'STAGE_REPORT_1635.md)，再读 [实施README]')
value=value.replace('('+PREFIX+'object_error_analysis.py)','('+PREFIX+'object_error_analyzer_rect_v2/object_error_analysis.py)')
if '补充复核任务' not in value:
    value+='\n\n补充复核任务：核查真实rect padding导致旧max640检查失效后的最小修订、三seed对象诊断的修复/损伤分母及背景误检同向增加。评价桥接只接受六个同checkpoint的五汇总指标观察等价；实际九源均保留，七份共同核心按正式C1顺序对齐，不伪造历史库字节。正式posthoc逐类证据接入仍未实现，不能把旧指标没有的字段视为已经补进正式分析器。\n'
prompt.write_bytes(value.encode('utf-8'))
print('Updated three derived navigation files; scientific inputs unchanged.')
