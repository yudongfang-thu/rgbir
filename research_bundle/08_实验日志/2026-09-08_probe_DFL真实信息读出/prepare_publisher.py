"""Create this entry's publisher from the accepted previous bounded publisher."""
from pathlib import Path
root=Path(__file__).parent
s=(root.parent/'2026-09-08_probe_定位学习位置与目标/sync_evidence.py').read_text(encoding='utf-8')
s=s.replace('completed CPU evidence only','completed single-batch DFL evidence only')
s=s.replace('rgbir_localization_learning_target_20260908','rgbir_dfl_information_20260908')
s=s.replace('LOCALIZATION_LEARNING_TARGET','RAW_DFL_INFORMATION')
s=s.replace('probe_定位学习位置与目标','probe_DFL真实信息读出')
s=s.replace('update_20260908_localization_learning_target','update_20260908_raw_dfl_information')
s=s.replace('2026-09-08 定位学习位置与目标诊断完成','2026-09-08 单批真实DFL信息读出完成')
s=s.replace('复用首32图与8批L2记录，核对实际学习anchor、教师坐标目标和GT控制，给出本版L2去留决定。','固定32图80GT，保存真实四边DFL分布、同次解码及逐对象GT读出，区分额外信息和可迁移性。')
s=s.replace('无模型前向/新训练、权重、原图或凭据。','新增一次固定32图零更新读出，无新训练，无权重、原图或凭据上传。')
with (root/'sync_evidence.py').open('x',encoding='utf-8') as f:f.write(s)
print('Publisher prepared, not executed')
