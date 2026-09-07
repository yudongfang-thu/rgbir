"""Append the implementation stage and its detailed child inventories."""
import datetime
from pathlib import Path

log=Path(__file__).resolve().parent
workspace=log.parents[1]
manifest=workspace/'MANIFEST.md'
stamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
names=('README.md','FORMAL_LAUNCH_ACCEPTANCE_1535.md','FOLLOWUP_STATUS.md',
 'deployment_release_gpu5.json','remote_admission_1532/source_manifest.json',
 'old_endpoint_analysis_1407.json','OLD_RESULTS_1407.md','PRIMARY_GEOMETRY_VISUAL_24.md',
 'INDEPENDENT_GEOMETRY_VISUAL_5.md','EXTERNAL_BASELINE_PREPARATION.md',
 'EVALUATION_BRIDGE_PREPARATION.md','object_analyzer_independent_review_v1/review_receipt.json',
 'formal_campaign.py','FORMAL_WORKER_VENV_FIX.md','formal_live_20260907_153211.json',
 'publish_independent_stage.py')
with manifest.open('a',encoding='utf-8') as stream:
    for name in names:
        path=(log/name).relative_to(workspace).as_posix()
        stream.write(f'| {stamp} | /experiment-bridge + /run-experiment | {path} | implementation | C1三seed正式启动；六兼容/64批校准/双开；L几何阻塞；详细源码与小产物路径见子清单 |\n')
print('Registered implementation outputs and detailed child inventories.')
