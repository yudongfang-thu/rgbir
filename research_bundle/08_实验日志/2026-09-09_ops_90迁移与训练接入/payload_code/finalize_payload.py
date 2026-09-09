"""Finish path and blocked-template consistency in this payload only."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent
target = '/mnt/dataX/ydf/projects/RGBT_campaign_90'
changes = []

def update(relative, old, new):
    path = root/relative
    source = path.read_text(encoding='utf-8')
    if old not in source:
        raise ValueError('Missing exact anchor: '+relative)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(source.replace(old, new))
    changes.append(dict(destination=relative, operation='final-path-or-blocked-template-edit'))

update('release_gpu5/task_conditional_reference/legacy_oev1/train_object_evidence.py',
       "REPO = Path('"+target+"')", "REPO = Path(os.environ.get('RGBIR90_PROJECT_ROOT', '"+target+"'))")
update('release_gpu5/task_conditional_reference/legacy_oev1/train_object_evidence.py',
       'def build_trainer(cfg, config_path, output, arm, max_steps=None):\n',
       "def build_trainer(cfg, config_path, output, arm, max_steps=None):\n    if cfg.get('port90', {}).get('status') != 'READY':\n        raise ValueError('PORT90_BLOCKED: technical input binding is incomplete')\n")
update('release_gpu5/task_conditional_reference/resource_dispatch.py',
       "PY=str(REPO/'environments/sn6-int8-kd/bin/python')",
       "PY=os.environ.get('RGBIR90_PYTHON', '"+target+"/environments/rgbir90/bin/python')")
update('release_gpu5/task_conditional_reference/resource_dispatch.py',
       "REPO=Path('"+target+"')", "REPO=Path(os.environ.get('RGBIR90_PROJECT_ROOT', '"+target+"'))")
update('release_gpu5/verify_compatibility.py',
       "if os.name != 'posix' or not str(resolved).startswith('/mnt/dataset/yudongfang/'):",
       "if os.name != 'posix' or Path(os.environ.get('RGBIR90_PROJECT_ROOT', '"+target+"')).resolve() not in resolved.parents:")
update('release_gpu5/verify_compatibility.py',
       'Real compatibility traces must be on /mnt/dataset/yudongfang, never the system disk',
       'Real compatibility traces must be under the 90 project data-disk root')
update('production_entry/train_c1_fast.py',
       'No resume or batch-limit option. Requires the original global GPU lease. The\nscientific configuration is an exact frozen per-seed copy of original C1.',
       'No resume or batch-limit option. Requires the unique 90 global GPU lease.\nThe original C1 scientific recipe is retained with 90 paths and new technical bindings.')
for path in (root/'release_gpu5').rglob('*.yaml'):
    source=path.read_text(encoding='utf-8')
    if 'port90:' in source:raise ValueError('Already finalized '+str(path))
    source=source.replace('protocol_status: DRAFT', 'protocol_status: TEMPLATE_BLOCKED')
    if 'formal_training_authorized:' not in source:source+='\nformal_training_authorized: false\n'
    if 'protocol_status:' not in source:source+='protocol_status: TEMPLATE_BLOCKED\n'
    source+='\nport90:\n  status: BLOCKED\n  server: "90"\n  source_only_template: true\n  reason: Frozen example has no bound 90 teacher/reference/data assets\n'
    with path.open('w',encoding='utf-8',newline='\n') as stream:stream.write(source)
    changes.append(dict(destination=path.relative_to(root).as_posix(),operation='mark-frozen-example-blocked'))
path=root/'PACKAGING_RECEIPT.json'
receipt=json.loads(path.read_text(encoding='utf-8'));receipt['changes']+=changes
receipt['template_block_is_technical_not_user_confirmation']=True
with path.open('w',encoding='utf-8',newline='\n') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps(dict(status='PAYLOAD_FINALIZED',new_hashes_computed=False,files_changed=len(changes))))
