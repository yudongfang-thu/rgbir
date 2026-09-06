import json,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent
files={n:(here/n).read_text(encoding='utf-8') for n in ['README.md','plan_lineage.md','PARALLEL_WORKER_REVIEW.md']}
code='from pathlib import Path\nimport json\nfiles='+repr(files)+'''\nroot=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
base=root/'artifacts/oev1_concurrency_20260906'
for name,contents in files.items():
    p=base/name
    if p.exists():assert p.read_text()==contents, 'existing note differs: '+str(p)
    else:
        with p.open('x') as f:f.write(contents)
index=root/'PROJECTS_INDEX_20260906_OEV1_CONCURRENCY.md'
with index.open('x') as f:f.write('# OEv1 concurrency scheduling addition\\n\\nArtifact: artifacts/oev1_concurrency_20260906/README.md\\nLocal: E:/SHARE/光sar/08_实验日志/2026-09-06_ops_单卡并发与计划澄清/README.md\\n\\nOnly new profile outputs, pending random seed ownership, and documented reservation scalar changes. Original train/eval recipe, existing runs, checkpoints and source releases retained. See PARALLEL_HANDOFF_PLAN.md for ownership and future legacy queue exit interpretation.\\n')
print(json.dumps({'status':'synced','files':list(files),'index':str(index)}))
'''
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
r=subprocess.run(['ssh','-o','BatchMode=yes','94',py,'-'],input=code.encode(),capture_output=True)
(here/'notes_sync.json').write_bytes(r.stdout)
print(r.stdout.decode(errors='replace'))
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
