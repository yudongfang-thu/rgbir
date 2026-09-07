"""Relocate accepted CPU analyzer source and review paths without changing logic."""
import datetime
import json
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
OUT=HERE/'legacy_object_analyzer_deployment_v1'
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907'
REMOTE=BASE+'/legacy_object_analyzer_v1'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
original=HERE/'object_analyzer_independent_review_v1/review_receipt.json'
review=json.loads(original.read_text(encoding='utf-8'))
assert review['status']=='ACCEPTED' and review['blocking_issues_remaining']==0
OUT.mkdir(exist_ok=False);(OUT/'reviewed_sources').mkdir()
mapping=[]
for row in review['source_files']:
    source=HERE/row['relative'];accepted=Path(row['accepted_copy'])
    if source.read_bytes()!=accepted.read_bytes():raise ValueError('Source changed since independent acceptance')
    (OUT/row['relative']).write_bytes(source.read_bytes())
    (OUT/'reviewed_sources'/row['relative']).write_bytes(accepted.read_bytes())
    mapping.append(dict(original=row['accepted_copy'],remote=REMOTE+'/reviewed_sources/'+row['relative'],bytes=accepted.stat().st_size))
    row['accepted_copy']=mapping[-1]['remote']
(OUT/'original_review_receipt.json').write_bytes(original.read_bytes())
review['path_mapping_only']=True
review['original_review_receipt']=REMOTE+'/original_review_receipt.json'
(OUT/'review_receipt.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'path_mapping_receipt.json').write_text(json.dumps(dict(created_at=datetime.datetime.now().astimezone().isoformat(),
    status='SOURCE_BYTES_EQUAL_PATH_RELOCATION_ONLY',mapped_by='/root',mapping=mapping,
    code_changed=False,original_review_rewritten=False,new_analysis_run=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
archive=HERE/'legacy_object_analyzer_v1_deployment.tar.gz'
with tarfile.open(archive,'x:gz') as stream:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():stream.add(path,arcname=path.relative_to(OUT).as_posix(),recursive=False)
check='from pathlib import Path\nfor name in '+repr([REMOTE,BASE+'/'+archive.name])+':\n if Path(name).exists(): raise FileExistsError(name)\n'
subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=check.encode(),check=True)
subprocess.run(['scp',str(archive),'94:'+BASE+'/'+archive.name],check=True)
code=('from pathlib import Path\nimport json,sys,tarfile,subprocess\nroot=Path('+repr(REMOTE)+')\nroot.mkdir(exist_ok=False)\n'
      'with tarfile.open('+repr(BASE+'/'+archive.name)+') as stream:\n'
      ' for entry in stream.getmembers():\n'
      '  if not entry.isfile() or Path(entry.name).is_absolute() or ".." in Path(entry.name).parts:raise ValueError("Invalid member")\n'
      ' stream.extractall(root)\n'
      'sys.path.insert(0,str(root))\nfrom object_error_analysis import verify_review\n'
      'assert verify_review(root/"review_receipt.json")\n'
      'result=subprocess.run(['+repr(PY)+',"-m","unittest","test_object_error_analysis","-v"],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)\n'
      'with (root/"pinned_cpu_tests.log").open("x") as stream:stream.write(result.stdout)\n'
      'if result.returncode:raise RuntimeError(result.stdout)\n'
      'print(json.dumps(dict(status="DEPLOYED_ACCEPTED_CPU_ONLY",tests_returncode=result.returncode,output=str(root),test_log=result.stdout)))\n')
result=subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
value=json.loads(result.stdout)
(OUT/'deployment_receipt.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:value[k] for k in ('status','tests_returncode','output')}))
