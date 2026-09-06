"""Snapshot new code to a unique remote release without overwriting existing artifacts."""
import argparse
import ast
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
SOURCE=WORKSPACE/'03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907'


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',required=True);a=p.parse_args()
    if not a.release.replace('_','').isalnum():raise ValueError('Simple release identifier required')
    files={}
    for path in SOURCE.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.yaml','.md','.json') and '__pycache__' not in path.parts:
            source=path.read_text(encoding='utf-8')
            if path.suffix=='.py':ast.parse(source)
            files[path.relative_to(SOURCE).as_posix()]=source
    files['EXPERIMENT_PLAN.md']=(HERE/'EXPERIMENT_PLAN.md').read_text(encoding='utf-8')
    remote=BASE+'/'+a.release
    code='from pathlib import Path\nimport json,datetime\nbase=Path('+repr(remote)+')\nfiles='+repr(files)+'''\nbase.mkdir(parents=True,exist_ok=False)
for name,content in files.items():
 p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content,encoding='utf-8')
 for_check=p.read_text(encoding='utf-8')
 if for_check!=content:raise RuntimeError('Round-trip source differs: '+str(p))
print(json.dumps({'status':'DEPLOYED','remote':str(base),'files':len(files),'time':datetime.datetime.now().astimezone().isoformat(),'overwrite':False}))
'''
    result=subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=code.encode('utf-8'),capture_output=True)
    (HERE/('deploy_'+a.release+'_stdout.json')).write_bytes(result.stdout)
    (HERE/('deploy_'+a.release+'_stderr.txt')).write_bytes(result.stderr)
    print(result.stdout.decode('utf-8',errors='replace'))
    if result.returncode:raise RuntimeError(result.stderr.decode('utf-8',errors='replace'))


if __name__=='__main__':main()
