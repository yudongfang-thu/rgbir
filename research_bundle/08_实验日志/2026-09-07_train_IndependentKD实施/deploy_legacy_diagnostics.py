"""Deploy the reviewed supplemental evaluator; GPU dispatch is a separate action."""
import datetime
import json
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'legacy_endpoint_eval_prepare_v1'
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907'
REMOTE=BASE+'/legacy_endpoint_eval_prepare_v1'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'

def main():
    receipt=json.loads((SOURCE/'review_receipt.json').read_text(encoding='utf-8'))
    if receipt.get('status')!='ACCEPTED' or not receipt.get('reviewer'):
        raise ValueError('Independent acceptance required before deployment')
    for row in receipt['source_files']:
        reviewed=SOURCE/row['accepted_copy'];actual=SOURCE/row['relative']
        if reviewed.resolve()==actual.resolve() or reviewed.read_bytes()!=actual.read_bytes():
            raise ValueError('Accepted source differs: '+row['relative'])
    files=sorted(p for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p.suffix.lower() in ('.py','.json','.md','.yaml','.yml','.txt','.log'))
    archive=HERE/'legacy_endpoint_eval_prepare_v1_deployment.tar.gz'
    with tarfile.open(archive,'x:gz') as stream:
        for path in files:stream.add(path,arcname=path.relative_to(SOURCE).as_posix(),recursive=False)
    check='from pathlib import Path\nfor name in '+repr([REMOTE,BASE+'/'+archive.name])+':\n if Path(name).exists(): raise FileExistsError(name)\n'
    subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=check.encode('utf-8'),check=True)
    subprocess.run(['scp',str(archive),'94:'+BASE+'/'+archive.name],check=True)
    code=('from pathlib import Path\nimport tarfile\nimport sys\n'
          'root=Path('+repr(REMOTE)+')\nroot.mkdir(exist_ok=False)\n'
          'with tarfile.open('+repr(BASE+'/'+archive.name)+') as stream:\n'
          ' for entry in stream.getmembers():\n'
          '  if not entry.isfile() or Path(entry.name).is_absolute() or ".." in Path(entry.name).parts: raise ValueError("Unexpected archive member")\n'
          ' stream.extractall(root)\n'
          'sys.path.insert(0,str(root))\nfrom legacy_checkpoint_evaluate import require_review\n'
          'require_review(root/"review_receipt.json")\nprint("DEPLOYED_SOURCE_REVIEW_BYTES_MATCH; NO_GPU_LAUNCH")\n')
    subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=code.encode('utf-8'),check=True)
    value=dict(status='DEPLOYED_NOT_LAUNCHED',created_at=datetime.datetime.now().astimezone().isoformat(),
        remote=REMOTE,local=str(SOURCE),python_entry=PY,reviewer=receipt['reviewer'],
        files=[dict(relative=p.relative_to(SOURCE).as_posix(),bytes=p.stat().st_size) for p in files])
    with (HERE/'legacy_diagnostics_deployment_v1.json').open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=value['status'],files=len(files),archive_bytes=archive.stat().st_size)))

if __name__=='__main__':main()
