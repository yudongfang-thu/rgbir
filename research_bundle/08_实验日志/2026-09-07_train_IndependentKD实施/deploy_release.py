"""Deploy an immutable, reviewable source release; never copy weights or credentials."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / '03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'
REMOTE = '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907'
PY = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('release')
    a = p.parse_args()
    if not a.release.replace('_','').isalnum():
        raise ValueError('Simple release name required')
    archive = HERE / (a.release + '.tar.gz')
    files = sorted(x for x in SOURCE.rglob('*') if x.is_file() and x.suffix in ('.py','.yaml','.yml','.md','.json') and '__pycache__' not in x.parts)
    with tarfile.open(archive, 'x:gz') as tar:
        for file in files:
            tar.add(file, arcname=str(file.relative_to(SOURCE)).replace('\\','/'))
    subprocess.run(['scp',str(archive),'94:'+REMOTE+'/'+archive.name],check=True)
    code = 'from pathlib import Path\nimport tarfile\np=Path('+repr(REMOTE+'/'+a.release)+')\np.mkdir(exist_ok=False)\nwith tarfile.open('+repr(REMOTE+'/'+archive.name)+') as t:t.extractall(p)\nprint(str(p))\n'
    subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=code.encode(),check=True)
    receipt = dict(release=a.release, remote=REMOTE+'/'+a.release, local=str(SOURCE),
        deployed_at=datetime.datetime.now().astimezone().isoformat(),
        files=[dict(relative=str(x.relative_to(SOURCE)),bytes=x.stat().st_size) for x in files])
    (HERE/('deployment_'+a.release+'.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
