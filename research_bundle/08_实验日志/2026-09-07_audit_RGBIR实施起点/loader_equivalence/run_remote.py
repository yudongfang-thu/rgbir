"""Execute a CPU-only new audit without changing a frozen implementation release."""
import json
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
PY = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
RELEASE = '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907/release_v4'
OUT_BASE = '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907/audits/loader_equivalence_attempt'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',type=int,required=True)
    args=parser.parse_args()
    local_out=HERE/f'attempt{args.attempt}'
    local_out.mkdir(exist_ok=False)
    source = WORKSPACE/'03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1/verify_loader_equivalence.py'
    code = source.read_text(encoding='utf-8')
    local_tracked = (source.parent/'tracked_pair_data.py').read_text(encoding='utf-8')
    remote = 'from pathlib import Path\nimport json,os,subprocess\n' + f'base=Path({(OUT_BASE+str(args.attempt))!r})\nrelease=Path({RELEASE!r})\n'
    remote += f'assert (release/"tracked_pair_data.py").read_text()=={local_tracked!r}, "Local tracked loader differs from reviewed release"\n'
    remote += 'base.mkdir(parents=True,exist_ok=False)\n'
    remote += f'(base/"verify_loader_equivalence.py").write_text({code!r})\n'
    remote += f'env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")\nr=subprocess.run([{PY!r},str(base/"verify_loader_equivalence.py"),"--module-dir",str(release),"--output",str(base/"result.json")],env=env,capture_output=True,text=True)\n'
    remote += '(base/"stdout.log").write_text(r.stdout)\n(base/"stderr.log").write_text(r.stderr)\nprint(json.dumps({"exit_code":r.returncode,"stdout":r.stdout,"stderr":r.stderr,"result":(base/"result.json").read_text() if (base/"result.json").exists() else None,"remote":str(base)}))\n'
    response = subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'], input=remote.encode(),capture_output=True)
    (local_out/'ssh_stdout.json').write_bytes(response.stdout)
    (local_out/'ssh_stderr.txt').write_bytes(response.stderr)
    if response.returncode:
        raise RuntimeError(response.stderr.decode(errors='replace'))
    payload=json.loads(response.stdout)
    (local_out/'verify_loader_equivalence.py').write_text(code,encoding='utf-8')
    if payload['result']:
        (local_out/'result.json').write_text(payload['result'],encoding='utf-8')
    print(json.dumps({key:value for key,value in payload.items() if key not in ('result','stdout','stderr')}))
    if payload['exit_code']:
        print(payload['stderr'][-6000:])
        if payload['result']:print(payload['result'][-6000:])
        raise SystemExit(payload['exit_code'])

if __name__=='__main__':main()
