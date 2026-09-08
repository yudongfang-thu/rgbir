# coding: utf-8
"""Deploy the small authorized CPU builder into new artifacts; never starts CUDA."""
import base64
import datetime
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PYTHON = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
PROJECT = '/mnt/dataset/yudongfang/projects/RGBT_campaign'
PARENT = PROJECT + '/artifacts/rgbir_hourly_screen_20260908'
OUTPUT = PARENT + '/subset_v1'
PREPARED = PROJECT + '/artifacts/rgbt_p3_causal_v1/prepared/dronevehicle'
ARGS = ['--student-yaml', PREPARED + '/rgb.data.yaml', '--privileged-yaml', PREPARED + '/infrared.data.yaml',
        '--mapping', PREPARED + '/mappings/rgb_to_infrared_train.json',
        '--groups', PROJECT + '/data/processed/dronevehicle/yolo/hbb_v1/rgb_train_source_groups.tsv',
        '--output', OUTPUT, '--size', '2048', '--seed', '20260908',
        '--rgb-checkpoint', PROJECT + '/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/rgb_seed42_native_b32a2/weights/last.pt',
        '--ir-checkpoint', PROJECT + '/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/infrared_seed42_native_b32a2/weights/last.pt']


def main():
    source = (HERE / 'build_subset.py').read_bytes()
    remote = """import base64,json,subprocess,sys
from pathlib import Path
source=base64.b64decode(SOURCE)
parent=Path(PARENT)
parent.mkdir(parents=True,exist_ok=True)
script=parent/'subset_builder_v1.py'
if script.exists():
    assert script.read_bytes()==source,'Existing builder differs; preserve it'
else:
    with script.open('xb') as f:f.write(source)
assert script.read_bytes()==source
subprocess.run([sys.executable,str(script),'--self-test'],check=True)
subprocess.run([sys.executable,str(script),*ARGS],check=True)
""".replace('SOURCE', repr(base64.b64encode(source).decode())).replace('PARENT', repr(PARENT)).replace('ARGS', repr(ARGS))
    run = subprocess.run(['ssh', '94', PYTHON, '-'], input=remote.encode('utf-8'), capture_output=True, timeout=180)
    receipt = {'recorded_at': datetime.datetime.now().astimezone().isoformat(), 'returncode': run.returncode,
               'stdout': run.stdout.decode('utf-8'), 'stderr': run.stderr.decode('utf-8'),
               'remote_output': OUTPUT, 'source_bytes': len(source), 'arguments': ARGS,
               'scope': 'Authorized CPU train-label metadata subset creation; no original input changes, images, GPU, predictions or hashes.'}
    target = HERE / 'remote_build_receipt.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_bytes((json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(receipt, ensure_ascii=False))
    run.check_returncode()


if __name__ == '__main__':
    main()
