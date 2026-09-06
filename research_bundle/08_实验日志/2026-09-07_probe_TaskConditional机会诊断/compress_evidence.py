"""Make new server gzip evidence copies and compare decoded bytes directly."""
import base64
import gzip
import json
import subprocess
from pathlib import Path
from collect_diagnostics import BASE, HERE, PY, RUNS


def main():
    code='import base64,gzip,json,datetime\nfrom pathlib import Path\n'
    code+=f'base=Path({BASE!r})\nruns={RUNS!r}\n'
    code+='''target=base/'evidence_compressed_v1'
target.mkdir(exist_ok=False)
files={};manifest=[]
for key,folder in runs.items():
 for name in ['d1_objects.jsonl','d2_anchors.jsonl','images.jsonl']:
  source=base/folder/name
  original=source.read_bytes()
  packed=gzip.compress(original,compresslevel=6,mtime=0)
  assert gzip.decompress(packed)==original
  output=target/key/(name+'.gz');output.parent.mkdir(exist_ok=True);output.write_bytes(packed)
  files[key+'/'+name+'.gz']=base64.b64encode(packed).decode()
  manifest.append({'source':str(source),'gzip':str(output),'local':key+'/'+name+'.gz','source_bytes':len(original),
   'gzip_bytes':len(packed),'server_decompressed_bytes_equal_original':True})
print(json.dumps({'created_at':datetime.datetime.now().astimezone().isoformat(),'manifest':manifest,'files':files}))
'''
    response=subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=code.encode(),capture_output=True)
    if response.returncode:raise RuntimeError(response.stderr.decode(errors='replace'))
    payload=json.loads(response.stdout)
    for relative,encoded in payload.pop('files').items():
        destination=HERE/relative
        original=destination.with_suffix('').read_bytes()
        packed=base64.b64decode(encoded)
        if gzip.decompress(packed)!=original:raise AssertionError('Decoded remote gzip differs from local original bytes: '+relative)
        if destination.exists():raise FileExistsError(destination)
        destination.write_bytes(packed)
    payload['local_decompressed_bytes_equal_previously_downloaded_original']=True
    payload['hashes_used']=False
    (HERE/'gzip_evidence_manifest.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'files':len(payload['manifest']),'raw_bytes':sum(x['source_bytes'] for x in payload['manifest']),
          'gzip_bytes':sum(x['gzip_bytes'] for x in payload['manifest']),'direct_bytes_verified_both_sides':True}))


if __name__=='__main__':main()
