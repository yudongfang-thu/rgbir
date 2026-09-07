"""Complete a partial Windows scp collection; existing files must be byte exact."""
from pathlib import Path,PurePosixPath
import json,subprocess,tarfile
ROOT=Path(__file__).resolve().parent
REMOTE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_evidence_priority_20260907/natural_flow_attempt2'
DEST=ROOT/'remote_completed_attempt2'
proc=subprocess.Popen(['ssh','94','tar','-C',REMOTE,'-czf','-','.'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
records=[]
with tarfile.open(fileobj=proc.stdout,mode='r|gz') as archive:
    for member in archive:
        relative=PurePosixPath(member.name)
        if relative.is_absolute() or '..' in relative.parts or member.issym() or member.islnk():raise ValueError('Unsafe member')
        target=Path('\\\\?\\'+str(DEST.joinpath(*relative.parts).absolute()))
        if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
        if not member.isfile():raise ValueError('Unexpected archive entry')
        raw=archive.extractfile(member).read()
        if target.exists():
            assert target.read_bytes()==raw,str(target)
            disposition='existing_exact'
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(raw)
            disposition='new_missing_file'
        records.append(dict(relative_path=str(relative),bytes=len(raw),disposition=disposition))
stderr=proc.stderr.read();code=proc.wait()
if code:raise RuntimeError(stderr.decode(errors='replace'))
assert json.loads((DEST/'queue_attempt1/completion.json').read_text())['status']=='COMPLETED'
receipt=dict(status='COLLECTED_BYTE_EXACT',remote=REMOTE,local=str(DEST),files=len(records),records=records,
  partial_scp_failure='Windows scp failed to create a deeply nested reviewed source directory; missing files copied through archive stream, existing files byte verified and unchanged',new_hashes=False)
(ROOT/'collection_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='records'}))
