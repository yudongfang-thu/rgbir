"""Prepare/mirror this entry's small evidence; publish means worktree copy, never commit/push.

Nothing executes on import. Root chooses an explicit RUNNING/COMPLETED/BLOCKED
state after updating the entry README. Other experiment banners are preserved.
"""
import argparse
import base64
import gzip
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent.parent
REPO=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
BRANCH='research/full-evidence-20260906'
REMOTE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_dfl_screen_20260908/review_v1'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
MARKER='[probe_对象坐标分布目标]('
CHECKS='publication_checks/update_20260908_object_dfl'
BANNER_PREFIX='> **2026-09-08 对象坐标DFL短筛'
TEXT_EXTENSIONS={'.py','.md','.json','.jsonl','.yaml','.yml','.csv','.tsv','.txt','.diff'}
MAX_BYTES=16_000_000
EXCLUDED_NAMES={'REMOTE_MIRROR_RECEIPT.json','GITHUB_PUBLICATION_RECEIPT.json','SYNC_SCOPE_MANIFEST.json',
    'snapshot.json','host_profile.json','host_snapshot.json','all_processes.json','process_snapshot.json',
    'resource_profile.json','admission.json','gpu_snapshot.json','host_status.json'}
CREDENTIAL_MARKER=re.compile(rb'BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}')


def write_new(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def collect(root=ROOT):
    files=[];excluded=[]
    candidates=[];resolved_root=root.resolve()
    for directory,dirs,names in os.walk(root,topdown=True,followlinks=False):
        safe=[]
        for name in sorted(dirs):
            child=Path(directory)/name
            if (child.is_symlink() or resolved_root not in child.resolve().parents
                    or name.lower() in ('__pycache__','.git','weights','credentials','node_modules','.venv')):
                excluded.append(dict(path=child.relative_to(root).as_posix()+'/',reason='excluded_or_external_directory'))
            else:safe.append(name)
        dirs[:]=safe
        candidates.extend(Path(directory)/name for name in names)
    for path in sorted(candidates):
        if not path.is_file():continue
        rel=path.relative_to(root);lower=[p.lower() for p in rel.parts];name=path.name.lower()
        inner_name=name[:-3] if name.endswith('.gz') else name
        compressed_json=name.endswith(('.json.gz','.jsonl.gz'))
        reason=None
        if path.is_symlink() or root.resolve() not in path.resolve().parents:reason='symlink_or_outside_entry'
        elif any(p in ('__pycache__','.git','weights','credentials','node_modules','.venv') for p in lower):reason='excluded_directory'
        elif inner_name in {n.lower() for n in EXCLUDED_NAMES}:reason='generated_sync_receipt_or_full_host_snapshot'
        elif inner_name.endswith(('_resource_profile.json','_admission.json')):reason='full_resource_or_admission_profile'
        elif any(s in name for s in ('credential','id_rsa','id_ed25519','private_key')):reason='credential_named_file'
        elif path.suffix.lower() not in TEXT_EXTENSIONS and not compressed_json:reason='not_small_evidence_type'
        if reason:
            excluded.append(dict(path=rel.as_posix(),reason=reason));continue
        before=path.stat();data=path.read_bytes();after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed while reading: '+str(rel))
        if len(data)>MAX_BYTES:raise ValueError('Large evidence requires explicit curation: '+str(rel))
        scan=data
        if compressed_json:
            with gzip.GzipFile(fileobj=io.BytesIO(data),mode='rb') as stream:scan=stream.read(32_000_001)
            if len(scan)>32_000_000:raise ValueError('Compressed object evidence exceeds bounded inspection size: '+str(rel))
        if CREDENTIAL_MARKER.search(scan):raise ValueError('Credential marker: '+str(rel))
        files.append(dict(relative_path=rel.as_posix(),bytes=len(data),mtime_ns=after.st_mtime_ns,data=data))
    if not any(row['relative_path']=='README.md' for row in files):raise ValueError('Entry README missing')
    return files,excluded


def public_rows(files):
    return [{k:v for k,v in row.items() if k!='data'} for row in files]


def plan(files,excluded,state,note):
    return dict(scope='OBJECT_DFL_EVIDENCE_PUBLICATION',execution_state=state,state_note=note,
        entry=ROOT.name,files=public_rows(files),excluded=excluded,total_bytes=sum(row['bytes'] for row in files),
        new_hash_computed=False,weights_uploaded=False,images_uploaded=False,
        full_host_profiles_excluded=True,credential_marker_scan_passed=True,
        prior_failures_and_reviews_retained=True,new_training_or_evaluation=False)


def mirror(files,excluded,args):
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w:gz') as archive:
        for row in files:
            member=tarfile.TarInfo(row['relative_path']);member.size=row['bytes']
            archive.addfile(member,io.BytesIO(row['data']))
    code='''from pathlib import Path
import sys,tarfile,json
root=Path(%r);root.mkdir(parents=True,exist_ok=False);rows=[]
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as archive:
 for member in archive:
  path=root/member.name
  if not member.isfile() or root not in path.resolve().parents:raise ValueError('Invalid artifact path')
  data=archive.extractfile(member).read();path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(data)
  if path.read_bytes()!=data:raise ValueError('Mirror bytes differ')
  rows.append(dict(relative_path=member.name,bytes=len(data),byte_exact=True))
receipt=dict(status='OBJECT_DFL_EVIDENCE_MIRRORED',destination=str(root),files=rows,new_hash_computed=False)
with (root/'mirror_receipt.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2)
print(json.dumps(receipt,ensure_ascii=False))
''' % REMOTE
    encoded=base64.b64encode(code.encode('utf-8')).decode('ascii')
    command="import base64;exec(base64.b64decode('"+encoded+"'))"
    result=subprocess.run(['ssh','94',PY,'-c',shlex.quote(command)],input=payload.getvalue(),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    receipt=json.loads(result.stdout)
    receipt.update(local_source_snapshot=plan(files,excluded,args.state,args.note))
    write_new(ROOT/'REMOTE_MIRROR_RECEIPT.json',receipt)
    print(receipt['status'],len(files),args.state)


def amend_text(path,transform):
    original=path.read_bytes();newline='\r\n' if b'\r\n' in original else '\n'
    lines=original.decode('utf-8').splitlines();updated=transform(lines)
    if path.read_bytes()!=original:raise ValueError('Concurrent edit detected: '+str(path))
    path.write_bytes((newline.join(updated)+newline).encode('utf-8'))


def publish(files,excluded,args):
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if branch!=BRANCH:raise ValueError('Wrong existing publication branch')
    mirror_path=ROOT/'REMOTE_MIRROR_RECEIPT.json'
    receipt=json.loads(mirror_path.read_text(encoding='utf-8'))
    expected=receipt['local_source_snapshot']
    if expected['files']!=public_rows(files) or expected['execution_state']!=args.state or expected['state_note']!=args.note:
        raise ValueError('Source/state changed since mirror; root must freeze a new review snapshot')
    destination=REPO/'research_bundle/08_实验日志'/ROOT.name
    dirty=subprocess.check_output(['git','status','--porcelain','--',str(destination),str(REPO/CHECKS)],cwd=REPO,text=True)
    if dirty.strip():raise ValueError('This entry/checks already has local modifications; preserve and review them first')
    copied=[]
    for row in files+[dict(relative_path=mirror_path.name,bytes=mirror_path.stat().st_size,data=mirror_path.read_bytes())]:
        path=destination/row['relative_path'];path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(row['data'])
        if path.read_bytes()!=row['data']:raise ValueError('Published bytes differ: '+row['relative_path'])
        copied.append(dict(path=row['relative_path'],bytes=len(row['data']),byte_exact=True))
    source_rows=[line for line in (WORKSPACE/'08_实验日志/README.md').read_text(encoding='utf-8').splitlines() if MARKER in line]
    if len(source_rows)!=1:raise ValueError('Expected one current local index row')
    def update_index(lines):
        matches=[i for i,line in enumerate(lines) if MARKER in line]
        if len(matches)>1:raise ValueError('Duplicate entry index markers')
        if matches:lines[matches[0]]=source_rows[0]
        else:lines.insert(next(i for i,line in enumerate(lines) if line.startswith('| 2026-09-08 |')),source_rows[0])
        return lines
    amend_text(REPO/'research_bundle/08_实验日志/README.md',update_index)
    states={'RUNNING':'执行中','COMPLETED':'本阶段已完成','BLOCKED':'本阶段已停止或阻塞'}
    banner=(BANNER_PREFIX+'：'+states[args.state]+'**：'+args.note+' '
        '原失败attempt与独立审阅保留；当前状态和证据边界见[条目README]'
        '(research_bundle/08_实验日志/'+ROOT.name+'/README.md)。')
    def update_banner(lines):
        # Replace only this entry's own exact marker, never C1 or any other banner.
        matches=[i for i,line in enumerate(lines) if line.startswith(BANNER_PREFIX)]
        if len(matches)>1:raise ValueError('Duplicate object-DFL banners')
        if matches:lines[matches[0]]=banner
        else:lines=[banner,'']+lines
        return lines
    amend_text(REPO/'README.md',update_banner)
    checks=REPO/CHECKS;checks.mkdir(parents=True,exist_ok=True)
    manifest=plan(files,excluded,args.state,args.note);manifest['files']=copied
    (checks/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (checks/'README.md').write_text('# 本次发布范围\n\n仅对象坐标分布目标条目的小源码、原始小产物、CPU分析、失败attempt及独立审阅，逐字节复制；排除权重、原图、凭据、pycache及完整主机资源/进程快照。\n\n'
        '发布状态：'+states[args.state]+'。'+args.note+'\n\n'
        '只更新本条目索引和独有banner，保留其他线程C1提速banner；本脚本不stage、commit或push，由root完成。未计算新hash。\n',encoding='utf-8')
    print('OBJECT_DFL_EVIDENCE_WORKTREE_PREPARED',len(copied),args.state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('manifest','mirror','publish'))
    parser.add_argument('--state',choices=('RUNNING','COMPLETED','BLOCKED'),required=True)
    parser.add_argument('--note',required=True,help='Brief accurate current-stage wording; no inferred completion/AP claims')
    args=parser.parse_args()
    if not args.note.strip() or '\n' in args.note or '\r' in args.note:raise ValueError('One explicit state sentence required')
    files,excluded=collect()
    if args.mode=='manifest':
        write_new(ROOT/'SYNC_SCOPE_MANIFEST.json',plan(files,excluded,args.state,args.note))
        print('OBJECT_DFL_EVIDENCE_MANIFEST_PREPARED',len(files),args.state)
    elif args.mode=='mirror':mirror(files,excluded,args)
    else:publish(files,excluded,args)


if __name__=='__main__':main()
