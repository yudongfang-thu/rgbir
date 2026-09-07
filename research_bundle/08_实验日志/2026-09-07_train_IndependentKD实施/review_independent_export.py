"""Bounded publication review: source bytes, excluded payloads and local links."""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote
import zipfile

ROOT=Path(__file__).resolve().parent
REPO=Path('\\\\?\\C:\\Users\\MSI-PC\\rgbir_review_worktrees\\evidence-20260906')
SAFETY=ROOT.parent/'2026-09-06_ops_GitHub完整审计包/review_publication_safety.py'
spec=importlib.util.spec_from_file_location('publication_safety',SAFETY)
safety=importlib.util.module_from_spec(spec);spec.loader.exec_module(safety)

def markdown_links(path, repo):
    """Same link rules, with Windows extended paths normalized before stat."""
    result=[]
    for number,line in enumerate(path.read_text(encoding='utf-8-sig',errors='replace').splitlines(),1):
        for match in re.finditer(r'!?\[[^\]\n]*\]\((<[^>]+>|[^)]+)\)',line):
            link=match.group(1).strip().strip('<>')
            if link.startswith(('https://','http://','mailto:','#','app://','codex://')):continue
            link=unquote(link.split('#',1)[0].split('?',1)[0])
            if not link:continue
            if re.match(r'^[A-Za-z]:[/\\]',link) or link.startswith(('/mnt/','/private/','file:')):
                result.append(dict(path=safety.rel(path,repo),line=number,rule='absolute_link',target=link));continue
            target=(repo/link.lstrip('/')) if link.startswith('/') else (path.parent/link)
            raw_target=str(target)
            # Python 3.8 ntpath intentionally skips normalization for the
            # extended prefix; normalize the drive path before restoring it.
            target=Path('\\\\?\\'+os.path.normpath(raw_target[4:])) if raw_target.startswith('\\\\?\\') else Path(os.path.normpath(raw_target))
            try:exists=target.exists()
            except OSError:exists=False
            if not exists:result.append(dict(path=safety.rel(path,repo),line=number,rule='missing_relative_target',target=link))
    return result

p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
inventory=json.loads(a.manifest.read_text(encoding='utf-8'))
findings=[];mismatches=[];missing=[];links=[];large=[]
for row in inventory['files']:
    source=Path(row['source']);target=REPO/row['repository_path']
    if not source.is_file() or not target.is_file():missing.append(row['repository_path']);continue
    if source.suffix.lower()!='.md' and source.read_bytes()!=target.read_bytes():mismatches.append(row['repository_path'])
    findings.extend(safety.scan_file(source,'source/'+row['path']))
    findings.extend(safety.scan_file(target,'stage/'+row['repository_path']))
    if target.stat().st_size>20*2**20:large.append(row['repository_path'])
    if target.suffix.lower()=='.zip':
        # Inspect the user-supplied method package without extracting paths.
        with zipfile.ZipFile(target) as archive,tempfile.TemporaryDirectory() as folder:
            for index,item in enumerate(archive.infolist()):
                if item.is_dir():continue
                suffix=Path(item.filename).suffix
                local=Path(folder)/(str(index)+suffix);local.write_bytes(archive.read(item))
                findings.extend(safety.scan_file(local,'zip/'+row['repository_path']+'::'+item.filename))
files=[x for x in REPO.rglob('*') if x.is_file() and '.git' not in x.relative_to(REPO).parts]
for path in files:
    if path.suffix.lower()=='.md':links.extend(markdown_links(path,REPO))
for name in ('README.md','LATEST_RESULTS.md','INDEPENDENT_KD_REVIEW_PROMPT.md'):
    findings.extend(safety.scan_file(REPO/name,'navigation/'+name))
result=dict(created_at=datetime.datetime.now().astimezone().isoformat(),manifest=str(a.manifest),
    exported_files=len(inventory['files']),source_and_export_raw_mismatches=mismatches,missing=missing,
    candidate_findings=findings,markdown_link_findings=links,files_over20MiB=large,
    status='ACCEPTED' if not any((findings,mismatches,missing,links,large)) else 'REVIEW_REQUIRED',
    policy='No credential values printed; raw code/numbers/images remain byte-identical to local sources; markdown links adapted; package text members inspected.')
out=ROOT/('publication_review_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],output=str(out),files=result['exported_files'],
    findings=len(findings),raw_mismatches=len(mismatches),missing=len(missing),links=len(links),large=len(large))))
