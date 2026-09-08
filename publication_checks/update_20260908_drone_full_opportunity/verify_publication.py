"""Verify byte-preserving evidence publication without recalculating file hashes."""
from pathlib import Path
import json,re,subprocess

here=Path(__file__).parent
repo=here.parent.parent
workspace=Path('E:/SHARE/光sar')
reports=[]
for entry,check in [('2026-09-08_probe_训练侧定位覆盖','update_20260908_train_loc_coverage'),('2026-09-08_probe_Drone教师完整推理','update_20260908_drone_full_opportunity')]:
    manifest=json.loads((repo/'publication_checks'/check/'manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        rel=Path(row['path'])
        local=workspace/'08_实验日志'/entry/rel
        public=repo/'research_bundle/08_实验日志'/entry/rel
        a,b=local.read_bytes(),public.read_bytes()
        assert a==b and len(b)==row['bytes'],str(rel)
        assert not public.name.endswith('_resource_profile.json'),str(rel)
        assert public.suffix.lower() not in {'.pt','.pth','.jpg','.jpeg','.png','.key'},str(rel)
        if public.suffix.lower()!='.gz':
            assert re.search(rb'BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{30,}',b) is None,str(rel)
    reports.append(dict(entry=entry,byte_exact_files=len(manifest['files'])))
check=subprocess.run(['git','-c','core.whitespace=cr-at-eol','diff','--cached','--check'],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
lines=check.stdout.decode('utf-8').splitlines()
assert len(lines)==3 and all('paired_rgbir_data.py:219: new blank line at EOF.' in x for x in lines),lines
adjusted=subprocess.run(['git','-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check'],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
assert adjusted.returncode==0,(adjusted.stdout+adjusted.stderr).decode('utf-8')
receipt=dict(status='PUBLICATION_VERIFIED',byte_exact=reports,known_frozen_source_whitespace=lines,
    exception_reason='Three byte-exact copies of original pinned source contain an extra EOF blank line; preserved as evidence, not edited.',
    other_whitespace_check_passed=True,credential_marker_scan_passed=True,raw_host_profiles_excluded=True,new_hash_computed=False)
with (here/'verification_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print('PUBLICATION_VERIFIED',sum(x['byte_exact_files'] for x in reports),'byte-exact files')
