"""Review only this incremental export and its root navigation; never print secret candidates."""
from pathlib import Path
import importlib.util
import json
import subprocess

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
STAGE=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')

def main():
    spec=importlib.util.spec_from_file_location('publication_scan',WORKSPACE/'08_实验日志/2026-09-06_ops_GitHub完整审计包/review_publication_safety.py')
    scanner=importlib.util.module_from_spec(spec);spec.loader.exec_module(scanner)
    manifest_path=STAGE/'TASK_CONDITIONAL_BUNDLE_MANIFEST_20260907.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    paths=[STAGE/r['repository_path'] for r in manifest['files']]
    paths += [STAGE/name for name in ['README.md','LATEST_RESULTS.md','TASK_CONDITIONAL_STATUS_20260907.md','TASK_CONDITIONAL_REVIEW_PROMPT.md']]
    findings=[];links=[];byte_errors=[]
    for p in paths:
        if p.suffix=='.md':
            # Markdown is an adapted export, so normalize its line endings only.
            p.write_bytes(p.read_text(encoding='utf-8-sig').encode('utf-8'))
        findings.extend(scanner.scan_file(p,p.relative_to(STAGE).as_posix()))
        if p.suffix=='.md':links.extend(scanner.markdown_links(p,STAGE))
    for row in manifest['files']:
        dst=STAGE/row['repository_path'];src=Path(row['source'])
        row['bytes']=dst.stat().st_size
        if src.suffix!='.md' and dst.read_bytes()!=src.read_bytes():byte_errors.append(row['repository_path'])
    manifest_path.write_bytes(json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8'))
    result={'scope':'incremental export plus current navigation','files':len(paths),
            'candidate_findings':findings,'markdown_link_findings':links,
            'non_markdown_byte_mismatches':byte_errors,
            'credential_values_printed':False,'weights_or_credentials_included':any(
                p.suffix.lower() in scanner.DISALLOWED_SUFFIXES for p in paths)}
    (HERE/'export_review.json').write_bytes(json.dumps(result,ensure_ascii=False,indent=2).encode('utf-8'))
    print(json.dumps({'files':len(paths),'credential_candidates':len(findings),
                      'link_findings':len(links),'byte_mismatches':len(byte_errors)}))

if __name__=='__main__':main()
