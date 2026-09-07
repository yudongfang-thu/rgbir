import concurrent.futures
import datetime
import json
from pathlib import Path
import urllib.request

BASE = Path(r'E:/SHARE/光sar/08_实验日志/2026-09-07_train_IndependentKD实施')
OUTPUT = BASE / 'external_baseline_availability.json'
REPOS = {'BCKD': ('TinyTigerPan/BCKD', 'main'), 'FGD': ('yzd-v/FGD', 'master'), 'LD': ('HikariTJU/LD', 'main')}

def get(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'RGBIR-research-read-only'})
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.load(r)
    except Exception as e:
        return {'error': str(e)}

def probe(item):
    name, (repo, branch) = item
    api = 'https://api.github.com/repos/' + repo
    endpoints = {'repository': api, 'forks': api + '/forks?per_page=100',
                 'releases': api + '/releases?per_page=100', 'issues': api + '/issues?state=all&per_page=100',
                 'tree': api + '/git/trees/' + branch + '?recursive=1'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        raw = dict(zip(endpoints, pool.map(get, endpoints.values())))
    result = {'repository': repo, 'branch': branch, 'queried_urls': endpoints}
    meta = raw['repository']
    result['metadata'] = {k: meta.get(k) for k in ('html_url', 'default_branch', 'archived', 'forks_count', 'open_issues_count', 'pushed_at')}
    for field in ('forks','releases','issues'):
        rows = raw[field]
        if isinstance(rows, dict):
            result[field] = rows
        elif field == 'forks':
            result[field] = [{k:r.get(k) for k in ('full_name','html_url','default_branch','pushed_at')} for r in rows]
        elif field == 'issues':
            result[field] = [{k:r.get(k) for k in ('number','title','state','html_url')} for r in rows if 'pull_request' not in r]
        else:
            result[field] = [{k:r.get(k) for k in ('name','tag_name','html_url')} for r in rows]
    tree = raw['tree']
    if isinstance(tree, dict) and tree.get('tree'):
        paths = [r['path'] for r in tree['tree'] if r.get('type') == 'blob']
        result['core_candidates'] = [p for p in paths if p.endswith('.py') and any(s in p.lower() for s in ('bckd','fgd','ld_head','ld_gfl','distillation','kd_loss','knowledge_distillation','atss_assigner'))]
        result['paper_or_supplement_files'] = [p for p in paths if p.lower().endswith('.pdf') or 'supp' in p.lower()]
    else:
        result['tree_error'] = tree.get('error', 'No tree')
    return name, result

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results = dict(pool.map(probe, REPOS.items()))
record = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'Public author repository metadata only; no weights, code execution, GPU or hashes', 'repositories': results}
with OUTPUT.open('x', encoding='utf-8') as f:
    json.dump(record, f, indent=2, ensure_ascii=False)
for name,r in results.items():
    print(name, json.dumps(r, ensure_ascii=False))
