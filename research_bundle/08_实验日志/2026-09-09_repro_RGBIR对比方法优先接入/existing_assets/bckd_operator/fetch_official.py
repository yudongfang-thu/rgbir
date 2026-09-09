"""Freeze three public author files to one existing Git commit; no new digest."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = 'TinyTigerPan/BCKD'
COMMIT = '121c9aa272c9195c0f05cd12727bfe57636a5005'
FILES = ['mmdet/models/losses/kd_loss.py',
         'configs/bckd/bckd_r50_gflv1_r101_fpn_coco_1x.py',
         'mmdet/models/dense_heads/ld_head.py']


def get(url):
    return subprocess.run(['curl.exe', '--fail', '--silent', '--show-error',
                           '--location', '--connect-timeout', '10', '--max-time', '45',
                           url], check=True, capture_output=True).stdout


if __name__ == '__main__':
    receipt_path = ROOT / 'SOURCE_RECEIPT.json'
    if receipt_path.exists():
        raise RuntimeError('Source receipt already exists; do not overwrite the frozen source')
    # GitHub API returned HTTP 403 rate limit; git ls-remote read the existing
    # refs/heads/main identifier. The downloaded source is pinned, not mutable main.
    commit = COMMIT
    assert len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
    rows = []
    for relative in FILES:
        url = f'https://raw.githubusercontent.com/{REPO}/{commit}/{relative}'
        raw = get(url)
        target = ROOT / 'official' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError(f'Refusing source overwrite: {target}')
        target.write_bytes(raw)
        assert target.read_bytes() == raw
        rows.append({'repository_path': relative, 'url': url,
                     'local_path': str(target.relative_to(ROOT)).replace('\\', '/'),
                     'bytes': len(raw), 'download_write_readback_exact': True})
    receipt = {'status': 'OFFICIAL_SOURCE_FROZEN', 'repository': REPO,
               'resolved_ref': 'main', 'commit': commit,
               'ref_resolution': 'git ls-remote https://github.com/TinyTigerPan/BCKD.git refs/heads/main',
               'prior_metadata_attempt': {'url': f'https://api.github.com/repos/{REPO}/commits/main',
                                          'status': 'HTTP_403_RATE_LIMIT', 'no_source_written': True},
               'prior_raw_transport_attempt': 'urllib connection reset before source write; curl transport used',
               'downloaded_at_utc': datetime.now(timezone.utc).isoformat(), 'files': rows,
               'new_hash_computed': False, 'git_identifier_is_preexisting': True,
               'code_executed_during_download': False, 'weights_downloaded': False}
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'commit': commit,
                      'bytes': sum(r['bytes'] for r in rows)}))
