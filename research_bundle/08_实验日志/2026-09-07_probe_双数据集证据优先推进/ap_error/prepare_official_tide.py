"""Download a pinned, unmodified author source snapshot and isolate dependencies."""
from pathlib import Path
import datetime, hashlib, io, json, urllib.request, zipfile

HERE = Path(__file__).resolve().parent
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'guangsar-cpu-tide-audit'})
    return urllib.request.urlopen(req, timeout=60).read()
def main():
    commit = json.loads(fetch('https://api.github.com/repos/dbolya/tide/commits/master'))['sha']
    raw = fetch('https://codeload.github.com/dbolya/tide/zip/' + commit)
    archive = zipfile.ZipFile(io.BytesIO(raw))
    root = HERE / 'official_tide'
    root.mkdir(exist_ok=False)
    files = []
    for member in archive.infolist():
        bits = Path(member.filename).parts[1:]
        if member.is_dir() or not bits: continue
        assert '..' not in bits
        rel = Path(*bits)
        if rel.parts[0] not in ('tidecv', 'README.md', 'LICENSE', 'setup.py', 'CHANGELOG.md'): continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = archive.read(member)
        path.write_bytes(data)
        files.append(dict(path=str(rel), sha256=hashlib.sha256(data).hexdigest()))
    receipt = dict(repository='https://github.com/dbolya/tide', commit=commit,
                   collected_at=datetime.datetime.now().astimezone().isoformat(),
                   archive_sha256=hashlib.sha256(raw).hexdigest(), files=files)
    (HERE / 'official_source_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    # Wheel extraction avoids modifying any environment's installed packages.
    deps = HERE / 'deps'
    for wheel in (HERE / 'wheels').glob('*.whl'):
        with zipfile.ZipFile(wheel) as z: z.extractall(deps)
    print(json.dumps({'commit': commit, 'files': len(files), 'deps':str(deps)}))
if __name__ == '__main__': main()
