"""Read public author repository metadata/files and local papers; no execution or hashing."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import json
import re

ROOT=Path(__file__).resolve().parent
def get(url):
    try:
        with urlopen(Request(url,headers={'User-Agent':'RGBIR-local-source-audit/1.0'}),timeout=25) as response:
            raw=response.read();return {'url':url,'status':response.status,'bytes':len(raw),'text':raw.decode('utf-8',errors='replace')}
    except HTTPError as error:return {'url':url,'status':error.code,'error':str(error)}
    except Exception as error:return {'url':url,'status':None,'error':repr(error)}

def save(name,obj):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def repo(name):
    slug=name.replace('/','__');meta=get('https://api.github.com/repos/'+name)
    if meta.get('status')!=200:save('primary/'+slug+'/metadata.json',meta);return meta
    metadata=json.loads(meta['text']);branch=metadata['default_branch']
    save('primary/'+slug+'/metadata.json',meta)
    tree=get('https://api.github.com/repos/'+name+'/git/trees/'+branch+'?recursive=1')
    save('primary/'+slug+'/tree.json',tree)
    rows=json.loads(tree.get('text','{}')).get('tree',[])
    wanted=[]
    for row in rows:
        path=row['path']
        if row['type']!='blob':continue
        if path.lower() in ('readme.md','requirements.txt') or ('configs/' in path and ('drone' in path.lower() or 'c2former' in path.lower() or 'm2d' in path.lower())):
            if row.get('size',0)<80000:wanted.append(path)
    for path in wanted:
        result=get('https://raw.githubusercontent.com/'+name+'/'+branch+'/'+path)
        save('primary/'+slug+'/files/'+path+'.response.json',result)
        if result.get('status')==200:
            dest=ROOT/'primary'/slug/'source'/path;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(result['text'],encoding='utf-8')
    return dict(repo=name,status=meta['status'],branch=branch,tree_files=len(rows),saved_source_files=wanted)

if __name__=='__main__':
    names=['chenbys/InfraredPrivilegedUAV','Zhao-Tian-yi/M2D-LIF','yuanmaoxun/C2Former']
    with ThreadPoolExecutor(max_workers=3) as pool:result=list(pool.map(repo,names))
    save('PRIMARY_FETCH_RECEIPT.json',{'sources':result,'new_hashes_computed':False,'source_code_executed':False})
    print(json.dumps(result,ensure_ascii=False,indent=2))
