from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from concurrent.futures import ThreadPoolExecutor
import json

root=Path(__file__).resolve().parent
urls={
 'M2D-LIF/readme.md':'https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/readme.md',
 'M2D-LIF/environment.yaml':'https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/environment.yaml',
 'M2D-LIF/data/DroneVehicle.yaml':'https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/data/DroneVehicle.yaml',
 'M2D-LIF/train_dist_obb.py':'https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/train_dist_obb.py',
 'M2D-LIF/val_obb.py':'https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/val_obb.py',
 'M2D-LIF/model_yaml_obb/yolov8_LIF_obb.yaml':'https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/model_yaml_obb/yolov8_LIF_obb.yaml',
 'M2D-LIF/ultralytics/models/yolo/obb/train.py':'https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/ultralytics/models/yolo/obb/train.py',
 'M2D-LIF/ultralytics/models/yolo/obb/val.py':'https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/ultralytics/models/yolo/obb/val.py',
 'M2D-LIF/ultralytics/data/dataset.py':'https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/ultralytics/data/dataset.py',
 'C2Former/README.md':'https://raw.githubusercontent.com/yuanmaoxun/C2Former/main/README.md',
 'C2Former/configs/s2anet/s2anet_c2former_fpn_1x_dota_le135.py':'https://raw.githubusercontent.com/yuanmaoxun/C2Former/main/configs/s2anet/s2anet_c2former_fpn_1x_dota_le135.py',
 'C2Former/configs/_base_/datasets/dronevehicle.py':'https://raw.githubusercontent.com/yuanmaoxun/C2Former/main/configs/_base_/datasets/dronevehicle.py',
 'C2Former/configs/_base_/schedules/schedule_2x.py':'https://raw.githubusercontent.com/yuanmaoxun/C2Former/main/configs/_base_/schedules/schedule_2x.py',
 'C2Former/mmrotate/datasets/dronevehicle.py':'https://raw.githubusercontent.com/yuanmaoxun/C2Former/main/mmrotate/datasets/dronevehicle.py',
 'C2Former/mmrotate/__init__.py':'https://raw.githubusercontent.com/yuanmaoxun/C2Former/main/mmrotate/__init__.py',
 'InfraredPrivilegedUAV/page.html':'https://github.com/chenbys/InfraredPrivilegedUAV',
}
def fetch(item):
    name,url=item;row=dict(path=name,url=url)
    try:
        with urlopen(Request(url,headers={'User-Agent':'RGBIR-public-source-audit'}),timeout=25) as r:
            raw=r.read();row.update(status=r.status,bytes=len(raw))
        out=root/'public_sources'/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
    except HTTPError as e:row.update(status=e.code,error=str(e))
    except Exception as e:row.update(status=None,error=repr(e))
    return row
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,urls.items()))
(root/'RAW_FETCH_RECEIPT.json').write_text(json.dumps(dict(sources=rows,new_hashes_computed=False,source_code_executed=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows,ensure_ascii=False,indent=2))
