import json
from pathlib import Path
D=Path(__file__).resolve().parent
x=json.loads((D/'dataset_model_inventory.json').read_text(encoding='utf-8-sig'))
out={}
for name,mods,nc in [('dronevehicle',['rgb','infrared'],5),('llvip',['visible','infrared'],1),('vedai',['rgb','ir'],8)]:
    d=x['datasets'][name]
    out[name]={'images':[d['images'][s]['path'] for s in mods],
               'labels':[d['labels'][s]['path'] for s in mods],
               'checkpoints':[d['models'][s]['weights']['last.pt']['path'] for s in mods],
               'expected_train_data':[d['models'][s]['parsed_args']['data'] for s in mods],
               'nc':nc,'role':d['probe_split'],'modalities':mods}
(D/'probe_config.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
