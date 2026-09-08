"""Prepare a distinct receipt-only L3 analyzer before its first AP exists."""
from pathlib import Path
root=Path(__file__).parent;out=root/'analysis';out.mkdir(exist_ok=True)
s=(root.parent/'2026-09-08_probe_快速方向筛选/newentry/analysis/analyze_direction.py').read_text(encoding='utf-8')
s=s.replace("ARMS = {'drone': ('N','C1','C2','F-rel'), 'llvip': ('N','L2-box','L2-GT')}","ARMS = {'llvip': ('N','L3-DFL','L3-GT')}")
s=s.replace("POPULATIONS = {'drone': (1469,22462,5), 'llvip': (2406,7879,1)}","POPULATIONS = {'llvip': (2406,7879,1)}")
s=s.replace("CONTRASTS = {'drone': (('C1','N'),('C2','N'),('F-rel','N'),('C2','C1'),('F-rel','C1')),\n             'llvip': (('L2-box','N'),('L2-GT','N'),('L2-box','L2-GT'))}","CONTRASTS = {'llvip': (('L3-DFL','N'),('L3-GT','N'),('L3-DFL','L3-GT'))}")
s=s.replace('DIRECTION_','OBJECT_DFL_').replace('L2-box','L3-DFL').replace('L2-GT','L3-GT')
s=s.replace('analyze_direction_source.py','analyze_object_dfl_source.py')
with (out/'analyze_object_dfl.py').open('x',encoding='utf-8') as f:f.write(s)
print('Prepared new analyzer; no AP read')
