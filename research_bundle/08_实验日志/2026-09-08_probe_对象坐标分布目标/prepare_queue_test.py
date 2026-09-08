"""Execute only the preparation script's new test-copy section; queue already exists."""
from pathlib import Path
root=Path(__file__).parent
s=(root/'prepare_queue.py').read_text(encoding='utf-8')
start=s.index("test=(source.parent/")
stop=s.index("print('New L3 queue",start)
source=root.parent/'2026-09-08_probe_快速方向筛选/newentry/release/run_direction_queue.py'
dest=root/'trainer_release'
exec(compile(s[start:stop],str(root/'prepare_queue.py'),'exec'))
