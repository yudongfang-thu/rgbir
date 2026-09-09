import json
import sys
from pathlib import Path
import torch
x = torch.tensor([-1, 0, 3, 8], dtype=torch.int64)
gain = torch.tensor(5.0)
out = {'torch': torch.__version__, 'case': 'author long grid indices with float shape-derived clamp bound'}
try:
    x.clone().clamp_(0, gain - 1)
    out['original'] = 'success'
except Exception as e:
    out['original'] = repr(e)
out['integer_bounds'] = x.clamp(0, 4).tolist()
out['float_then_long_reference'] = x.float().clamp(0, gain - 1).long().tolist()
out['cuda_initialized'] = torch.cuda.is_initialized()
Path(sys.argv[1]).write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out))
