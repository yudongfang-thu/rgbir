import json
import os
from pathlib import Path
import sys
import time

root = Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
sys.path.insert(0, str(root))
from tools.project_resource_guard import require_bound_lease_from_environment

lease = require_bound_lease_from_environment()
import torch
from PIL import Image

torch.cuda.set_per_process_memory_fraction(0.02, 0)
t0 = time.perf_counter()
a = torch.ones((256, 256), device='cuda', requires_grad=True)
for _ in range(16):
    b = (a @ a).mean()
    b.backward()
    assert torch.isfinite(a.grad).all().item()
    a.grad = None
torch.cuda.synchronize()
assert b.item() == 256.0
image_path = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/data/raw/LLVIP/visible/train/010001.jpg')
with Image.open(image_path) as image:
    image.load()
    image_size = image.size
out = {'status': 'CUDA_AND_DATA_READ_PASS', 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
       'device': torch.cuda.get_device_name(0), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
       'lease_id': os.environ.get('JSTARS_RESOURCE_LEASE_ID'), 'forward_backward_iterations': 16,
       'elapsed_seconds': time.perf_counter()-t0, 'torch_peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
       'torch_peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
       'source_image': str(image_path), 'decoded_size': image_size,
       'scope': 'One dynamically assigned GPU; short functional check, not an all-GPU stability test; no training resumed'}
Path(sys.argv[1]).write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out))
