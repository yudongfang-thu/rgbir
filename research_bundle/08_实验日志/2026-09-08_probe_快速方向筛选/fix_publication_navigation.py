from pathlib import Path
import subprocess
repo=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
name='research_bundle/08_实验日志/README.md'
old=subprocess.check_output(['git','show','HEAD:'+name],cwd=repo)
nl=b'\r\n' if b'\r\n' in old else b'\n'
row='| 2026-09-08 | [probe_快速方向筛选](2026-09-08_probe_快速方向筛选/README.md) | probe | 六个已准入短训/full dev共32m；Drone C2未扩大C1收益、LLVIP L2未超过GT，原F因系数约束blocked；另冻结LLVIP C0与F-GM独立短探针 |'.encode()
if b'2026-09-08_probe_' not in old:
    i=old.index(b'| 2026-09-08 |')
    (repo/name).write_bytes(old[:i]+row+nl+old[i:])
print('Navigation restored with original line endings; original body unchanged')
