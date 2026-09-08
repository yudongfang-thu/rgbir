from pathlib import Path
import sys
B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign');A=B/'artifacts'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
sys.path.insert(0,str(A/'rgbir_task_conditional_v1_20260907/release_v8'))
import resource_dispatch
root=A/'rgbir_direction_screen_20260908';out=root/'initial_llvip_recheck_attempt1';out.mkdir(exist_ok=False)
dest=out/'evaluation'
cmd=[PY,str(root/'initial_eval_release_v1/initial_baseline_eval.py'),'--reference-dir',str(A/'rgbir_independent_kd_v2_20260907/release_gpu5'),'--config',str(root/'release_v1/configs/llvip_native_evaluation.yaml'),'--checkpoint',str(B/'runs/rgbt_p3_causal_v1/formal_native/llvip/visible_seed42_native_b32a2/weights/last.pt'),'--output',str(dest)]
resource_dispatch.run_job(dict(id='direction_llvip_initial_eval',arm='initial',stage='eval',kind='eval',formal=False,vram_mib=2048,rss_mib=8192,command=cmd,expected_receipt=str(dest/'initial_evaluation_receipt.json')),out)
