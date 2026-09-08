from pathlib import Path
import sys,json,time
B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
A=B/'artifacts'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
sys.path.insert(0,str(A/'rgbir_task_conditional_v1_20260907/release_v8'))
import resource_dispatch
out=A/'rgbir_direction_screen_20260908/state_swap_attempt1'
out.mkdir(exist_ok=False)
release=A/'rgbir_direction_screen_20260908/swap_release_v1'
for variant in ('parameter_only','buffer_only'):
    dest=out/variant
    command=[PY,str(release/'evaluate_state_swap.py'),'--reference-dir',str(A/'rgbir_independent_kd_v2_20260907/release_gpu5'),'--initial',str(B/'runs/rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1/weights/last.pt'),'--finetuned',str(A/'rgbir_hourly_screen_20260908/ft_screen_attempt1/runs/N/weights/last.pt'),'--native-config',str(A/'rgbir_throughput_20260908/short_screen_E8_release_attempt1/short_screen_E8_release/configs/drone_N_s42_E8.yaml'),'--native-profile-binding',str(A/'rgbir_independent_kd_v2_20260907/evaluator_profile_attempt2/evaluation_profile_binding/binding.json'),'--variant',variant,'--output',str(dest)]
    resource_dispatch.run_job(dict(id='direction_swap_'+variant,arm=variant,stage='eval',formal=False,kind='eval',vram_mib=2048,rss_mib=8192,command=command,expected_receipt=str(dest/'swap_evaluation_receipt.json')),out)
(out/'completion.json').write_text(json.dumps(dict(status='COMPLETED',new_hash_computed=False)))
