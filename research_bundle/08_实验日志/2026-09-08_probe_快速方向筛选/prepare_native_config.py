from pathlib import Path
import yaml
root=Path(__file__).parent/'newentry/release/configs'
c=yaml.safe_load((root/'llvip_N_s42_FT3.yaml').read_text())
c['paths']['student_data_yaml']=c['auxiliary_data_identity']['student_data_yaml']
c['paths']['privileged_data_yaml']=c['auxiliary_data_identity']['privileged_data_yaml']
c['scope']='NATIVE_FULL_DEV_PROJECTION_ONLY'
with (root/'llvip_native_evaluation.yaml').open('x',encoding='utf-8') as f:
    f.write(yaml.safe_dump(c,sort_keys=False))
