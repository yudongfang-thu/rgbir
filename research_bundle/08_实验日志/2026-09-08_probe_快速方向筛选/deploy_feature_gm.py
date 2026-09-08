"""Deploy one new, immutable feature relation follow-up through the existing lease."""
from pathlib import Path
root=Path(__file__).parent
source=(root/'deploy_direction.py').read_text(encoding='utf-8')
source=source.replace("release=root/'newentry/release'","release=root/'newentry/feature_gm_release'")
source=source.replace('rgbir_direction_screen_20260908/release_v1','rgbir_direction_screen_20260908/feature_gm_release_v1')
source=source.replace('rgbir_direction_screen_20260908/screen_attempt1','rgbir_direction_screen_20260908/feature_gm_attempt1')
source=source.replace('run_direction_queue.py','run_feature_gm_queue.py').replace('direction_screen_s42','feature_gm_drone_s42')
source=source.replace('deployment_attempt1.json','feature_gm_deployment_attempt1.json')
exec(compile(source,str(root/'deploy_direction.py'),'exec'))
