"""Publish a new immutable confidence release; its queue waits for the main matrix."""
from pathlib import Path
root=Path(__file__).parent
source=(root/'deploy_direction.py').read_text(encoding='utf-8')
source=source.replace("release=root/'newentry/release'","release=root/'newentry/confidence_release'")
source=source.replace('rgbir_direction_screen_20260908/release_v1','rgbir_direction_screen_20260908/confidence_release_v1')
source=source.replace('rgbir_direction_screen_20260908/screen_attempt1','rgbir_direction_screen_20260908/confidence_attempt1')
source=source.replace('run_direction_queue.py','run_confidence_queue.py').replace('direction_screen_s42','confidence_llvip_s42')
source=source.replace('deployment_attempt1.json','confidence_deployment_attempt1.json')
exec(compile(source,str(root/'deploy_direction.py'),'exec'))
