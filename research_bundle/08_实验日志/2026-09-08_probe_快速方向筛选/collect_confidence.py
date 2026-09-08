from pathlib import Path
root=Path(__file__).parent
source=(root/'collect_direction.py').read_text(encoding='utf-8')
source=source.replace("('results_'+tag)","('confidence_results_'+tag)")
source=source.replace('rgbir_direction_screen_20260908/screen_attempt1','rgbir_direction_screen_20260908/confidence_attempt1')
source=source.replace("'direction_config.yaml'","'direction_config.yaml','confidence_config.yaml'")
exec(compile(source,str(root/'collect_direction.py'),'exec'))
