from pathlib import Path
root=Path(__file__).parent
source=(root/'deploy_source.py').read_text(encoding='utf-8')
source=source.replace("release=root/'release'","release=root/'witness_release'")
source=source.replace('rgbir_selection_coverage_20260908/release_v','rgbir_selection_coverage_20260908/witness_release_v')
source=source.replace("('deployment_v'+version+'.json')","('witness_deployment_v'+version+'.json')")
exec(compile(source,str(root/'deploy_source.py'),'exec'))
