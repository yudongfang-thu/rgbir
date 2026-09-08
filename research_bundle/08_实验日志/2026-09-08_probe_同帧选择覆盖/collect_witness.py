from pathlib import Path
root=Path(__file__).parent
source=(root/'collect_evidence.py').read_text(encoding='utf-8')
source=source.replace("('evidence_'+tag)","('witness_evidence_'+tag)")
source=source.replace('rgbir_selection_coverage_20260908/attempt','rgbir_selection_coverage_20260908/witness_attempt')
source=source.replace("('release_v'+%r)","('witness_release_v'+%r)")
source=source.replace("('mapping_cpu.json','selection_cpu.json')","('witness_cpu.json',)")
exec(compile(source,str(root/'collect_evidence.py'),'exec'))
