"""Explicitly opt six fixed endpoints into post-hoc classes and prior object evidence."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
LOG=HERE.parent
source=LOG/'legacy_evaluation_bridge_observed_v1/prepared_a3/manifest_accepted_bridge.json'
manifest=json.loads(source.read_text(encoding='utf-8-sig'))
for row in manifest['runs']:
    row['posthoc_class_metrics']=True
    label='baseline' if row['arm']=='N' else 'candidate'
    row['error_analysis_file']=str(LOG/'old_c0_object_diagnostics_v1/attempt2_outputs'/('seed'+str(row['seed']))/(label+'_error_analysis.json'))
manifest['implementation_checks_passed']=False
manifest['scope']='Separate post-hoc enrichment; original training and metric JSONs unchanged; no automatic extension authorized.'
with (HERE/'actual_manifest.json').open('x',encoding='utf-8') as stream:
    json.dump(manifest,stream,ensure_ascii=False,indent=2)
