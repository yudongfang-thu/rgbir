"""Recompute the completed historical nine endpoints with the accepted analyzer."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ANALYZER=HERE.parent/'2026-09-07_audit_RGBIR实施起点/analyzer_source_v2/analyze_results.py'


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    spec=importlib.util.spec_from_file_location('accepted_results',ANALYZER)
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    manifest=dict(runs=[dict(arm=arm,seed=seed,path=str((args.snapshot/'raw'/f'{arm}{seed}').resolve()),
        protocol_id='oev1_frozen_drone_e200') for arm in ('N','C','R') for seed in (0,42,123)],implementation_checks_passed=False)
    result=analyzer.analyze_manifest(manifest,HERE)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print('Accepted analyzer completed; descriptive development evidence, no automatic claim upgrade.')


if __name__=='__main__':main()
