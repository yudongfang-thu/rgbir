"""Export already-frozen dev metadata; no image reads, inference or new bins."""
import argparse
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prepare(args):
    rows=[json.loads(line) for line in args.images.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    summary=read(args.summary);a=read(args.baseline_evaluation);b=read(args.candidate_evaluation)
    contract=read(args.contract)
    if (summary.get('status')!='completed' or summary.get('dataset')!='dronevehicle'
            or summary.get('split')!='val' or summary.get('images')!=200 or len(rows)!=200):
        raise ValueError('Require the already-frozen completed Drone dev200 metadata')
    roster=contract['roster']
    if len(roster)!=1469 or len(set(roster))!=1469:raise ValueError('Require full actual dev1469 roster')
    names=[]
    for endpoint in (a,b):
        if endpoint.get('dataset')!='dronevehicle' or endpoint.get('official_test_accessed') is not False:
            raise ValueError('Metadata class inventory must come from actual development endpoints')
        inventory={str(row['class_id']):row['name'] for row in endpoint['per_class']}
        if set(inventory)!={'0','1','2','3','4'}:raise ValueError('Require exact five-class inventory')
        names.append(inventory)
    if names[0]!=names[1]:raise ValueError('Endpoint class names disagree')
    images={}
    for row in rows:
        image=row['rgb_path']
        if row.get('split')!='val' or image not in roster or image in images:
            raise ValueError('Old metadata has duplicate/non-dev image identity')
        expected='low' if row['mean_rgb_luma']<78.283 else 'high'
        if row['luminance_proxy']!=expected:raise ValueError('Existing frozen luminance labels contradict their recorded rule')
        source=row.get('source_group') or 'UNKNOWN'
        if source.startswith('unavailable:'):source='UNKNOWN'
        images[image]=dict(source_group=source,brightness_bin=row['luminance_proxy'])
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    metadata=dict(frozen=True,key_type='image',class_names=names[0],images=images,
        brightness_definition=dict(source='Earlier outcome-blind Drone dev200 D1/D2 images.jsonl; reuse recorded labels only',
            measurement='mean of original PIL Image.convert(L) pixels (previously computed; not recomputed here)',
            rule='low < 78.283; high >= 78.283',threshold=78.283,is_day_night_label=False,
            coverage_scope='Only previously frozen 200 development images; all other images UNKNOWN'))
    with (output/'frozen_metadata.json').open('x',encoding='utf-8') as stream:
        json.dump(metadata,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    copies=[]
    for i,path in enumerate((args.images,args.summary,args.baseline_evaluation,args.candidate_evaluation,args.contract,Path(__file__))):
        destination=output/'source_evidence'/f'{i:02d}_{path.name}';destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path,destination);copies.append(dict(original=str(path.resolve()),copy=str(destination)))
    receipt=dict(status='EXPORTED_EXISTING_FROZEN_METADATA',original_images=200,full_dev_images=1469,
        known_brightness_images=200,unknown_brightness_images=1269,
        known_source_images=sum(r['source_group']!='UNKNOWN' for r in images.values()),
        bins_recomputed=False,pixels_read=False,outcomes_used_to_select_metadata=False,
        class_names_source='Matching five class IDs/names in the two actual completed seed42 dev reevaluations',
        source_copies=copies)
    with (output/'metadata_export_receipt.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('images','summary','baseline-evaluation','candidate-evaluation','contract','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    prepare(parser.parse_args())
