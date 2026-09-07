"""Known-truth CPU fixtures; no model inference or AP result is used."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import yaml

import object_error_analysis as oa


def row(image='image-a',gt=None,gc=None,pred=None,pc=None,conf=None):
    gt=[] if gt is None else gt;pred=[] if pred is None else pred
    return dict(image=image,canvas_shape=[640,640],original_shape=[640,640],
        gt_boxes=gt,gt_classes=[0]*len(gt) if gc is None else gc,
        pred_boxes=pred,pred_classes=[0]*len(pred) if pc is None else pc,
        pred_confidence=[.9]*len(pred) if conf is None else conf)


class KnownTruthTests(unittest.TestCase):
    def test_repair_damage_have_separate_baseline_denominators(self):
        gt=[[0,0,10,10],[20,0,30,10],[40,0,50,10]]
        a=row(gt=gt,pred=[gt[0]])
        b=row(gt=gt,pred=[gt[1]])
        result=oa.analyze_pair([a],[b],nc=2)
        s=result['summary']
        self.assertEqual((s['baseline_correct'],s['baseline_incorrect'],s['repaired'],s['damaged']), (1,2,1,1))
        self.assertEqual(s['repair_rate'],.5)
        self.assertEqual(s['damage_rate'],1.)
        self.assertEqual([r['transition'] for r in result['objects']],['damaged','repaired','still_incorrect'])
        self.assertEqual(json.loads(result['objects'][1]['sample_key']),['image-a',1])

    def test_duplicate_prediction_counts_once_and_is_not_background(self):
        gt=[[0,0,10,10]]
        matched=oa.match_image(row(gt=gt,pred=gt+gt,conf=[.8,.9]))
        self.assertEqual(matched['objects'][0]['matched_prediction_index'],1)
        self.assertEqual(matched['prediction_counts']['correct'],1)
        self.assertEqual(matched['prediction_counts']['duplicate_or_assignment_competition'],1)
        self.assertEqual(matched['background_fp'],0)

    def test_class_wrong_is_not_repair_and_not_background(self):
        gt=[[0,0,10,10]]
        a=row(gt=gt,pred=gt,pc=[1])
        b=row(gt=gt,pred=gt)
        result=oa.analyze_pair([a],[b],nc=2)
        self.assertEqual(result['objects'][0]['baseline']['state'],'wrong_class')
        self.assertEqual(result['baseline']['prediction_counts']['wrong_class'],1)
        self.assertEqual(result['baseline']['background_fp'],0)
        self.assertEqual(result['summary']['repair_rate'],1.)
        self.assertIsNone(result['summary']['damage_rate'])

    def test_empty_gt_images_count_in_background_denominator(self):
        a=[row('a'),row('b',gt=[[0,0,10,10]])]
        b=[row('a',pred=[[0,0,10,10]]),row('b',gt=[[0,0,10,10]])]
        result=oa.analyze_pair(a,b,nc=1)
        self.assertEqual(result['candidate']['background_fp_per_image'],.5)
        self.assertEqual(result['background_groups']['class_id']['0']['candidate_per_image'],.5)
        empty=oa.analyze_pair([row()],[row()])['summary']
        self.assertIsNone(empty['repair_rate']);self.assertIsNone(empty['damage_rate'])

    def test_threshold_inclusivity_and_any_class_background_rule(self):
        gt=[[0,0,10,10]]
        exact=oa.match_image(row(gt=gt,pred=[[0,0,20,10]],conf=[.25]))
        self.assertTrue(exact['objects'][0]['correct'])
        low=oa.match_image(row(gt=gt,pred=gt,conf=[.249999]))
        self.assertFalse(low['objects'][0]['correct'])
        boundary=oa.match_image(row(gt=gt,pred=[[0,0,100,10]],pc=[1]))
        self.assertEqual(boundary['background_fp'],0)
        self.assertEqual(boundary['prediction_counts']['localization_or_mixed'],1)

    def test_ties_resolve_original_prediction_then_gt_order(self):
        gt=[[0,0,10,10],[0,0,10,10]]
        one=oa.match_image(row(gt=gt,pred=[gt[0]]))
        self.assertTrue(one['objects'][0]['correct'])
        self.assertEqual(one['objects'][1]['state'],'assignment_competition')
        two=oa.match_image(row(gt=gt,pred=gt,conf=[.9,.9]))
        self.assertEqual([x['matched_prediction_index'] for x in two['objects']],[0,1])

    def test_gt_order_classes_and_population_must_be_identical(self):
        a=row(gt=[[0,0,10,10],[20,0,30,10]],gc=[0,1])
        for change in (dict(gt_boxes=list(reversed(a['gt_boxes']))),dict(gt_classes=[1,0]),dict(image='other')):
            with self.subTest(change=change),self.assertRaises(ValueError):
                oa.analyze_pair([a],[dict(a,**change)])
        with self.assertRaisesRegex(ValueError,'Duplicate image'):
            oa.analyze_pair([a,a],[a])

    def test_canvas_scales_and_missing_metadata_are_explicit(self):
        boxes=[[0,0,31,32],[0,0,32,32],[0,0,96,96]]
        self.assertEqual([oa.scale_bin(x) for x in boxes],list(oa.SCALE_NAMES))
        a=row(gt=boxes)
        result=oa.analyze_pair([a],[copy.deepcopy(a)])
        self.assertEqual(result['objects'][0]['source_group'],'UNKNOWN')
        self.assertEqual(result['objects'][0]['brightness_bin'],'UNKNOWN')
        with self.assertRaisesRegex(ValueError,'canvas'):
            oa.analyze_pair([dict(a,canvas_shape=[1280,1280])],[a])

    def test_metadata_groups_do_not_omit_zero_gt_images(self):
        a=[row('a',gt=[[0,0,10,10]]),row('b')]
        b=[copy.deepcopy(a[0]),row('b',pred=[[0,0,10,10]])]
        meta={x:dict(source_group='s',brightness_bin='low_proxy') for x in ('a','b')}
        result=oa.analyze_pair(a,b,meta,nc=2)
        self.assertEqual(result['background_groups']['brightness_bin']['low_proxy']['images'],2)
        self.assertEqual(result['background_groups']['source_group']['s']['candidate_per_image'],.5)
        self.assertEqual(result['object_groups']['class_id']['1']['gt_objects'],0)

    def test_inputs_remain_unchanged(self):
        a=row(gt=[[0,0,10,10]],pred=[[0,0,10,10]])
        before=copy.deepcopy(a)
        oa.analyze_pair([a],[copy.deepcopy(a)],nc=1)
        self.assertEqual(a,before)


class FileEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def endpoint(self,name='N'):
        folder=self.root/name;folder.mkdir()
        evidence=folder/'eval_evidence';evidence.mkdir()
        objects=folder/'objects.jsonl.gz'
        roster=['/synthetic/dev/image-%04d.jpg'%i for i in range(1469)]
        with gzip.open(objects,'wt',encoding='utf-8') as stream:
            for image in roster:stream.write(json.dumps(row(image))+'\n')
        cfg=dict(model='/synthetic/model.pt',paths={},expected_nc=2,epochs=200,imgsz=640,
                 seed=42,dataset='dronevehicle',arm=name,source='paired',method_id='fixture-'+name)
        config=evidence/'config.yaml';config.write_text(yaml.safe_dump(cfg))
        contract=dict(schema='rgbir-evaluation-contract-v1',roster=roster,observed_images=1469,
            expected_val_images=1469,official_test_accessed=False,endpoint='fixed_budget_last_ema',
            effective_kwargs={'conf':.001,'iou':.7},evaluator_identity={'native':'fixture'})
        cp=folder/'evaluation_contract.json';cp.write_text(json.dumps(contract));(evidence/'contract.json').write_bytes(cp.read_bytes())
        raw=dict(status='completed',endpoint='fixed_budget_last_ema',split='val',official_test_accessed=False,
            dataset='dronevehicle',seed=42,method_id=cfg['method_id'],arm=name,source='paired',
            checkpoint='/synthetic/'+name+'/last.pt',objects=str(objects),evaluation_contract=str(cp))
        metric=folder/'evaluation_val.json';metric.write_text(json.dumps(raw))
        (evidence/'metric.json').write_bytes(metric.read_bytes());(evidence/'objects.gz').write_bytes(objects.read_bytes())
        receipt=dict(terminal_status='COMPLETED',run_kind='eval',data_role='development_val',dataset='dronevehicle',seed=42,
            inputs={k:raw[k] for k in ('method_id','arm','source','checkpoint','endpoint')},
            metric_snapshots=['metric.json','objects.gz'],source_snapshots={'config':['config.yaml','contract.json']})
        (evidence/'run_receipt.json').write_text(json.dumps(receipt))
        return metric

    def test_actual_object_and_contract_bytes_are_bound(self):
        metric=self.endpoint()
        self.assertEqual(len(oa.load_evaluation(metric)['rows']),1469)
        objects=metric.parent/'objects.jsonl.gz'
        with gzip.open(objects,'at',encoding='utf-8') as stream:stream.write(json.dumps(row('extra'))+'\n')
        with self.assertRaisesRegex(ValueError,'Input bytes'):
            oa.load_evaluation(metric)

    def test_missing_receipt_and_early_or_test_endpoint_rejected(self):
        metric=self.endpoint();raw=oa.read_json(metric)
        raw['official_test_accessed']=True;metric.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError,'development endpoint'):
            oa.load_evaluation(metric)

    def test_frozen_metadata_no_brightness_inference(self):
        meta=self.root/'meta.json'
        meta.write_text(json.dumps(dict(frozen=True,key_type='stem',images={'a':{'source_group':'folder','brightness_bin':'low'}})))
        with self.assertRaisesRegex(ValueError,'proxy definition'):
            oa.load_metadata(meta,['/images/a.jpg'])
        raw=oa.read_json(meta);raw['brightness_definition']={'source':'already frozen luminance quartiles','is_day_night_label':False}
        meta.write_text(json.dumps(raw))
        rows,_,definition,_=oa.load_metadata(meta,['/images/a.jpg','/images/b.jpg'])
        self.assertEqual(rows['/images/b.jpg']['brightness_bin'],'UNKNOWN')
        self.assertEqual(rows['/images/a.jpg']['source_group'],'folder')
        self.assertTrue(definition)

    def test_unreviewed_cli_outputs_cannot_match_accepted_analyzer_contract(self):
        a,b=self.endpoint('N'),self.endpoint('C1')
        output=self.root/'analysis'
        args=argparse.Namespace(baseline_evaluation=a,candidate_evaluation=b,output=output,
                               metadata=None,source_groups_tsv=None,review_receipt=None)
        before=a.read_bytes()
        result=oa.run(args)
        self.assertEqual(result['analyzer_status'],'NOT_ACCEPTED')
        diagnosis=oa.read_json(output/'candidate_error_analysis.json')
        self.assertNotIn('contract',diagnosis)
        self.assertEqual(diagnosis['draft_contract'],oa.ERROR_CONTRACT)
        self.assertIsNone(diagnosis['localization_diagnostics_consistent'])
        self.assertEqual(a.read_bytes(),before)
        with self.assertRaises(FileExistsError):oa.run(args)

    def test_review_must_bind_all_three_actual_files(self):
        review=self.root/'review.json'
        review.write_text(json.dumps(dict(status='ACCEPTED',schema='rgbir-object-error-review-v1',source_files=[])))
        with self.assertRaisesRegex(ValueError,'script, truth fixtures'):
            oa.verify_review(review)


if __name__=='__main__':unittest.main(verbosity=2)
