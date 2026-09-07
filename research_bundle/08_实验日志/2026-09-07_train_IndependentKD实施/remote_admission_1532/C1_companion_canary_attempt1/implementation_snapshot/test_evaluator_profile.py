"""CPU filesystem/metric fixtures only; never a real AP or GPU profile."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import yaml
import evaluator_profile as ep


class SourceDataset:
    pass


class SourceLoader:
    def __init__(self):
        self.dataset = SourceDataset()


class SourceNetwork:
    pass


class EvaluatorProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.data=self.root/'rgb.yaml'
        self.data.write_text('path: '+self.root.as_posix()+'\ntrain: train\nval: val\nnames: [car]\n')
        self.cfg=dict(dataset='dronevehicle',model='/synthetic/yolo11n.pt',expected_nc=5,
            imgsz=640,batch=32,workers=4,torch_version='fixture',ultralytics_version='fixture',
            expected_val_images=1469,paths={'student_data_yaml':str(self.data)})
        self.roster=[str(self.root/f'{i}.jpg') for i in range(1469)]
        self.sources=[]
        for name in ('evaluator_profile.py','evaluate_independent.py','native_loader.py'):
            p=self.root/name;p.write_text('# synthetic source '+name+'\n')
            self.sources.append(p)

    def binding(self):
        out=self.root/'new_probe';out.mkdir()
        return ep.make_binding(out,self.cfg,dict(effective_kwargs={'imgsz':640,'batch':32,'quantize':None},
            roster=self.roster,observed_images=1469),self.sources)

    def validate(self,path,cfg=None):
        with patch.object(ep,'dev_roster',return_value=self.roster):
            return ep.validate_evaluation_profile_binding(path,cfg or self.cfg,self.root)

    def test_five_metrics_and_per_class_require_exact_equality(self):
        row={name:.125 for name in ep.METRICS}
        row['per_class']=[{'class_id':0,'AP50':.25}]
        self.assertEqual(ep.assert_equal_metrics(row,dict(row)),dict.fromkeys(ep.METRICS,0.))
        for name in ep.METRICS:
            with self.subTest(metric=name),self.assertRaises(ValueError):
                ep.assert_equal_metrics(row,dict(row,**{name:.12500001}))
        with self.assertRaises(ValueError):
            ep.assert_equal_metrics(row,dict(row,per_class=[]))

    def test_nonfinite_metrics_rejected(self):
        row={name:.125 for name in ep.METRICS}
        with self.assertRaises(ValueError):
            ep.assert_equal_metrics(row,dict(row,recall=float('nan')))

    def test_readonly_baseline_requires_actual_n42_last_e200(self):
        old=self.root/'old';checkpoint=old/'weights/last.pt'
        checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'SYNTHETIC NOT WEIGHTS')
        receipt=dict(status='training_completed',arm='weight0',seed=42,last_epoch=200,
            epochs_configured=200,official_test_accessed=False,checkpoint=str(checkpoint))
        path=old/'completion_receipt.json';path.write_text(json.dumps(receipt))
        content=path.read_bytes()
        self.assertEqual(ep.baseline_identity(checkpoint)[0],old)
        self.assertEqual(content,path.read_bytes())
        for changes in ({'seed':0},{'last_epoch':199},{'arm':'C1'},{'official_test_accessed':True}):
            path.write_text(json.dumps(dict(receipt,**changes)))
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                ep.baseline_identity(checkpoint)

    def test_config_identity_ignores_method_payload_but_not_eval_recipe(self):
        self.assertEqual(ep.configuration_identity(self.cfg),
                         ep.configuration_identity(dict(self.cfg,arm='C1',classification_coefficient=.4)))
        self.assertNotEqual(ep.configuration_identity(self.cfg),
                            ep.configuration_identity(dict(self.cfg,batch=16)))
        with self.assertRaises(ValueError):
            ep.configuration_identity(dict(self.cfg,dataset='llvip'))

    def test_dev_roster_only_val_and_rejects_duplicates_and_test(self):
        val=self.root/'val';val.mkdir()
        (val/'a.jpg').write_bytes(b'fixture');(val/'b.png').write_bytes(b'fixture')
        self.assertEqual(len(ep.dev_roster(self.data)),2)
        roster=self.root/'list.txt';roster.write_text('val/a.jpg\nval/a.jpg\n')
        self.data.write_text(yaml.safe_dump(dict(path=str(self.root),train='train',val='list.txt')))
        with self.assertRaises(ValueError):
            ep.dev_roster(self.data)
        self.data.write_text(yaml.safe_dump(dict(path=str(self.root),val='val',test='sealed')))
        with self.assertRaises(ValueError):
            ep.dev_roster(self.data)

    def test_actual_config_source_data_bytes_derive_verified_profile_identity(self):
        binding=self.binding()
        value=self.validate(binding)
        self.assertEqual(value['expected_val_images'],1469)
        self.assertEqual(value['configuration']['batch'],32)
        self.assertIn(str(self.root/'evaluate_independent.py'),value['evaluator_sources'])
        with self.assertRaises(ValueError):
            self.validate(binding,dict(self.cfg,batch=16))
        self.sources[-1].write_text('# changed actual loader\n')
        with self.assertRaisesRegex(ValueError,'source bytes changed'):
            self.validate(binding)

    def test_same_path_data_or_roster_change_is_rejected(self):
        binding=self.binding()
        self.data.write_text(self.data.read_text()+'# changed\n')
        with self.assertRaisesRegex(ValueError,'YAML bytes'):
            self.validate(binding)

    def test_missing_source_or_incomplete_population_not_accepted(self):
        binding=self.binding();value=ep.read_json(binding)
        value['source_files']=[r for r in value['source_files'] if not r['original'].endswith('evaluate_independent.py')]
        binding.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'source binding missing'):
            self.validate(binding)
        value['roster']=value['roster'][:2];binding.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'population'):
            self.validate(binding)

    def test_runtime_sources_use_loaded_network_without_validator_model(self):
        # Pinned 8.4.115 on_val_start has a dataloader but no validator.model.
        validator = SimpleNamespace(dataloader=SourceLoader())
        loaded_model = SimpleNamespace(model=SourceNetwork())
        sources = set()
        result = ep.capture_runtime_sources(validator, loaded_model, sources)
        self.assertFalse(hasattr(validator, 'model'))
        self.assertEqual(set(result), {'loader', 'dataset', 'network'})
        self.assertTrue(result['network']['class_name'].endswith('.SourceNetwork'))
        self.assertEqual(sources, {Path(__file__).resolve()})

    def test_runtime_source_binding_does_not_silently_omit_unknown_classes(self):
        validator = SimpleNamespace(dataloader=SourceLoader())
        with patch.object(ep.inspect, 'getsourcefile', return_value=None):
            with self.assertRaisesRegex(ValueError, 'runtime source unavailable'):
                ep.capture_runtime_sources(validator, SimpleNamespace(model=SourceNetwork()), set())


if __name__=='__main__':
    unittest.main(verbosity=2)
