"""Pinned YOLO11 object-relative class-evidence KD; single-GPU pilot trainer."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

REPO = Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch
import yaml
import ultralytics
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from object_evidence_loss import object_evidence_loss, EvidenceConfig
from paired_rgbir_data import DualLabelRGBIRDataset
from tools.project_resource_guard import require_bound_lease_from_environment, bound_lease_resource_record_from_environment
from tools.write_jstars_run_receipt import emit_bound_run_receipt, implementation_files


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def raw_prediction(pred):
    if isinstance(pred, tuple):
        pred = pred[1]
    if not isinstance(pred, dict) or not {'scores', 'boxes', 'feats'} <= set(pred):
        raise TypeError('Pinned YOLO raw prediction dictionary required')
    return pred


def combine_loss(native_total, kd, batch_size, weight):
    """Native already includes B; add the KD scalar once, never per component."""
    return native_total.sum() + float(batch_size) * float(weight) * kd


def load_frozen(path, expected_data, names):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = payload.get('ema') or payload.get('model')
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f'No model/EMA in {path}')
    args_path = Path(path).parents[1]/'args.yaml'
    args = yaml.safe_load(args_path.read_text())
    if str(args['data']) != str(expected_data):
        raise ValueError(f'Frozen model training data mismatch: {path}')
    if model.names != names:
        raise ValueError(f'Frozen model class order mismatch: {path}')
    model = model.float().eval()
    model.requires_grad_(False)
    return model


class EvidenceCriterion:
    def __init__(self, native, teacher, reference, cfg, arm, trainer, sanity):
        self.native, self.teacher, self.reference = native, teacher, reference
        self.cfg, self.arm, self.trainer, self.sanity = cfg, arm, trainer, sanity
        self.calls = 0
        self.selected_total = 0
        self.gradient_checks = []
        self.last_stats = {}
        self.evidence_cfg = EvidenceConfig(**cfg['evidence'])

    def __deepcopy__(self, memo):
        # Training-only state must never duplicate frozen models into EMA/export.
        return None

    def __call__(self, prediction, batch):
        native_total, items = self.native(prediction, batch)
        if 'teacher_batch' not in batch:
            return native_total, items
        self.calls += 1
        student = raw_prediction(prediction)
        self.teacher.eval(); self.reference.eval()
        with torch.no_grad():
            teacher = raw_prediction(self.teacher(batch['strong_img']))
            reference = raw_prediction(self.reference(batch['img']))
        # weight0 runs the same auxiliary path, but applies a literal zero dose.
        kd, stats = object_evidence_loss(student, teacher, reference, batch,
            strides=tuple(int(x) for x in self.trainer.model.stride),
            config=self.evidence_cfg, arm='paired', seed=self.cfg['seed']+self.calls)
        weight = 0.0 if self.arm == 'weight0' else float(self.cfg['kd_weight'])
        b = int(batch['img'].shape[0])
        total = combine_loss(native_total, kd, b, weight)
        if not bool(torch.isfinite(total)):
            raise FloatingPointError('Non-finite total loss; no automatic recovery')
        self.selected_total += int(stats['selected_count'])
        log_now = self.calls <= 3 or self.sanity or self.calls % self.cfg['log_every_batches'] == 0
        grad_check = (self.sanity and (self.calls == 1 or (stats['selected_count'] and not any(
            c['kd_score_gradient_l2'] > 0 for c in self.gradient_checks))))
        if grad_check:
            gn = torch.autograd.grad(native_total.sum(), student['scores'], retain_graph=True)[0]
            gk = torch.autograd.grad(kd, student['scores'], retain_graph=True, allow_unused=True)[0]
            gt = torch.autograd.grad(total, student['scores'], retain_graph=True)[0]
            expected = gn + weight*b*gk if gk is not None else gn
            error = float((gt-expected).abs().max())
            # AMP score tensors can be fp16; compare in that actual dtype.
            if not torch.allclose(gt, expected, rtol=3e-3, atol=3e-5):
                raise AssertionError('KD scaling/gradient composition differs from one scalar addition')
            zero_total = combine_loss(native_total, kd, b, 0.0)
            gz = torch.autograd.grad(zero_total, student['scores'], retain_graph=True)[0]
            if not torch.equal(gz, gn) or not torch.equal(zero_total, native_total.sum()):
                raise AssertionError('weight0 is not exactly native loss/score gradient')
            check = {'batch': self.calls, 'native_score_gradient_l2':float(gn.float().norm()),
                'kd_score_gradient_l2':float(gk.float().norm()) if gk is not None else 0.0,
                'total_gradient_composition_max_error':error, 'weight0_exact_loss_gradient':True,
                'teacher_has_grad':any(p.grad is not None for p in self.teacher.parameters()),
                'reference_has_grad':any(p.grad is not None for p in self.reference.parameters())}
            if check['teacher_has_grad'] or check['reference_has_grad']:
                raise AssertionError('Frozen model accumulated a gradient')
            self.gradient_checks.append(check)
            append_json(self.trainer.save_dir/'gradient_checks.jsonl', check)
        stats.update(batch=self.calls, arm=self.arm, epoch=int(self.trainer.epoch),
            native_total=float(native_total.detach().sum()), kd_weight=weight,
            weighted_kd_total=float(kd.detach())*b*weight, total_loss=float(total.detach()),
            optimizer_updates=self.trainer.real_updates,
            time_seconds=time.time()-self.trainer.wall_started)
        self.last_stats = stats
        if log_now:
            stats['student_files'] = list(batch.get('im_file', []))
            append_json(self.trainer.save_dir/'kd_batches.jsonl', stats)
        return total, items


def build_trainer(cfg, config_path, output, arm, max_steps=None):
    mapping = json.loads(Path(cfg['paths']['paired_train_mapping']).read_text())
    teacher_data = check_det_dataset(cfg['paths']['privileged_data_yaml'], autodownload=False)
    class ObjectEvidenceTrainer(DetectionTrainer):
        def __init__(self, *args, **kwargs):
            self.wall_started = time.time()
            self.real_updates = 0
            self.update_attempts = 0
            self.skipped_amp_updates = 0
            self.batch_visits = 0
            self.criterion_ref = None
            super().__init__(*args, **kwargs)

        def _build_train_pipeline(self):
            if self.batch_size != cfg['batch'] or self.args.batch != cfg['batch']:
                raise RuntimeError('Frozen batch changed: automatic OOM batch reduction is rejected')
            super()._build_train_pipeline()

        def build_dataset(self, img_path, mode='train', batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            if mode != 'train':
                return ds
            td = build_yolo_dataset(self.args, teacher_data['train'], batch, teacher_data,
                mode='train', rect=False, stride=max(int(self.model.stride.max()),32))
            return DualLabelRGBIRDataset(ds, td, mapping,
                max_teacher_cache=cfg['teacher_cache_images'])

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            batch['strong_img'] = batch['strong_img'].to(self.device, non_blocking=True).float()/255
            batch['teacher_batch'] = {key:batch['strong_'+key].to(self.device,non_blocking=True)
                for key in ('batch_idx','cls','bboxes')}
            self.batch_visits += 1
            if max_steps is not None and self.batch_visits == 1:
                # CPU tensor copies permit direct P/N comparison, without hashes.
                keys = ('img','cls','bboxes','batch_idx','strong_img')
                torch.save({key:batch[key].detach().cpu() for key in keys}, self.save_dir/'first_batch.pt')
            return batch

        def _setup_train(self):
            super()._setup_train()
            if self.world_size != 1:
                raise RuntimeError('This pilot permits one GPU only')
            if bool(self.amp) != bool(cfg['amp']):
                raise RuntimeError('AMP contract changed during setup')
            # Establish student/EMA/optimizer before attaching auxiliary models.
            names = self.model.names
            teacher = load_frozen(cfg['teacher'], cfg['paths']['privileged_data_yaml'], names).to(self.device)
            reference = load_frozen(cfg['reference'], cfg['paths']['student_data_yaml'], names).to(self.device)
            self.criterion_ref = EvidenceCriterion(self.model.init_criterion(), teacher, reference,
                cfg, arm, self, max_steps is not None)
            self.model.criterion = self.criterion_ref
            if set(self.ema.ema.state_dict()) != set(self.model.state_dict()):
                raise AssertionError('Auxiliary state leaked into student or EMA')
            opt_ids = {id(p) for group in self.optimizer.param_groups for p in group['params']}
            if any(id(p) in opt_ids for m in (teacher,reference) for p in m.parameters()):
                raise AssertionError('Auxiliary parameters leaked into optimizer')
            if max_steps is not None:
                torch.save({k:v.detach().cpu() for k,v in self.model.state_dict().items()},self.save_dir/'initial_student.pt')
            write_json(self.save_dir/'runtime_ready.json', {'status':'ready','arm':arm,
                'train_images':len(self.train_loader.dataset),'val_images':len(self.test_loader.dataset),
                'train_batches':len(self.train_loader),'batch':self.batch_size,'workers':self.train_loader.num_workers,
                'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'student_only_state':True,'frozen_models_outside_optimizer':True,
                'amp':bool(self.amp),'class_names':names,
                'teacher_labels_used_by_kd':True,'student_native_gt_only':True})

        def optimizer_step(self):
            scale = self.scaler.get_scale()
            super().optimizer_step()
            self.update_attempts += 1
            if self.scaler.get_scale() < scale:
                self.skipped_amp_updates += 1
            else:
                self.real_updates += 1
            if max_steps is not None and self.real_updates >= max_steps:
                self.stop = True
            if self.update_attempts % 10 == 0:
                write_json(self.save_dir/'progress.json', {'status':'running','arm':arm,
                    'epoch':int(self.epoch)+1,'epochs':cfg['epochs'],'optimizer_updates':self.real_updates,
                    'update_attempts':self.update_attempts,'amp_skips':self.skipped_amp_updates,
                    'batches':self.criterion_ref.calls,'selected_objects':self.criterion_ref.selected_total,
                    'seconds':time.time()-self.wall_started,'last_kd':self.criterion_ref.last_stats.get('loss_unweighted')})

        def _handle_nan_recovery(self, epoch):
            if self.loss is not None and not bool(torch.isfinite(self.loss).all()):
                raise FloatingPointError('NaN/Inf loss: automatic checkpoint recovery disabled')
            return False

        def save_model(self):
            for model in (self.model,self.ema.ema):
                if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values() if v.is_floating_point()):
                    raise FloatingPointError('Nonfinite live/EMA state; no repaired checkpoint may be saved')
            return super().save_model()

        def validate(self):
            return {},0.0

        def final_eval(self):
            return {},0.0

    keys=('imgsz','epochs','batch','nbs','workers','optimizer','lr0','lrf','momentum',
        'weight_decay','warmup_epochs','warmup_momentum','warmup_bias_lr','cos_lr',
        'close_mosaic','patience','amp','deterministic','seed')
    overrides={k:cfg[k] for k in keys}
    overrides.update(model=cfg['model'],data=cfg['paths']['student_data_yaml'],device='0',
        val=False,save=True,save_period=-1,pretrained=True,plots=False,cache=False,rect=False,
        multi_scale=False,compile=False,channels_last=False,project=str(output.parent),
        name=output.name,exist_ok=True,**cfg['augmentation'])
    return ObjectEvidenceTrainer(overrides=overrides)


def run(args):
    cfg=yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        if args.seed not in (0,42,123): raise ValueError('Seed not in planned set')
        cfg['seed']=args.seed
    for key in ('student_data_yaml','privileged_data_yaml'):
        if 'test' in yaml.safe_load(Path(cfg['paths'][key]).read_text()):
            raise ValueError('Pilot requires train/val-only data files')
    if str(ultralytics.__version__) != cfg['ultralytics_version'] or str(torch.__version__) != cfg['torch_version']:
        raise RuntimeError('Pinned environment changed')
    for key in ('model','teacher','reference'):
        if not Path(cfg[key]).is_file(): raise FileNotFoundError(cfg[key])
    lease = require_bound_lease_from_environment()
    if len(lease['gpus']) != 1: raise RuntimeError('Exactly one bound physical GPU required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    shutil.copy2(args.config,args.output/'protocol_config.yaml')
    snapshot=args.output/'implementation_snapshot';snapshot.mkdir()
    for p in (Path(__file__),HERE/'object_evidence_loss.py',HERE/'paired_rgbir_data.py'):
        shutil.copy2(p,snapshot/p.name)
    write_json(args.output/'launch_manifest.json',{'method_id':cfg['method_id'], 'arm':args.arm,
        'seed':cfg['seed'],'command':[sys.executable,*sys.argv],'gpu':lease['gpus'],
        'inputs':{key:{'path':cfg[key],'bytes':Path(cfg[key]).stat().st_size} for key in ('model','teacher','reference')},
        'canary_max_updates':args.max_steps,'teacher_labels_used_by_kd':True,
        'test_accessed':False,'environment':{'torch':str(torch.__version__),'ultralytics':str(ultralytics.__version__)}})
    started=time.time()
    trainer=None
    try:
        trainer=build_trainer(cfg,args.config,args.output,args.arm,args.max_steps)
        trainer.train()
        criterion=trainer.criterion_ref
        if args.max_steps is None and trainer.epoch+1 != cfg['epochs']:
            raise RuntimeError('Full run ended before frozen epoch budget')
        if args.max_steps is not None and (trainer.real_updates < args.max_steps or criterion.selected_total == 0):
            raise RuntimeError('Canary did not complete updates or selected no objects')
        if args.max_steps is not None and not any(c['kd_score_gradient_l2']>0 for c in criterion.gradient_checks):
            raise RuntimeError('Canary has no nonzero KD score-gradient evidence')
        status='canary_completed' if args.max_steps is not None else 'training_completed'
        receipt={'status':status,'method_id':cfg['method_id'],'arm':args.arm,'seed':cfg['seed'],
            'epochs_configured':cfg['epochs'],'last_epoch':trainer.epoch+1,'optimizer_updates':trainer.real_updates,
            'optimizer_update_attempts':trainer.update_attempts,'amp_skipped_updates':trainer.skipped_amp_updates,
            'ema_updates':trainer.ema.updates,'batches':criterion.calls,'selected_objects':criterion.selected_total,
            'gradient_checks':criterion.gradient_checks,'kd_last':criterion.last_stats,
            'gpu_allocated_peak_mib':torch.cuda.max_memory_allocated()/2**20,
            'gpu_reserved_peak_mib':torch.cuda.max_memory_reserved()/2**20,
            'resources':bound_lease_resource_record_from_environment(),
            'seconds':time.time()-started,'checkpoint':str(args.output/'weights/last.pt'),
            'official_test_accessed':False,'single_seed_exploratory':True}
        write_json(args.output/'completion_receipt.json',receipt)
        emit_bound_run_receipt(run_dir=args.output/'run_evidence',method_identity=cfg['method_identity'],
            dataset=cfg['dataset'],data_role='development_train',seed=cfg['seed'],run_kind='train',
            trainers=[Path(__file__),HERE/'paired_rgbir_data.py',*implementation_files(DetectionTrainer)],
            losses=implementation_files(EvidenceCriterion,criterion.native,object_evidence_loss),
            configs=[args.config,Path(cfg['paths']['student_data_yaml']),Path(cfg['paths']['privileged_data_yaml'])],
            split_rosters=[Path(cfg['paths']['paired_train_mapping'])],metric_files=[args.output/'completion_receipt.json'],
            environment={'torch':str(torch.__version__),'ultralytics':str(ultralytics.__version__)},
            inputs={'method_id':cfg['method_id'],'arm':args.arm,'teacher_labels_used_by_kd':True,
                'student_native_gt_only':True,'initial_weights':cfg['model'],'teacher_weights':cfg['teacher'],
                'reference_weights':cfg['reference'],'canary':args.max_steps is not None})
        print(json.dumps({'status':status,'output':str(args.output),'updates':trainer.real_updates}),flush=True)
    except BaseException as error:
        write_json(args.output/'failure_receipt.json',{'status':'failed','arm':args.arm,
            'error':repr(error),'traceback':traceback.format_exc(),'seconds':time.time()-started,
            'updates':getattr(trainer,'real_updates',0),'official_test_accessed':False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=['paired','weight0'],required=True)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--max-steps',type=int)
    args=parser.parse_args()
    if args.max_steps is not None and args.max_steps<=0:parser.error('max-steps must be positive')
    run(args)

if __name__=='__main__':main()
