"""Run an author's CFT checkpoint on three preselected LLVIP fit image pairs.

This checks restored execution only. It does not evaluate AP, read dev/test,
construct a random replacement model, train, or claim benchmark reproduction.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                  OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4",
                  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
import argparse
import csv
import json
from pathlib import Path
import resource
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/mnt/dataX/ydf/projects/RGBT_campaign_90'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    def save(name, value):
        (a.output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    try:
        archive = a.root/'artifacts/reproduction_20260909_attempt1'
        receipt = json.loads((archive/'weight_downloaded.json').read_text())
        source_receipt = json.loads((archive/'source_downloaded.json').read_text())
        weight = Path(receipt['weight'])
        assert receipt['status'] == 'DOWNLOADED' and weight.stat().st_size == receipt['bytes']
        source = a.root/'external_reproductions/cft/author_source'
        os.environ['MPLCONFIGDIR'] = str(a.root/'cache/cft_matplotlib')
        os.environ['TORCH_HOME'] = str(a.root/'cache/torch')
        sys.path.insert(0, str(source))
        sys.dont_write_bytecode = True
        import torch
        import cv2
        import numpy as np
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        from models.experimental import attempt_load
        from utils.datasets import letterbox
        from utils.general import non_max_suppression
        cv2.setNumThreads(1)
        manifest = a.root/'data_attempt1/processed/llvip/splits/grouped_v1/fit.tsv'
        with manifest.open(newline='') as f:
            records = list(csv.DictReader(f, delimiter='\t'))
        assert len(records) == 9619
        selected = [records[i] for i in (0, len(records)//2, len(records)-1)]
        assert all(r['sequence_prefix'] not in {'01','04','07','12','25'} for r in selected)
        plan = dict(status='STARTED', task='author_checkpoint_execution_only',
                    reproduction_identity='PROTOCOL-ADAPTED', dataset='LLVIP', split='fit',
                    source_revision=source_receipt['revision'], checkpoint=str(weight),
                    checkpoint_bytes=weight.stat().st_size, selected=selected,
                    input_size=1024, letterbox_auto=True, confidence=0.25, nms_iou=0.45,
                    cpu_threads=4, student_rgb_only=False, model_inputs=['RGB','IR'],
                    test_accessed=False, dev_accessed=False, ap_computed=False,
                    new_digest_calculated=False, torch_version=torch.__version__,
                    runtime_compatibility=['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for trusted author legacy module checkpoint'],
                    limitations=['Author training population includes our dev; no generalization comparison',
                                 'Three fit pairs validate execution, not paper metrics or throughput',
                                 'Torch2.10 runtime differs from original environment'])
        save('plan.json', plan)
        t = time.perf_counter()
        model = attempt_load(str(weight), map_location=torch.device('cpu'))
        from cft_checkpoint_compat import restore_cft_block_names
        compatibility = restore_cft_block_names(model)
        plan['checkpoint_class_compatibility'] = compatibility
        plan['runtime_compatibility'].append(compatibility['adaptation'])
        save('compatibility.json', compatibility)
        save('plan.json', plan)
        assert not model.training and not torch.cuda.is_initialized()
        assert len(model.names) == 1, 'Expected LLVIP one-class checkpoint'
        stride = int(model.stride.max())
        loaded_seconds = time.perf_counter()-t
        items = []
        for record in selected:
            tensors, shapes = [], []
            for modality in ('visible','infrared'):
                im = cv2.imread(record[modality], cv2.IMREAD_COLOR)
                assert im is not None
                shapes.append(list(im.shape))
                processed = letterbox(im, new_shape=1024, auto=True, stride=stride)[0]
                processed = np.ascontiguousarray(processed[:, :, ::-1].transpose(2,0,1))
                tensors.append(torch.from_numpy(processed).float().div(255).unsqueeze(0))
            assert tensors[0].shape == tensors[1].shape
            t = time.perf_counter()
            with torch.no_grad():
                out, train_out = model(tensors[0], tensors[1], augment=False)
                assert out.ndim == 3 and out.shape[-1] == 6 and bool(torch.isfinite(out).all())
                detections = non_max_suppression(out.clone(), conf_thres=0.25, iou_thres=0.45)[0]
            elapsed = time.perf_counter()-t
            np.savez_compressed(a.output/(record['stem']+'_raw_output.npz'), raw=out.cpu().numpy())
            item = dict(stem=record['stem'], original_shapes=shapes, input_shape=list(tensors[0].shape),
                        raw_shape=list(out.shape), prediction_finite=True, detections=detections.tolist(),
                        detections_count=len(detections), forward_and_nms_seconds=elapsed)
            items.append(item)
            save(record['stem']+'_inference.json', item)
            print(json.dumps({k:item[k] for k in ('stem','input_shape','raw_shape','detections_count','forward_and_nms_seconds')}), flush=True)
        result = dict(plan, status='AUTHOR_CHECKPOINT_FORWARD_COMPLETED', model_type=type(model).__module__+'.'+type(model).__name__,
                      parameter_count=sum(p.numel() for p in model.parameters()), model_load_seconds=loaded_seconds,
                      completed_pairs=len(items), elapsed_seconds=time.perf_counter()-start,
                      max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      cuda_initialized=torch.cuda.is_initialized(), training_started=False,
                      items=[{k:v for k,v in item.items() if k!='detections'} for item in items])
        save('receipt.json', result)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        save('failure.json', dict(status='FAILED', error=repr(exc), traceback=traceback.format_exc(),
                                 elapsed_seconds=time.perf_counter()-start))
        raise


if __name__ == '__main__':
    main()
