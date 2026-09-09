"""Run the original fixed-endpoint evaluator on the single GPU already leased."""
import argparse
import json
from pathlib import Path
import sys

REFERENCE = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/release_gpu5')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(REFERENCE))
    import torch
    from runtime import legacy
    import evaluate_independent as original
    lease = legacy.require_bound_lease_from_environment()
    # Old select_device('0') rewrites CUDA_VISIBLE_DEVICES. Initialize the
    # already bound visibility first, so logical device0 stays on its lease.
    torch.cuda.init()
    with (args.run/'evaluation_device_binding.json').open('x') as stream:
        json.dump(dict(physical_gpus=lease['gpus'], logical_device=torch.cuda.current_device(),
                       bound_context_initialized_before_original_selector=True), stream, indent=2)
    emitter = legacy.emit_bound_run_receipt

    def emit(**kwargs):
        kwargs['trainers'] = [*kwargs['trainers'], Path(__file__)]
        return emitter(**kwargs)

    legacy.emit_bound_run_receipt = emit
    try:
        original.run(argparse.Namespace(config=args.run/'protocol_config.yaml', run=args.run, attempt=1))
    finally:
        legacy.emit_bound_run_receipt = emitter


if __name__ == '__main__':
    main()
