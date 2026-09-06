"""Read-only CPU reproduction of the accepted frame's region rejection on 94."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-base', type=Path, default=BASE)
    args = parser.parse_args()
    sys.path.insert(0, str(args.artifact_base / 'release_v5'))
    from geometry_contract import GeometryContract
    source = args.artifact_base / 'd2_llvip_verified_attempt1'
    contract_path = args.artifact_base / 'geometry/review_evidence_v1/geometry_contract_accepted_exact_v1.json'
    images = [json.loads(line) for line in (source / 'images.jsonl').read_text().splitlines() if line.strip()]
    objects = [json.loads(line) for line in (source / 'd1_objects.jsonl').read_text().splitlines() if line.strip()]
    assert len(images) == 1
    boxes = np.asarray([row['gt_box_input'] for row in objects])
    contract = GeometryContract.load(contract_path)
    result = dict(source=str(source), geometry_contract=str(contract_path), input_boxes=boxes.tolist(), levels={})
    for stride in (8, 16):
        mask, reasons = contract.object_decisions([images[0]['rgb_path']], boxes,
            np.zeros(len(boxes), dtype=int), [images[0]['pair_info']], stride)
        result['levels'][str(stride)] = dict(mask=mask.tolist(), reasons=reasons)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
