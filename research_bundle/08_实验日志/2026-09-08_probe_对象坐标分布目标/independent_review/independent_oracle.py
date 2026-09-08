"""Independent CPU reference: direct signed-distance affine + all-bin hat basis.

This file does not import the executor's operator. No content hashes are computed.
"""
from fractions import Fraction as F
import math


def rational(x):
    return F(x) if isinstance(x, (int, float)) else F(str(x))


def reference(probabilities, teacher_gt, student_gt, teacher_center,
              student_center, teacher_stride, student_stride):
    if any(x is None for x in (probabilities, teacher_gt, student_gt,
                              teacher_center, student_center, teacher_stride, student_stride)):
        return {"accepted": False, "reason": "missing"}
    if len(probabilities) != 4 or any(len(p) != 16 for p in probabilities):
        return {"accepted": False, "reason": "invalid_probability_shape"}
    if any(not math.isfinite(p) or p < 0 for row in probabilities for p in row):
        return {"accepted": False, "reason": "invalid_probability"}
    if any(abs(math.fsum(row) - 1) > 1e-6 for row in probabilities):
        return {"accepted": False, "reason": "invalid_probability_mass"}
    tg, sg = list(map(rational, teacher_gt)), list(map(rational, student_gt))
    tc, sc = list(map(rational, teacher_center)), list(map(rational, student_center))
    ts, ss = rational(teacher_stride), rational(student_stride)
    if ts <= 0 or ss <= 0 or any(g[i+2] <= g[i] for g in (tg,sg) for i in (0,1)):
        return {"accepted": False, "reason": "invalid_geometry"}
    distances, targets, coefficients, outside = [], [], [], []
    for side, probs in enumerate(probabilities):
        axis = side % 2
        sign = -1 if side < 2 else 1
        ratio = (sg[axis+2] - sg[axis]) / (tg[axis+2] - tg[axis])
        a = ratio * ts / ss
        b = sign * (sg[axis] + ratio*(tc[axis] - tg[axis]) - sc[axis]) / ss
        mapped = [a * k + b for k in range(16)]
        violated = [{"source_bin": j, "mass": probs[j], "mapped_distance": float(d)}
                    for j,d in enumerate(mapped) if probs[j] > 0 and not 0 <= d <= 15]
        coefficients.append([float(a), float(b)])
        distances.append([float(d) for d in mapped])
        outside.append(violated)
        # Hat basis over every destination bin avoids duplicating executor's scatter.
        q = [math.fsum(probs[j] * float(max(F(0), F(1)-abs(d-k)))
                       for j,d in enumerate(mapped) if probs[j] > 0)
             for k in range(16)]
        targets.append(q)
    accepted = not any(outside)
    return {"accepted": accepted, "reason": "accepted" if accepted else "positive_mass_out_of_support",
            "distances": distances, "targets": targets if accepted else None,
            "affine": coefficients, "outside": outside,
            "input_mass": [math.fsum(p) for p in probabilities],
            "mapped_first_moment": [math.fsum(p*d for p,d in zip(ps,ds))
                                    for ps,ds in zip(probabilities,distances)]}


def small_truth():
    unit = [[1/16]*16 for _ in range(4)]
    checks = {}
    same = reference(unit,[0,0,16,16],[0,0,16,16],[8,8],[8,8],1,1)
    checks["identity_complete_support"] = same["accepted"] and same["targets"] == unit
    translated = reference(unit,[0,0,16,16],[10,20,26,36],[8,8],[18,28],1,1)
    checks["joint_translation"] = translated["accepted"] and translated["targets"] == unit
    scaled = reference(unit,[0,0,16,16],[0,0,32,32],[8,8],[16,16],1,2)
    checks["scale_with_stride"] = scaled["accepted"] and scaled["targets"] == unit
    compressed = reference(unit,[0,0,16,16],[0,0,16,16],[8,8],[8,8],1,2)
    checks["half_bin_mass_and_expectation"] = compressed["accepted"] and all(
        abs(math.fsum(q)-1)<1e-14 and abs(math.fsum(i*p for i,p in enumerate(q))-3.75)<1e-14
        for q in compressed["targets"])
    point15 = [[0]*15+[1] for _ in range(4)]
    last = reference(point15,[0,0,16,16],[0,0,16,16],[8,8],[8,8],1,1)
    checks["last_bin_exact"] = last["accepted"] and last["targets"] == point15
    shifted = reference(unit,[0,0,16,16],[0,0,16,16],[8,8],[9,8],1,1)
    checks["positive_tail_rejected"] = not shifted["accepted"] and any(shifted["outside"])
    tiny = [[1e-300]+[0]*14+[1] for _ in range(4)]
    tinybad = reference(tiny,[0,0,16,16],[0,0,16,16],[8,8],[9,8],1,1)
    checks["tiny_positive_not_dropped"] = not tinybad["accepted"]
    zero_bad = reference(unit,[0,0,0,16],[0,0,16,16],[8,8],[8,8],1,1)
    checks["zero_width_rejected"] = not zero_bad["accepted"]
    checks["missing_rejected"] = not reference(None,[0,0,16,16],[0,0,16,16],[8,8],[8,8],1,1)["accepted"]
    return checks


if __name__ == "__main__":
    import json
    checks = small_truth()
    print(json.dumps({"independent_truth": checks,"passed":all(checks.values())},indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)
