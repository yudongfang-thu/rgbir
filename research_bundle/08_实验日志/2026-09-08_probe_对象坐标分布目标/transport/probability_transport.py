"""Complete T=1 DFL probability transport in annotation-defined object coordinates."""
import math
from fractions import Fraction

SIDES=('left','top','right','bottom')
BINS=16

def finite_number(x):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):
        raise ValueError('nonfinite_or_non_numeric_input')
    return x

def fraction(x):return Fraction(finite_number(x))

def stable_softmax(logits):
    if len(logits)!=BINS:raise ValueError('expected_sixteen_logits')
    values=[float(finite_number(x)) for x in logits];maximum=max(values)
    exponentials=[math.exp(x-maximum) for x in values]
    if any(x==0 for x in exponentials):raise ValueError('positive_softmax_mass_underflow')
    denominator=math.fsum(exponentials)
    probabilities=[x/denominator for x in exponentials]
    if any(x<=0 or not math.isfinite(x) for x in probabilities):raise ValueError('positive_softmax_mass_underflow')
    return probabilities

def affine_coefficients(teacher_gt,student_gt,teacher_center,student_center,teacher_stride,student_stride):
    if len(teacher_gt)!=4 or len(student_gt)!=4 or len(teacher_center)!=2 or len(student_center)!=2:
        raise ValueError('invalid_geometry_shape')
    tg,sg,tc,sc=[[fraction(x) for x in v] for v in (teacher_gt,student_gt,teacher_center,student_center)]
    ts,ss=fraction(teacher_stride),fraction(student_stride)
    if ts<=0 or ss<=0:raise ValueError('nonpositive_stride')
    if any(gt[axis+2]<=gt[axis] for gt in (tg,sg) for axis in (0,1)):raise ValueError('nonpositive_GT_extent')
    result=[]
    for side in range(4):
        axis=side%2;sign=-1 if side<2 else 1
        scale=(sg[axis+2]-sg[axis])/(tg[axis+2]-tg[axis])
        a=ts*scale/ss
        b=sign*(sg[axis]+(tc[axis]-tg[axis])*scale-sc[axis])/ss
        result.append((a,b))
    return result

def scatter_complete(probabilities,a,b):
    if len(probabilities)!=BINS or any(not math.isfinite(p) or p<0 for p in probabilities):raise ValueError('invalid_probability_vector')
    mapped=[a*j+b for j in range(BINS)]
    rejected=[j for j,(p,d) in enumerate(zip(probabilities,mapped)) if p>0 and not 0<=d<=BINS-1]
    base=dict(affine_a=dict(numerator=a.numerator,denominator=a.denominator),
        affine_b=dict(numerator=b.numerator,denominator=b.denominator),identity=(a==1 and b==0),
        mapped_distance_bins=[float(d) for d in mapped],rejected_source_bins=rejected,
        rejected_positive_mass=math.fsum(probabilities[j] for j in rejected),source_mass=math.fsum(probabilities),
        source_expectation_bin=math.fsum(j*p for j,p in enumerate(probabilities)),
        mapped_expectation_bin=math.fsum(float(d)*p for d,p in zip(mapped,probabilities)))
    if rejected:return dict(base,status='OUTSIDE_STUDENT_SUPPORT',target_probabilities=None)
    terms=[[] for _ in range(BINS)]
    for p,d in zip(probabilities,mapped):
        if p==0:continue
        lower=d.numerator//d.denominator;part=d-lower
        if part==0:terms[lower].append(p)
        else:
            terms[lower].append(p*float(1-part));terms[lower+1].append(p*float(part))
    target=[math.fsum(t) for t in terms];mass=math.fsum(target)
    expectation=math.fsum(j*p for j,p in enumerate(target))
    mass_error=mass-base['source_mass'];mean_error=expectation-base['mapped_expectation_bin']
    if not math.isclose(mass,base['source_mass'],rel_tol=0,abs_tol=1e-12):raise ArithmeticError('mass_closure_failure')
    if not math.isclose(expectation,base['mapped_expectation_bin'],rel_tol=1e-12,abs_tol=1e-12):raise ArithmeticError('expectation_closure_failure')
    return dict(base,status='SUPPORTED',target_probabilities=target,target_mass=mass,target_expectation_bin=expectation,
        mass_closure_error=mass_error,expectation_closure_error_bin=mean_error)

def transport_logits(logits,teacher_gt,student_gt,teacher_center,student_center,teacher_stride,student_stride):
    """Return all four transported sides, or an explicit whole-object rejection."""
    try:
        if len(logits)!=4:raise ValueError('expected_four_sides')
        probabilities=[stable_softmax(side) for side in logits]
        affine=affine_coefficients(teacher_gt,student_gt,teacher_center,student_center,teacher_stride,student_stride)
    except (ValueError,TypeError,OverflowError) as exc:
        return dict(status='INVALID_INPUT',reason=str(exc),target_probabilities=None)
    sides=[scatter_complete(p,a,b) for p,(a,b) in zip(probabilities,affine)]
    supported=all(s['status']=='SUPPORTED' for s in sides)
    for side,result in enumerate(sides):
        axis=side%2;sign=-1 if side<2 else 1
        scale=(student_gt[axis+2]-student_gt[axis])/(teacher_gt[axis+2]-teacher_gt[axis])
        original_edge=teacher_center[axis]+sign*teacher_stride*result['source_expectation_bin']
        mapped_edge=student_gt[axis]+(original_edge-teacher_gt[axis])*scale
        via_distance=student_center[axis]+sign*student_stride*result['mapped_expectation_bin']
        result['side']=SIDES[side];result['mapped_edge_coordinate']=mapped_edge
        result['edge_from_mapped_expectation']=via_distance;result['edge_closure_error']=via_distance-mapped_edge
        if not math.isclose(mapped_edge,via_distance,rel_tol=1e-12,abs_tol=1e-10):raise ArithmeticError('edge_coordinate_closure_failure')
    # No partial four-side target is exposed for an unsupported object.
    target=[side['target_probabilities'] for side in sides] if supported else None
    return dict(status='SUPPORTED' if supported else 'OUTSIDE_STUDENT_SUPPORT',temperature=1,
        source_logits=logits,source_probabilities=probabilities,sides=sides,target_probabilities=target,
        algebraic_identity=all(a==1 and b==0 for a,b in affine),clamped=False,truncated=False,renormalized=False)
