# OGSOD + SpaceNet6 D0-v1 Report

Frozen property: `P4`
Decision: `DEFER_CONTEXT_ONLY`

D0 is a data-property audit. It does not establish a detection-AP improvement and does not authorize training automatically.

## Property-fit selection

Eligible candidates: `P4`.
Weakest-dataset standardized scores: `{"P1": -0.1626533399396483, "P2": -0.1626533399396483, "P3": -0.1626533399396483, "P4": 0.44170102845337006}`.

## Frozen-property confirmation

- ogsod: context NMI effect 0.051169 (95% CI 0.048774, 0.053504); Sobel direction 0.091416; object-context residual -0.006955 (95% CI -0.009283, -0.004652).
- spacenet6: context NMI effect 0.105569 (95% CI 0.076678, 0.142570); Sobel direction 0.184731; object-context residual -0.080620 (95% CI -0.116435, -0.051679).

The paired identity signal is reproducible in local context, while object regions do not improve over matched context. This supports deferring a new object/local module; the existing global paired OS-SSL route already targets the observed property.

## Gate failures

- None

## Access boundary

Only property-fit/property-confirm rows derived from the prior training split were read. Method-dev and official evaluation data were not used.
