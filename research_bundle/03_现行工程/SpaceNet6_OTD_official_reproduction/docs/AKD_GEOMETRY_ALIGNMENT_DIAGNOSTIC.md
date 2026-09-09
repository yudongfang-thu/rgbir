# AKD teacher-feature geometry audit

## Released behavior

The released AKD train pipeline has this order:

1. load SAR image;
2. load boxes, labels, and optical teacher features;
3. resize the SAR image and boxes to the configured 900-pixel scale;
4. horizontally flip the SAR image and boxes with probability 0.5;
5. format and collect the unchanged teacher features.

`RandomFlip` iterates over `img_fields`, `bbox_fields`, `mask_fields`, and
`seg_fields`. The five `kd_features_p2` through `kd_features_p6` arrays are in none
of those collections, so they remain in the unflipped optical coordinate frame.

## Tensor-layout evidence

The feature exporter saves each FPN output with its batch dimension. Actual files
from both frozen teacher roots were audited on gp94:

| Level | Saved layout | Loaded diagnostic layout |
|---|---:|---:|
| P2 | `(1,256,232,232)` | `(256,232,232)` |
| P3 | `(1,256,116,116)` | `(256,116,116)` |
| P4 | `(1,256,58,58)` | `(256,58,58)` |
| P5 | `(1,256,29,29)` | `(256,29,29)` |
| P6 | `(1,256,15,15)` | `(256,15,15)` |

All audited arrays are C-contiguous `float32` NCHW files. The diagnostic loader
removes exactly batch dimension 0 rather than calling unconstrained `squeeze()`.
For the resulting CHW maps, a horizontal reflection is `axis=-1`.

## Diagnostic implementation

`RandomFlipWithKD` delegates the random decision and ordinary image/target work to
the released `RandomFlip`, then mirrors all five CHW arrays if and only if the
realized direction is horizontal. `np.flip` creates a negative-stride view, so each
result is copied to contiguous storage before tensor conversion.

`LoadAnnotationsWithKDFeatureRoot` is separate from the released loader and accepts
the feature root directly from each diagnostic config. paper24 and public36 roots
therefore switch without data duplication.

## Boundaries

This implementation handles only the released horizontal-flip intervention. It
does not resample teacher maps during SAR resize because the saved optical FPN maps
already correspond to the 900-pixel teacher input used by the released AKD graph.
It does not add vertical/diagonal flips, alter AKD weights, regenerate features,
change SAR channel scaling, or modify existing R1/R2/R3 protocols.
