# Fixed subset N/C0 direction check

Known-positive-control screener validation: one seed42, generic pretrained initialization, normal BN, fixed2048 train subset and independent E8 schedule. AP displays percent; differences are percentage points. No SD, new-method causal claim or automatic E200 admission.

|Arm|Status|mAP50-95 (%)|AP50 (%)|AP75 (%)|
|---|---|---:|---:|---:|
| N | COMPLETED | 27.588046 | 43.153525 | 31.457494 |
| C0 | COMPLETED | 26.459942 | 41.735800 | 30.036811 |

|Contrast|Status|delta mAP (pp)|delta AP50 (pp)|delta AP75 (pp)|
|---|---|---:|---:|---:|
| C0 - N | COMPUTED | -1.128105 | -1.417726 | -1.420683 |

Primary observed direction: negative. Missing/failed/incomplete/conflicting inputs are null, never zero. Per-class raw values, P/R and fixed C0-N differences are in summary.json.
Positive only indicates retention of this known control direction here. Nonpositive means this fixed screener check did not retain that direction; it does not establish C0 ineffectiveness. Neither outcome validates general method ranking.
