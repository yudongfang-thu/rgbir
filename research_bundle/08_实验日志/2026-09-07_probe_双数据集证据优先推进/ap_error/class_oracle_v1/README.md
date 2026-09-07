# Official TIDE class-oracle follow-up

Descriptive follow-up requested after the frozen six-endpoint result exposed a large macro Cls contribution. No thresholds, model selection or training were changed. Each row is the same official independent Cls oracle decomposed into per-class AP differences; these are not achievable KD gains.

|IoU|Arm|Class|GT|AP mean±SD|Cls oracle dAP mean±SD|C0−N dAP seeds 0/42/123|
|---|---|---|---:|---:|---:|---|
|0.5|N|car|18965|94.4393 ± 0.0190|0.5743 ± 0.0074|-0.0058, -0.0116, +0.0076|
|0.5|N|freight car|710|60.4184 ± 0.6018|18.2118 ± 1.9514|+0.1105, +1.5596, -1.4823|
|0.5|N|truck|1336|72.3509 ± 0.1933|9.0609 ± 0.3324|+0.8098, +0.2929, +0.8082|
|0.5|N|bus|751|94.2090 ± 0.5166|2.3977 ± 0.1493|+0.4115, -0.0294, -0.2942|
|0.5|N|van|700|62.6796 ± 0.4251|21.5537 ± 0.7945|-1.1961, -0.6250, +1.9354|
|0.5|C0|car|18965|94.5766 ± 0.2236|0.5710 ± 0.0167|-0.0058, -0.0116, +0.0076|
|0.5|C0|freight car|710|60.7637 ± 1.2662|18.2744 ± 0.4732|+0.1105, +1.5596, -1.4823|
|0.5|C0|truck|1336|72.0729 ± 0.3477|9.6978 ± 0.2754|+0.8098, +0.2929, +0.8082|
|0.5|C0|bus|751|94.1155 ± 0.2534|2.4270 ± 0.4145|+0.4115, -0.0294, -0.2942|
|0.5|C0|van|700|62.9643 ± 0.3140|21.5918 ± 0.8803|-1.1961, -0.6250, +1.9354|
|0.75|N|car|18965|75.9482 ± 0.2145|0.3483 ± 0.0134|-0.0203, -0.0217, -0.0046|
|0.75|N|freight car|710|44.5629 ± 0.9751|11.2982 ± 2.0130|+0.7627, +1.3869, -1.5582|
|0.75|N|truck|1336|59.4553 ± 0.8406|5.6099 ± 0.3990|+0.0087, -0.1758, +0.7415|
|0.75|N|bus|751|87.6272 ± 0.6927|1.7181 ± 0.1961|+0.2630, +0.1420, -0.4339|
|0.75|N|van|700|51.4821 ± 0.5495|12.1184 ± 0.6252|-0.8470, +0.6733, +1.4753|
|0.75|C0|car|18965|75.6409 ± 0.2970|0.3327 ± 0.0204|-0.0203, -0.0217, -0.0046|
|0.75|C0|freight car|710|46.2861 ± 1.8197|11.4954 ± 0.4979|+0.7627, +1.3869, -1.5582|
|0.75|C0|truck|1336|58.5099 ± 0.3165|5.8014 ± 0.3084|+0.0087, -0.1758, +0.7415|
|0.75|C0|bus|751|87.7510 ± 0.6449|1.7085 ± 0.1833|+0.2630, +0.1420, -0.4339|
|0.75|C0|van|700|51.6071 ± 0.7692|12.5523 ± 0.6176|-0.8470, +0.6733, +1.4753|

Matrices in JSON: predicted class is row, associated true class is column. The matrices describe TIDE Cls error predictions, including extra cross-class post-NMS detections; they are not a single-label classification confusion matrix of GT objects. Cls oracle may remove an erroneous detection if its GT is already detected and may repair one best erroneous prediction otherwise.

A macro AP bottleneck establishes where errors affect this evaluator. It does not establish that the IR teacher supplies correct transferable knowledge, or that the RGB student can learn the oracle.
