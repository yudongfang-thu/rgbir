"""Append a newly observed static landmark; preserve batch2 and accepted v1."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE/"primary_visual_observations_batch2.json").read_text())
data["protocol"] = "tc_geometry_primary_visual_batch3_v1"
data["coordinate_basis"] += " Rightmost bollard inspected again using an original-image crop with explicit coordinates, magnified 4x; top-center visibility in IR is less crisp and remains subject to independent review."
data["observations"][0]["points"].append(dict(id="g", feature="center of far-right short bollard top surface", rgb=[1198, 174], ir=[1197, 179]))
data["observations"][0]["interpretation"] = "Additional static point broadens spatial support. The original six points, uncertainty and thresholds are unchanged. This seventh point and the new hull require independent review before creating accepted contract v2."
target = HERE/"primary_visual_observations_batch3.json"
if target.exists():
    if json.loads(target.read_text()) != data:
        raise RuntimeError("Existing annotation differs; preserve it and create another version.")
else:
    target.write_text(json.dumps(data, indent=2)+"\n")
print(target)
